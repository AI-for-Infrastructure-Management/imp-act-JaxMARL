import jax
import jax.numpy as jnp
import numpy as np
from numbers import Number
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import csv

# Get Hydra's logger
log = logging.getLogger(__name__)

DEBUG = False
if DEBUG:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_check_tracer_leaks", True)

################################################################################
# 1) Define the parametric heuristic policy
################################################################################
def parametric_heuristic_policy(interval, threshold):
    """
    Return a policy function that uses `interval` and `threshold`.
    This "factory" returns a closure capturing those parameters.
    """
    def policy(key, state, obs, env=None):
        tstep = state.env_state.timestep
        obs_insp = state.env_state.observation

        # Step 1: default action = 0
        actions = jnp.zeros_like(obs_insp, dtype=jnp.int32)

        # Step 2: if tstep % interval == 0 => action 1
        actions = jnp.where(tstep % interval == 0, 1, actions)

        # Step 3: if obs_insp > threshold => action 2 (priority)
        actions = jnp.where(obs_insp > threshold, 2, actions)

        actions_dict = {f"agent_{i}": actions[i] for i in range(env.num_agents)}
        return actions_dict
    return policy


################################################################################
# Budget prioritization policy wrapper
################################################################################
def get_budget_prioritized_policy(policy, params):
    """Returns a prioritized policy function with configurable parameters.
    
    Args:
        policy: The base policy function to wrap
        params: Dictionary containing prioritization configuration:
            - prioritization_key: Strategy for prioritization ("cost", "segment_lengths", "volumes", "random", "list")
            - prioritization_sign: "negative" to reverse prioritization order
            - random_seed: Seed for random prioritization (or "random" for a random seed)
    """
    if params.get("prioritization_key") == "random":
        seed = params.get("random_seed")
        if seed == "random":
            seed = np.random.randint(0, 2**32 - 1)
            prio_key = jax.random.PRNGKey(seed)
        else:
            prio_key = jax.random.PRNGKey(seed)
    
    if params.get("prioritization_key") == "list":
        prioritization_list = jnp.array(params.get("prioritization_list"))
        if len(prioritization_list) == 0:
            raise ValueError("Prioritization list cannot be empty when using 'list' key.")

    def prioritized_policy(key, state, obs, env):
        """Policy that prioritizes actions based on configured strategy within budget constraints."""
        road_env = env._env.env
        road_env_state = state.env_state

        action = policy(key, state, obs, env)

        # Convert dict to array for processing
        action_arr = jnp.array([action[f"agent_{i}"] for i in range(env.num_agents)])

        # Step 4: Prioritize repair actions
        forced_action, forced_repair_mask = road_env._apply_forced_repair_constraint(
            action_arr, road_env_state.worst_obs_counter
        )

        do_nothing_action = jnp.zeros_like(action_arr)

        action_arr = jnp.where(
            forced_repair_mask,
            do_nothing_action,
            action_arr,
        )

        # Make sure real damage state cannot be used as info
        road_env_state = road_env_state.replace(
            damage_state = jnp.zeros_like(road_env_state.damage_state),
        )
        
        do_nothing_forced_repair_mask = jnp.full_like(forced_repair_mask, False)
        upfront_cost = road_env._get_budget_action_cost(
            road_env_state, do_nothing_action, do_nothing_forced_repair_mask
        )
        future_upfront_cost = upfront_cost * (
            road_env.get_budget_remaining_time(road_env_state.timestep) - 1
        )

        # Calculate adjusted costs
        action_cost = road_env._get_budget_action_cost(road_env_state, action_arr, forced_repair_mask)
        adjusted_cost = action_cost - upfront_cost

        remaining_budget = (
            road_env_state.budget_remaining
            - jnp.sum(upfront_cost)
            - jnp.sum(future_upfront_cost)
        )

        # Apply constraints if needed
        def apply_constraints():
            # Select actions based on most effective cost-benefit ratio (negative due to )
            if params["prioritization_key"] == "cost":
                priorities = adjusted_cost
            elif params["prioritization_key"] == "segment_lengths":
                priorities = road_env.segment_lengths
            elif params["prioritization_key"] == "volumes":
                priorities = road_env.initial_edge_volumes
            elif params["prioritization_key"] == "random":
                priorities = jax.random.uniform(prio_key, shape=action_arr.shape)
            elif params["prioritization_key"] == "list":
                if len(prioritization_list) != len(action_arr):
                    raise ValueError("Length of prioritization list must match number of agents.")
                priorities = prioritization_list
            else:
                raise ValueError(f"Unknown prioritization key: {params['prioritization_key']}")
            
            if params.get("prioritization_sign") == "negative":
                priorities = -priorities

            # Don't constrain forced repairs
            priorities = jnp.where(forced_repair_mask, -jnp.inf, priorities)
            sorted_indices = jnp.argsort(priorities, descending=True)
            cumulative_costs = jnp.cumsum(adjusted_cost[sorted_indices])
            valid_mask = cumulative_costs <= remaining_budget

            # Create array of constrained actions in original order
            constrained_action = jnp.zeros_like(action_arr)
            constrained_action = constrained_action.at[sorted_indices].set(
                jnp.where(
                    valid_mask,
                    action_arr[sorted_indices],
                    do_nothing_action[sorted_indices],
                )
            )

            return constrained_action, True

        constrained_action, constraint_applied = jax.lax.cond(
            jnp.sum(adjusted_cost) > remaining_budget,
            lambda: apply_constraints(),
            lambda: (action_arr, False),
        )

        top_k_param = params.get("top_k")

        if top_k_param is not None:
            def apply_top_k_constraint(top_k_value):
                top_k_scalar = jnp.asarray(top_k_value)
                top_k_scalar = jnp.maximum(jnp.round(top_k_scalar).astype(jnp.int32), 0)

                if params["prioritization_key"] == "cost":
                    priorities = adjusted_cost
                elif params["prioritization_key"] == "segment_lengths":
                    priorities = road_env.segment_lengths
                elif params["prioritization_key"] == "volumes":
                    priorities = road_env.initial_edge_volumes
                elif params["prioritization_key"] == "random":
                    priorities = jax.random.uniform(prio_key, shape=action_arr.shape)
                elif params["prioritization_key"] == "list":
                    if len(prioritization_list) != len(action_arr):
                        raise ValueError("Length of prioritization list must match number of agents.")
                    priorities = prioritization_list
                else:
                    raise ValueError(f"Unknown prioritization key: {params['prioritization_key']}")

                if params.get("prioritization_sign") == "negative":
                    priorities = -priorities

                priorities = jnp.where(forced_repair_mask, -jnp.inf, priorities)
                sorted_indices = jnp.argsort(priorities, descending=True)
                rank_positions = jnp.arange(sorted_indices.shape[0])
                top_k_mask = rank_positions < top_k_scalar
                selection_mask = jnp.zeros_like(forced_repair_mask, dtype=bool)
                selection_mask = selection_mask.at[sorted_indices].set(top_k_mask)

                return jnp.where(
                    selection_mask,
                    constrained_action,
                    do_nothing_action,
                )

            remaining_time = road_env.get_budget_remaining_time(road_env_state.timestep)

            constrained_action = jax.lax.cond(
                remaining_time > 1,
                lambda value: apply_top_k_constraint(value),
                lambda value: constrained_action,
                top_k_param,
            )

        actions_dict = {f"agent_{i}": constrained_action[i] for i in range(len(obs))}
        return actions_dict
    return prioritized_policy


################################################################################
# 2) Run a single rollout with parametric policy
################################################################################
def run_rollout_with_params(key, env, numeric_params, non_jitted_params, num_steps, config):
    """Build the heuristic policy from generic parameter dicts and run one rollout.

    Args:
        key: JAX PRNGKey for this episode.
        env: Wrapped JaxMARL road environment.
        numeric_params: Dict of numeric parameters for the policy (jit-safe).
    non_jitted_params: Dict of non-jitted parameters (handled outside jit).
        num_steps: Episode length.
        config: Hydra config (used e.g. for other policy options).
    """

    interval = numeric_params.get("inspection_interval", config.policy_parameters.get("inspection_interval"))
    threshold = numeric_params.get("repair_threshold", config.policy_parameters.get("repair_threshold"))

    policy = parametric_heuristic_policy(interval, threshold)

    # Prioritization and other policy settings come from config + non-jitted params
    policy_params = dict(config.policy_parameters)
    policy_params.update(non_jitted_params)
    policy_params.update(numeric_params)

    if policy_params.get("prioritization_enabled", False):
        policy = get_budget_prioritized_policy(policy, policy_params)

    return run_rollout(key, env, policy, num_steps)


################################################################################
# 3) Rollout function
################################################################################
def run_rollout(key, env, policy, num_steps):
    """Run a single rollout in the environment."""
    def scan_step(carry, _):
        key, total_reward, last_obs, last_state, episodes = carry
        key, key_act = jax.random.split(key)
        actions = policy(key_act, last_state, last_obs, env)
        key, key_step = jax.random.split(key)
        obs, state, reward, done, infos = env.step(key_step, last_state, actions)

        total_reward = total_reward + reward["__all__"]
        episodes = jax.lax.cond( 
            done["__all__"].any(),
            lambda: episodes + 1,
            lambda: episodes,
        )
        return (key, total_reward, obs, state, episodes), (done["__all__"], infos)
    
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset)
    
    key, key_scan = jax.random.split(key)
    carry, (dones, infos) = jax.lax.scan(
        scan_step,
        (key_scan, 0.0, obs, state, 0),
        None,
        length=num_steps
    )
    _, total_reward, _, _, episodes = carry

    log_wrapper_return = jnp.nanmean(
        jnp.where(
            infos["returned_episode"],
            infos["returned_episode_returns"],
            jnp.nan,
        )
    )
    return total_reward / episodes, dones.any(), log_wrapper_return


################################################################################
# 4) Main: build the parameter grid, vmap over combos & episodes, compute stats
################################################################################
@hydra.main(config_path="config/heuristics", config_name="toy_example_segment_based_heuristic", version_base=None)
def main(cfg: DictConfig):
    print(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.to_container(cfg))}")

    # Get Hydra output directory for this run (if available)
    try:
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_output_dir = None

    if cfg.get("double_precision_mode", False):
        jax.config.update("jax_enable_x64", True)
        log.info("Using double precision mode")

    # Extract configuration
    MAP_NAME = cfg.map
    NUM_EPISODES = cfg.num_episodes
    NUM_STEPS = cfg.num_steps
    NORM_CONSTANT = cfg.norm_constant
    CHUNK_SIZE = cfg.optimization.chunk_size
    plot_hist = cfg.plot_histogram

    bool_types = (bool, np.bool_)

    def _is_numeric_value(value):
        return isinstance(value, (Number, np.number)) and not isinstance(value, bool_types)

    policy_top_k = None
    if hasattr(cfg, "policy_parameters") and cfg.policy_parameters is not None:
        policy_top_k = cfg.policy_parameters.get("top_k", None)
    if policy_top_k is not None and not _is_numeric_value(policy_top_k):
        raise ValueError(
            "policy_parameters.top_k must be numeric; remove the key altogether to disable top-k prioritization."
        )

    # ---------------------------------------------------------------------
    # Build parameter grids grouped by whether they can be jitted.
    # "Jitted" parameters participate in the vmapped/jitted inner loop.
    # Non-jitted ones are iterated in Python outside the compiled region.
    # ---------------------------------------------------------------------
    jitted_param_grids = {}
    non_jitted_param_lists = {}

    for name, p_cfg in cfg.optimization.parameters.items():
        if "range" in p_cfg:
            r = p_cfg["range"]
            start = float(r.get("from", 0.0))
            stop = float(r["to"])
            step = float(r.get("step", 1.0))
            if step <= 0:
                raise ValueError(f"Parameter '{name}': range step must be positive.")
            count = int(np.floor((stop - start) / step)) + 1
            if count <= 0:
                raise ValueError(
                    f"Parameter '{name}': invalid range definition (from={start}, to={stop}, step={step})."
                )
            values = [start + step * i for i in range(count)]
        elif "list" in p_cfg:
            values = list(p_cfg["list"])
        else:
            raise ValueError(
                f"Parameter '{name}' must define either a 'range' or a 'list'."
            )

        if len(values) == 0:
            raise ValueError(f"Parameter '{name}' must provide at least one value.")

        numeric_values_only = all(_is_numeric_value(v) for v in values)
        if name == "top_k" and not numeric_values_only:
            raise ValueError(
                "Optimization parameter 'top_k' accepts only numeric values; remove it entirely to disable top-k prioritization."
            )

        jittable_flag = p_cfg.get("jittable")
        all_numeric = numeric_values_only
        if jittable_flag is None:
            can_jit = all_numeric
        else:
            can_jit = bool(jittable_flag)
            if can_jit and not all_numeric:
                raise ValueError(
                    f"Parameter '{name}' is marked jittable but has non-numeric entries."
                )

        if can_jit:
            jitted_param_grids[name] = jnp.array(values)
        else:
            non_jitted_param_lists[name] = values

    if jitted_param_grids:
        log.info("Jitted optimization parameters:")
        for k, v in jitted_param_grids.items():
            log.info(f"  {k}: {v}")
    if non_jitted_param_lists:
        log.info("Non-jitted optimization parameters:")
        for k, v in non_jitted_param_lists.items():
            log.info(f"  {k}: {v}")

    # Build meshgrid over all jitted parameter ranges
    jitted_names = list(jitted_param_grids.keys())
    if jitted_names:
        jitted_arrays = [jitted_param_grids[n] for n in jitted_names]
        mesh = jnp.meshgrid(*jitted_arrays, indexing="ij")
        stacked = jnp.stack(mesh, axis=-1)
        jitted_combos = stacked.reshape(-1, len(jitted_names))
    else:
        # No jitted params -> single dummy combo
        jitted_combos = jnp.zeros((1, 0))

    num_jitted_combos = jitted_combos.shape[0]

    # Now we need random keys for each (jitted_combo, episode).
    # So in total, we need num_jitted_combos * NUM_EPISODES distinct keys.
    key = jax.random.PRNGKey(0)
    all_keys = jax.random.split(key, num_jitted_combos * NUM_EPISODES)
    # Reshape to (num_jitted_combos, NUM_EPISODES).
    all_keys = all_keys.reshape(num_jitted_combos, NUM_EPISODES, 2)

    env = make('road_env', map_name=MAP_NAME)
    env = LogWrapper(env)

    # We'll define a factory that, for fixed non-jitted params, returns a jitted
    # function running all episodes over all jitted combos.
    def make_run_chunk(non_jitted_params):
        def run_episodes_for_jitted_combo(jitted_combo, keys_for_episodes):
            """Map jitted combo to dict and run all episodes for given params."""

            def rollout_one_episode(k, combo_vals):
                if jitted_names:
                    jitted_dict = {name: combo_vals[i] for i, name in enumerate(jitted_names)}
                else:
                    jitted_dict = {}
                return run_rollout_with_params(
                    k, env, jitted_dict, non_jitted_params, NUM_STEPS, cfg
                )

            return jax.vmap(rollout_one_episode, in_axes=(0, None))(keys_for_episodes, jitted_combo)

        return jax.jit(
            jax.vmap(
                run_episodes_for_jitted_combo,
                in_axes=(0, 0),
            )
        )

    # shape of results per categorical combination:
    # (num_jitted_combos, NUM_EPISODES, 3) because each rollout returns
    # (total_reward, done_any, log_wrapper_return)
    start_time = time.time()
    # Outer loop over all combinations of non-jitted parameters in Python
    from itertools import product

    non_jitted_param_names = list(non_jitted_param_lists.keys())
    non_jitted_param_values = [non_jitted_param_lists[n] for n in non_jitted_param_names]

    all_results = []
    all_meta = []

    if not non_jitted_param_names:
        non_jitted_param_combos = [{}]
    else:
        non_jitted_param_combos = [
            dict(zip(non_jitted_param_names, vals))
            for vals in product(*non_jitted_param_values)
        ]

    for s_idx, s_params in enumerate(non_jitted_param_combos):
        log.info(
            f"Evaluating non-jitted parameter set {s_idx+1}/{len(non_jitted_param_combos)}: {s_params}"
        )

        run_chunk = make_run_chunk(s_params)

        results_list = []
        num_chunks = (num_jitted_combos + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in tqdm(range(num_chunks), desc="Running chunked vmap over jitted combos"):
            start = i * CHUNK_SIZE
            end = min((i + 1) * CHUNK_SIZE, num_jitted_combos)
            combos_chunk = jitted_combos[start:end]
            keys_chunk = all_keys[start:end]
            results_chunk = run_chunk(combos_chunk, keys_chunk)
            results_list.append(results_chunk)

        results_s = tuple(jnp.concatenate(r, axis=0) for r in zip(*results_list))
        all_results.append(results_s)
        all_meta.append({"non_jitted_params": s_params})

    # For simplicity, if multiple non-jitted param sets exist, we now concatenate
    # jitted combos along axis 0 and treat each (non-jitted, jitted) combo as
    # a separate entry when ranking.
    rewards_list = []
    dones_list = []
    logs_list = []
    meta_expanded = []

    for s_idx, (rewards_s, dones_s, logs_s) in enumerate(all_results):
    # rewards_s: (num_jitted_combos, NUM_EPISODES)
        rewards_list.append(rewards_s)
        dones_list.append(dones_s)
        logs_list.append(logs_s)
        for n_idx in range(rewards_s.shape[0]):
            meta_expanded.append({
                "non_jitted_params": all_meta[s_idx]["non_jitted_params"],
                "jitted_index": n_idx,
            })

    rewards = jnp.concatenate(rewards_list, axis=0) if rewards_list else jnp.zeros((0, NUM_EPISODES))
    dones = jnp.concatenate(dones_list, axis=0) if dones_list else jnp.zeros((0, NUM_EPISODES), dtype=bool)
    logs = jnp.concatenate(logs_list, axis=0) if logs_list else jnp.zeros((0, NUM_EPISODES))

    end_time = time.time()
    log.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    # Compute mean & std across episodes, for each full combo
    mean_rewards = jnp.mean(rewards, axis=1)  # shape (num_total_combos,)
    std_rewards  = jnp.std(rewards, axis=1)
    mean_logs = jnp.mean(logs, axis=1)
    std_logs  = jnp.std(logs, axis=1)
    any_dones_per_combo = jnp.any(dones, axis=1)  # shape (num_total_combos,)

    # Identify the best combo based on highest mean reward
    best_idx = jnp.argmax(mean_rewards)
    best_jitted_idx = meta_expanded[best_idx]["jitted_index"]
    best_jitted_vals = jitted_combos[best_jitted_idx]
    best_jitted_params = {name: float(best_jitted_vals[i]) for i, name in enumerate(jitted_names)}
    best_non_jitted_params = meta_expanded[best_idx]["non_jitted_params"]
    best_reward = mean_rewards[best_idx]
    
    log.info(
    f"Best jitted params: {best_jitted_params}, non-jitted params: {best_non_jitted_params} "
        f"with mean reward = {best_reward / NORM_CONSTANT:.6f} "
        f"and std = {std_rewards[best_idx] / NORM_CONSTANT:.6f}"
    )

    # Optionally print out the top few combos
    sorted_indices = jnp.argsort(-mean_rewards)  # descending
    top5 = sorted_indices[:5]
    log.info(f"Top 5 combos by mean reward:")
    for rank, idx in enumerate(top5, start=1):
        jitted_idx = meta_expanded[idx]["jitted_index"]
        jitted_vals = jitted_combos[jitted_idx]
        jitted_params = {name: float(jitted_vals[i]) for i, name in enumerate(jitted_names)}
        nj_params = meta_expanded[idx]["non_jitted_params"]
        log.info(
            f"Rank {rank}: jitted_params={jitted_params}, non_jitted_params={nj_params}, "
            f"mean_reward={mean_rewards[idx]/NORM_CONSTANT:.6f}"
        )
        if not any_dones_per_combo[idx]:
            log.warning(f"Warning: No episodes ended in a done state for this combo.")
        if not jnp.allclose(mean_rewards[idx], mean_logs[idx]):
            log.warning(f"Warning: Mean results from the heuristic policy and log wrapper do not match.")
        if not jnp.allclose(std_rewards[idx], std_logs[idx]):
            log.warning(f"Warning: Std results from the heuristic policy and log wrapper do not match.")

    # Save full results as CSV if Hydra output dir is available
    if hydra_output_dir is not None and rewards.shape[0] > 0:

        csv_path = os.path.join(hydra_output_dir, "heuristic_optimization_results.csv")
        log.info(f"Saving full optimization results to {csv_path}")

        # Column groups: non-jitted params, jitted params, metrics
        non_jitted_names = list(non_jitted_param_names)
        jitted_names_local = list(jitted_names)

        header = non_jitted_names + jitted_names_local + [
            "mean_reward",
            "std_reward",
        ]

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for idx in range(rewards.shape[0]):
                meta = meta_expanded[idx]
                nj_params = meta["non_jitted_params"]
                jitted_idx = meta["jitted_index"]
                jitted_vals = jitted_combos[jitted_idx]

                row = []
                for name in non_jitted_names:
                    row.append(nj_params.get(name))
                for i, name in enumerate(jitted_names_local):
                    row.append(float(jitted_vals[i]))

                row.extend([
                    float(mean_rewards[idx]),
                    float(std_rewards[idx]),
                ])

                writer.writerow(row)

    # If you'd like a histogram for a single combo or the best combo:
    if plot_hist:
        plt.hist(rewards[best_idx], bins=20)
        plt.title(f"Histogram of Rewards for Best Combo")
        plt.xlabel("Total reward")
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":
    main()