import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

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
            - priorization_key: Strategy for prioritization ("cost", "segment_lengths", "volumes", "random")
            - priorization_sign: "negative" to reverse prioritization order
            - random_seed: Seed for random prioritization (or "random" for a random seed)
    """
    if params.get("priorization_key") == "random":
        seed = params.get("random_seed")
        if seed == "random":
            seed = np.random.randint(0, 2**32 - 1)
            prio_key = jax.random.PRNGKey(seed)
        else:
            prio_key = jax.random.PRNGKey(seed)

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
            if params["priorization_key"] == "cost":
                priorities = adjusted_cost
            elif params["priorization_key"] == "segment_lengths":
                priorities = road_env.segment_lengths
            elif params["priorization_key"] == "volumes":
                priorities = road_env.initial_edge_volumes
            elif params["priorization_key"] == "random":
                priorities = jax.random.uniform(prio_key, shape=action_arr.shape)
            else:
                raise ValueError(f"Unknown priorization key: {params['priorization_key']}")
            
            if params.get("priorization_sign") == "negative":
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

        actions_dict = {f"agent_{i}": constrained_action[i] for i in range(len(obs))}
        return actions_dict
    return prioritized_policy


################################################################################
# 2) Run a single rollout with parametric policy
################################################################################
def run_rollout_with_params(key, env, interval, threshold, seed, num_steps, config):
    """
    Given specific (interval, threshold, seed), build the correct policy, then run a single rollout.
    """
    policy = parametric_heuristic_policy(interval, threshold)
    
    if hasattr(env._env.env, "_get_budget_action_cost") and config.get("priorization_params") is not None:
        prio_params = dict(config.priorization_params)
        if prio_params.get("priorization_key") == "random":
            prio_params["random_seed"] = seed
        policy = get_budget_prioritized_policy(policy, prio_params)
        
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
@hydra.main(config_path="config/heuristics", config_name="toy_example_humble_heuristic", version_base=None)
def main(cfg: DictConfig):
    print(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.to_container(cfg))}")

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
        
    # Extract optimization parameters
    intervals = jnp.arange(cfg.optimization.interval_start, cfg.optimization.interval_end)
    thresholds = jnp.arange(cfg.optimization.threshold_start, cfg.optimization.threshold_end)
    
    # Add seed range if configured
    if hasattr(cfg.optimization, "seed_start") and hasattr(cfg.optimization, "seed_end"):
        seeds = jnp.arange(cfg.optimization.seed_start, cfg.optimization.seed_end)
        log.info(f"Seeds range: {cfg.optimization.seed_start} to {cfg.optimization.seed_end-1}")
        # Create 3D grid with intervals, thresholds, and seeds
        i_grid, t_grid, s_grid = jnp.meshgrid(intervals, thresholds, seeds, indexing="ij")
        combos = jnp.stack([i_grid, t_grid, s_grid], axis=-1)
        log.info(f"Total parameter combinations: {len(intervals) * len(thresholds) * len(seeds)}")
    else:
        log.info(f"Intervals range: {cfg.optimization.interval_start} to {cfg.optimization.interval_end-1}")
        log.info(f"Thresholds range: {cfg.optimization.threshold_start} to {cfg.optimization.threshold_end-1}")
        # Create 2D grid with just intervals and thresholds
        i_grid, t_grid = jnp.meshgrid(intervals, thresholds, indexing="ij")
        combos = jnp.stack([i_grid, t_grid], axis=-1)
        log.info(f"Total parameter combinations: {len(intervals) * len(thresholds)}")
    
    # Flatten to (num_combos, param_count)
    combos = combos.reshape(-1, combos.shape[-1])
    num_combos = combos.shape[0]

    # Now we need random keys for each (combo, episode).
    # So in total, we need num_combos * NUM_EPISODES distinct keys.
    key = jax.random.PRNGKey(0)
    all_keys = jax.random.split(key, num_combos * NUM_EPISODES)
    # Reshape to (num_combos, NUM_EPISODES).
    all_keys = all_keys.reshape(num_combos, NUM_EPISODES, 2)

    env = make('road_env', map_name=MAP_NAME)
    env = LogWrapper(env)

    # We'll define a function that, for one combo, runs all episodes with vmap.
    def run_episodes_for_combo(combo, keys_for_episodes):
        """
        combo = [interval, threshold, (optional)seed] shape (2,) or (3,)
        keys_for_episodes = shape (NUM_EPISODES, 2) 
        """
        interval, threshold = combo[0], combo[1]
        # Use the seed from the combo if available, otherwise use the key
        seed = combo[2] if combo.shape[0] > 2 else keys_for_episodes[0, 0]

        # For each episode, run one rollout
        def rollout_one_episode(k):
            return run_rollout_with_params(k, env, interval, threshold, seed, NUM_STEPS, cfg)
        
        return jax.vmap(rollout_one_episode)(keys_for_episodes)

    run_chunk = jax.jit(
                jax.vmap(
                    run_episodes_for_combo,
                    in_axes=(0, 0),
                )
    )   
    # shape of results: (num_combos, NUM_EPISODES, 3)
    # because each single rollout returns (total_reward, done_any, log_wrapper_return)
    start_time = time.time()
    results_list = []
    num_chunks = (num_combos + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in tqdm(range(num_chunks), desc="Running chunked vmap over combos"):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, num_combos)
        combos_chunk = combos[start:end]
        keys_chunk = all_keys[start:end]
        # This is the original run_all logic but scoped to the chunk
        results_chunk = run_chunk(combos_chunk, keys_chunk)
        results_list.append(results_chunk)

    # Stack results from each chunk into a single array
    results = tuple(jnp.concatenate(r, axis=0) for r in zip(*results_list))
    end_time = time.time()
    log.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    rewards, dones, logs = results  # unpack the tuple

    # Compute mean & std across episodes, for each combo
    mean_rewards = jnp.mean(rewards, axis=1)  # shape (num_combos,)
    std_rewards  = jnp.std(rewards, axis=1)
    mean_logs = jnp.mean(logs, axis=1)
    std_logs  = jnp.std(logs, axis=1)
    any_dones_per_combo = jnp.any(dones, axis=1)  # shape (num_combos,)

    # Identify the best combo based on highest mean reward
    best_idx = jnp.argmax(mean_rewards)
    best_interval = combos[best_idx, 0]
    best_threshold = combos[best_idx, 1]
    best_reward = mean_rewards[best_idx]
    
    # Report best result including seed if applicable
    if combos.shape[1] > 2:
        best_seed = combos[best_idx, 2]
        log.info(f"Best combo is (interval={best_interval}, threshold={best_threshold}, seed={best_seed}) "
                f"with mean reward = {best_reward / NORM_CONSTANT:.6f}"
                f" and std = {std_rewards[best_idx] / NORM_CONSTANT:.6f}")
    else:
        log.info(f"Best combo is (interval={best_interval}, threshold={best_threshold}) "
                f"with mean reward = {best_reward / NORM_CONSTANT:.6f}"
                f" and std = {std_rewards[best_idx] / NORM_CONSTANT:.6f}")

    # Optionally print out the top few combos
    sorted_indices = jnp.argsort(-mean_rewards)  # descending
    top5 = sorted_indices[:5]
    log.info(f"Top 5 combos by mean reward:")
    for rank, idx in enumerate(top5, start=1):
        if combos.shape[1] > 2:
            log.info(f"Rank {rank}: interval={combos[idx,0]}, threshold={combos[idx,1]}, seed={combos[idx,2]}, "
                    f"mean_reward={mean_rewards[idx]/NORM_CONSTANT:.6f}")
        else:
            log.info(f"Rank {rank}: interval={combos[idx,0]}, threshold={combos[idx,1]}, "
                    f"mean_reward={mean_rewards[idx]/NORM_CONSTANT:.6f}")
        if not any_dones_per_combo[idx]:
            log.warning(f"Warning: No episodes ended in a done state for this combo.")
        if not jnp.allclose(mean_rewards[idx], mean_logs[idx]):
            log.warning(f"Warning: Mean results from the heuristic policy and log wrapper do not match.")
        if not jnp.allclose(std_rewards[idx], std_logs[idx]):
            log.warning(f"Warning: Std results from the heuristic policy and log wrapper do not match.")

    # If you'd like a histogram for a single combo or the best combo:
    if plot_hist:
        plt.hist(rewards[best_idx], bins=20)
        if combos.shape[1] > 2:
            plt.title(f"Histogram of Rewards for Best Combo: (i={best_interval}, t={best_threshold}, s={best_seed})")
        else:
            plt.title(f"Histogram of Rewards for Best Combo: (i={best_interval}, t={best_threshold})")
        plt.xlabel("Total reward")
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":
    main()