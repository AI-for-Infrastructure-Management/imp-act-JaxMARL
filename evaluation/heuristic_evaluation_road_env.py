import time
import jax
import jax.numpy as jnp
import logging
import numpy as np
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    LogWrapper,
)

# Get Hydra's logger
log = logging.getLogger(__name__)

DEBUG = False
if DEBUG:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_check_tracer_leaks", True)

def get_policy_do_nothing(params=None):
    """Returns a do_nothing policy function.
    
    Args:
        params: Not used for this policy, included for consistency.
    """
    def policy(key, state, obs, env):
        """Policy that does nothing."""
        return {agent: 0 for agent in obs.keys()}
    return policy

def get_policy_random(params=None):
    """Returns a random policy function.
    
    Args:
        params: Not used for this policy, included for consistency.
    """
    def policy(key, state, obs, env):
        """Policy that picks a random action (0, 1, or 2) for each agent."""
        num_agents = len(obs)
        num_actions = 3  # Update this if you have more or fewer discrete actions
        keys = jax.random.split(key, num_agents)
        actions = jax.vmap(lambda k: jax.random.randint(k, (), 0, num_actions))(keys)
        return {f"agent_{i}": actions[i] for i in range(num_agents)}
    return policy

def get_policy_humble_heuristic(params):
    """Returns a humble_heuristic policy function with configurable parameters.
    
    Args:
        params: Dictionary containing:
            - inspection_interval: Timestep interval for inspection (action 1)
            - repair_threshold: Observation threshold above which repair action is taken (action 2)
    """
    inspection_interval = params.get("inspection_interval", 6)
    repair_threshold = params.get("repair_threshold", 1)
    
    def policy(key, state, obs, env):
        """Policy that inspects at specified intervals and repairs when observation exceeds threshold."""
        road_env = env._env.env
        road_env_state = state.env_state
        
        tstep = state.env_state.timestep
        obs_insp = state.env_state.observation
        # Step 1: Initialize with default action 0
        actions = jnp.zeros_like(obs_insp, dtype=jnp.int32)
        # Step 2: Apply condition for inspection based on configured interval
        actions = jnp.where(tstep % inspection_interval == 0, 1, actions)
        # Step 3: Apply condition for repair based on configured threshold
        actions = jnp.where(obs_insp > repair_threshold, 2, actions)

        actions_dict = {f"agent_{i}": actions[i] for i in range(env.num_agents)}
        return actions_dict
    return policy

def get_budget_prioritized_policy(policy, params):
    """Returns a prioritized policy function with configurable parameters.
    
    Args:
        params: Dictionary containing:
            - inspection_interval: Timestep interval for inspection (action 1)
            - repair_threshold: Observation threshold above which repair action is taken (action 2)
    """

    if params.get("priorization_key") == "random":
        seed = params.get("random_seed")
        if params.get("random_seed") == "random":
            seed = np.random.randint(0, 2**32 - 1)
        log.info(f"Random seed for budget prioritization: {seed}")
        prio_key = jax.random.PRNGKey(seed)


    def prioritized_policy(key, state, obs, env):
        """Policy that inspects at specified intervals and repairs when observation exceeds threshold."""
        road_env = env._env.env
        road_env_state = state.env_state

        action = policy(key, state, obs, env)
        action = jnp.array([action[f"agent_{i}"] for i in range(env.num_agents)])

        # Step 4: Prioritize repair actions
        forced_action, forced_repair_mask = road_env._apply_forced_repair_constraint(
            action, road_env_state.worst_obs_counter
        )

        do_nothing_action = jnp.zeros_like(action)

        action = jnp.where(
            forced_repair_mask,
            do_nothing_action,
            action,
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
        action_cost = road_env._get_budget_action_cost(road_env_state, action, forced_repair_mask)
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
                priorities = jax.random.uniform(prio_key, shape=action.shape)
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
            constrained_action = jnp.zeros_like(action)
            constrained_action = constrained_action.at[sorted_indices].set(
                jnp.where(
                    valid_mask,
                    action[sorted_indices],
                    do_nothing_action[sorted_indices],
                )
            )

            return constrained_action, True

        constrained_action, constraint_applied = jax.lax.cond(
            jnp.sum(adjusted_cost) > remaining_budget,
            lambda: apply_constraints(),
            lambda: (action, False),
        )

        actions_dict = {f"agent_{i}": constrained_action[i] for i in range(len(obs))}
        return actions_dict
    return prioritized_policy

def run_rollout(key, env, policy, num_steps):
    """Run a rollout in the environment."""
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
    carry, (dones, infos) = jax.lax.scan(scan_step, (key_scan, 0.0, obs, state, 0), None, length=num_steps)
    _, total_reward, _, _, episodes = carry
    log_wrapper_return = jnp.nanmean(
        jnp.where(
            infos["returned_episode"],
            infos["returned_episode_returns"],
            jnp.nan,
        )
    )
    return total_reward / episodes, dones.any(), log_wrapper_return

def make_get_rollout_data(config):

    # Environment
    env = make('road_env', map_name=config['map'], record_rollout=True)
    env = LogWrapper(env)

    # Policy
    policy_factory = globals()[f"get_policy_{config['policy']}"]
    # Extract policy parameters if they exist
    policy_params = config.get("policy_params", {})
    policy = policy_factory(policy_params)

    @jax.vmap
    @jax.jit
    def get_rollout_data(key):

        def scan_step(carry, _):
            key, last_obs, last_state = carry

            key, key_act, key_step = jax.random.split(key, 3)
            actions = policy(key_act, last_state, last_obs, env)
            obs, state, reward, done, infos = env.step(key_step, last_state, actions)
            return (key, obs, state), (state, actions, reward, infos)

        key, key_reset, key_scan = jax.random.split(key, 3)
        init_obs, init_state = env.reset(key_reset)
        init_carry = (key_scan, init_obs, init_state)

        #! NOTE:
        # 1. Rollout is limited to max_timesteps
        # 2. To include last step, env_step is used instead of step (which would
        # otherwise reset the env when done)
        carry, (env_state, actions, reward, infos) = jax.lax.scan(scan_step, init_carry, None, length=env.env.max_timesteps)

        return init_state, env_state, actions, reward, infos

    return get_rollout_data


@hydra.main(config_path="config/heuristics", config_name="toy_example_humble_heuristic", version_base=None)
def main(cfg: DictConfig):
    # Log the configuration
    print(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.to_container(cfg))}")

    if cfg.get("double_precision_mode", False):
        jax.config.update("jax_enable_x64", True)
        log.info("Using double precision mode")

    #### Inputs
    MAP_NAME = cfg.map
    policy_name = cfg.policy
    
    # Get policy with parameters
    policy_factory = globals()[f"get_policy_{policy_name}"]
    # Extract policy parameters if they exist
    policy_params = cfg.get("policy_params", {})
    policy = policy_factory(policy_params)

    if cfg.get("priorization_params") is not None:
        policy = get_budget_prioritized_policy(policy, cfg.priorization_params)

    NUM_EPISODES = cfg.num_episodes
    NUM_STEPS = cfg.num_steps
    NORM_CONSTANT = cfg.norm_constant

    """Main function to run the example."""
    key = jax.random.PRNGKey(0)
    key, key_rollout = jax.random.split(key)
    # Initialise environment.
    env = make('road_env', map_name=MAP_NAME)
    env = LogWrapper(env)

    jit_vmap_rollout = jax.jit(
        jax.vmap(
            run_rollout,
            in_axes=(0, None, None, None),
            out_axes=0,
        ),
        static_argnums=(1,2,3),
    )

    keys = jax.random.split(key_rollout, NUM_EPISODES)

    # === TIMING START ===
    start = time.perf_counter()

    results, dones, log_wrapper_return = jit_vmap_rollout(keys, env, policy, NUM_STEPS)

    results.block_until_ready()
    end = time.perf_counter()
    # === TIMING END ===

    log.info(f"{'='*35} RESULTS {'='*35}")

    log.info(f"Total rollout time: {end - start:.4f} seconds")

    mean_reward = jax.numpy.mean(results)
    log_wrapper_mean_reward = jnp.mean(log_wrapper_return)
    std_reward = jax.numpy.std(results)
    log_wrapper_std_reward = jnp.std(log_wrapper_return)

    log.info(f"Total number of runs: {NUM_EPISODES}")

    log.info(f"Total reward: {mean_reward / NORM_CONSTANT:.6f}")
    log.info(f"Std reward: {std_reward / NORM_CONSTANT:.6f}")

    if not jnp.allclose(mean_reward, log_wrapper_mean_reward):
        log.warning("Warning: Mean results from the heuristic policy and log wrapper do not match.")

    if not jnp.allclose(std_reward, log_wrapper_std_reward):
        log.warning("Warning: Std results from the heuristic policy and log wrapper do not match.")
    
    if not dones.any():
        log.warning(f"Warning: No episodes ended in a done state.")

    if cfg.plot_histogram:
        import matplotlib.pyplot as plt
        plt.hist(results, bins=100)
        plt.xlabel("Total reward")
        plt.ylabel("Frequency")
        plt.title(f"Episode rewards for {MAP_NAME} using {policy_name}")
        plt.show()


if __name__ == "__main__":
    main()