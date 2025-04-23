import time
import jax
import jax.numpy as jnp
import logging

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
    def policy(key, state, obs):
        """Policy that does nothing."""
        return {agent: 0 for agent in obs.bs.keys()}
    return policy

def get_policy_random(params=None):
    """Returns a random policy function.
    
    Args:
        params: Not used for this policy, included for consistency.
    """
    def policy(key, state, obs):
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
    
    def policy(key, state, obs):
        """Policy that inspects at specified intervals and repairs when observation exceeds threshold."""
        tstep = state.env_state.timestep
        obs_insp = state.env_state.observation
        # Step 1: Initialize with default action 0
        actions = jnp.zeros_like(obs_insp, dtype=jnp.int32)
        # Step 2: Apply condition for inspection based on configured interval
        actions = jnp.where(tstep % inspection_interval == 0, 1, actions)
        # Step 3: Apply condition for repair based on configured threshold
        actions = jnp.where(obs_insp > repair_threshold, 2, actions)
        actions_dict = {f"agent_{i}": actions[i] for i in range(len(obs))}
        return actions_dict
    return policy

def run_rollout(key, env, policy, num_steps):
    """Run a rollout in the environment."""
    def scan_step(carry, _):
        key, total_reward, last_obs, last_state = carry
        key, key_act = jax.random.split(key)
        actions = policy(key_act, last_state, last_obs)
        key, key_step = jax.random.split(key)
        obs, state, reward, done, infos = env.step(key_step, last_state, actions)
        total_reward = total_reward + reward["__all__"]
        return (key, total_reward, obs, state), (done["__all__"], infos)
    
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset)
    
    key, key_scan = jax.random.split(key)
    carry, (dones, infos) = jax.lax.scan(scan_step, (key_scan, 0.0, obs, state), None, length=num_steps)
    _, total_reward, _, _ = carry
    log_wrapper_return = jnp.nanmean(
        jnp.where(
            infos["returned_episode"],
            infos["returned_episode_returns"],
            jnp.nan,
        )
    )
    return total_reward, dones.any(), log_wrapper_return

@hydra.main(config_path="config/heuristics", config_name="toy_example_do_nothing", version_base=None)
def main(cfg: DictConfig):
    # Log the configuration
    print(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.to_container(cfg))}")

    if cfg.get("double_precision_mode", False):
        jax.config.update("jax_enable_x64", True)
        log.info("Using double precision mode")

    #### Inputs
    env_kwargs = cfg.env_kwargs
    policy_name = cfg.policy
    
    # Get policy with parameters
    policy_factory = globals()[f"get_policy_{policy_name}"]
    # Extract policy parameters if they exist
    policy_params = cfg.get("policy_params", {})
    policy = policy_factory(policy_params)

    NUM_EPISODES = cfg.num_episodes
    NUM_STEPS = cfg.num_steps
    NORM_CONSTANT = cfg.norm_constant

    """Main function to run the example."""
    key = jax.random.PRNGKey(0)
    key, key_rollout = jax.random.split(key)
    # Initialise environment.
    env = make('road_env', **env_kwargs)
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