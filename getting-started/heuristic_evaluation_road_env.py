import time
import jax
import jax.numpy as jnp
from jaxmarl import make

from jaxmarl.wrappers.baselines import (
    LogWrapper,
)

DEBUG = False
if DEBUG:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_check_tracer_leaks", True)

def policy_do_nothing(key, state, obs):
    """Policy that does nothing."""
    return {agent: 2 for agent in obs.keys()}

def policy_humble_heuristic(key, state, obs):
    """Policy that inspects every n timesteps and takes action 2 if the observation is > 1"""
    tstep = state.env_state.timestep
    obs_insp = state.env_state.observation
    # Step 1: Initialize with default action 0
    actions = jnp.zeros_like(obs_insp, dtype=jnp.int32)
    # Step 2: Apply condition for timestep % 46 == 0 → action 1
    actions = jnp.where(tstep % 5 == 0, 1, actions)
    # Step 3: Apply condition for obs > 1 → action 2 (takes priority)
    actions = jnp.where(obs_insp > 1, 2, actions)
    actions_dict = {f"agent_{i}": actions[i] for i in range(len(obs))}
    return actions_dict

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

def main(plot_hist=True):

    #### Inputs
    MAP_NAME = "CologneBonnDusseldorf-v1" # "ToyExample-v2" "Cologne-v1" "CologneBonnDusseldorf-v1"
    policy = policy_do_nothing

    NUM_EPISODES = 1_000
    NUM_STEPS = 50
    NORM_CONSTANT = 1e6

    """Main function to run the example."""
    key = jax.random.PRNGKey(0)
    key, key_rollout = jax.random.split(key)
    # Initialise environment.
    env = make('road_env', map_name=MAP_NAME)
    env = LogWrapper(env)

    # num_runs, num_steps = 100000, 50

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

    print(f"Total rollout time: {end - start:.4f} seconds")

    mean_reward = jax.numpy.mean(results)
    log_wrapper_mean_reward = jnp.mean(log_wrapper_return)
    std_reward = jax.numpy.std(results)
    log_wrapper_std_reward = jnp.std(log_wrapper_return)

    print(f"Total reward: {mean_reward / NORM_CONSTANT}")
    print(f"Std reward: {std_reward / NORM_CONSTANT}")

    print(f"Log wrapper total reward: {log_wrapper_mean_reward / NORM_CONSTANT}")
    print(f"Log wrapper std reward: {log_wrapper_std_reward / NORM_CONSTANT}")


    print(f"Total number of runs: {NUM_EPISODES}")
    print(f"Any dones: {dones.any()}")

    if plot_hist:
        import matplotlib.pyplot as plt
        plt.hist(results, bins=100)
        plt.xlabel("Total reward")
        plt.ylabel("Frequency")
        plt.title("Histogram of total rewards")
        plt.show()


if __name__ == "__main__":
    main(plot_hist=False)