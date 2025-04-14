import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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
    def policy(key, state, obs):
        tstep = state.env_state.timestep
        obs_insp = state.env_state.observation

        # Step 1: default action = 0
        actions = jnp.zeros_like(obs_insp, dtype=jnp.int32)

        # Step 2: if tstep % interval == 0 => action 1
        actions = jnp.where(tstep % interval == 0, 1, actions)

        # Step 3: if obs_insp > threshold => action 2 (priority)
        actions = jnp.where(obs_insp > threshold, 2, actions)

        actions_dict = {f"agent_{i}": actions[i] for i in range(len(obs))}
        return actions_dict
    return policy


################################################################################
# 2) Run a single rollout with parametric policy
################################################################################
def run_rollout_with_params(key, env, interval, threshold, num_steps):
    """
    Given specific (interval, threshold), build the correct policy, then run a single rollout.
    """
    policy = parametric_heuristic_policy(interval, threshold)
    return run_rollout(key, env, policy, num_steps)


################################################################################
# 3) The same rollout function you already have
################################################################################
def run_rollout(key, env, policy, num_steps):
    """Run a single rollout in the environment."""
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
    carry, (dones, infos) = jax.lax.scan(
        scan_step,
        (key_scan, 0.0, obs, state),
        None,
        length=num_steps
    )
    _, total_reward, _, _ = carry

    log_wrapper_return = jnp.nanmean(
        jnp.where(
            infos["returned_episode"],
            infos["returned_episode_returns"],
            jnp.nan,
        )
    )
    return total_reward, dones.any(), log_wrapper_return


################################################################################
# 4) Main: build the parameter grid, vmap over combos & episodes, compute stats
################################################################################
def main(plot_hist=True):

    # Inputs
    MAP_NAME = "ToyExample-v2" # "ToyExample-v2" "Cologne-v1" "CologneBonnDusseldorf-v1"
    NUM_EPISODES = 1000      # episodes per (interval, threshold) pair
    NUM_STEPS = 50
    NORM_CONSTANT = 1e6
    CHUNK_SIZE = 5
    # Let’s say we want intervals [1..50], thresholds [1..5].
    intervals = jnp.arange(1, 51)
    thresholds = jnp.arange(1, 6)

    # We'll build a grid of shape (50, 5, 2),
    #   then flatten it to (250, 2).
    i_grid, t_grid = jnp.meshgrid(intervals, thresholds, indexing="ij")
    # i_grid.shape = (50, 5), t_grid.shape = (50, 5)
    combos = jnp.stack([i_grid, t_grid], axis=-1)  # shape = (50, 5, 2)
    combos = combos.reshape(-1, 2)                 # shape = (250, 2)
    num_combos = combos.shape[0]  # 250

    # Now we need random keys for each (combo, episode).
    # So in total, we need num_combos * NUM_EPISODES distinct keys.
    key = jax.random.PRNGKey(0)
    all_keys = jax.random.split(key, num_combos * NUM_EPISODES)
    # Reshape to (num_combos, NUM_EPISODES).
    all_keys = all_keys.reshape(num_combos, NUM_EPISODES, 2)

    # If we want the same key for each combo:
    # combo_keys = jnp.repeat(jnp.expand_dims(key, 0), num_combos, axis=0)
    # all_keys = jax.vmap(lambda ck: jax.random.split(ck, NUM_EPISODES))(combo_keys)

    # Make the environment
    env = make('road_env', map_name=MAP_NAME)
    env = LogWrapper(env)

    # We'll define a function that, for one combo, runs all episodes with vmap.
    def run_episodes_for_combo(combo, keys_for_episodes):
        """
        combo = [interval, threshold] shape (2,)
        keys_for_episodes = shape (NUM_EPISODES, 2) or (NUM_EPISODES, ...)
        """
        interval, threshold = combo[0], combo[1]

        # For each episode, run one rollout
        def rollout_one_episode(k):
            return run_rollout_with_params(k, env, interval, threshold, NUM_STEPS)
        
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
    print(f"\n Evaluation completed in {end_time - start_time:.2f} seconds.")

    rewards, dones, logs = results  # unpack the tuple

    # Now you can handle them separately
    # mean_rewards = jnp.mean(rewards, axis=1)

    # Separate them out
    # rewards = results[..., 0]   # shape (num_combos, NUM_EPISODES)
    # dones   = results[..., 1]   # shape (num_combos, NUM_EPISODES)
    # logs    = results[..., 2]   # shape (num_combos, NUM_EPISODES)

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

    print(f"Best combo is (interval={best_interval}, threshold={best_threshold}) "
          f"with mean reward = {best_reward / NORM_CONSTANT:.6f}"
          f" and std = {std_rewards[best_idx] / NORM_CONSTANT:.6f}")

    # Optionally print out the top few combos
    # This is a bit more advanced, sorting or partial sorting the combos by mean reward
    sorted_indices = jnp.argsort(-mean_rewards)  # descending
    top5 = sorted_indices[:5]
    print("\nTop 5 combos (interval, threshold) by mean reward:")
    for rank, idx in enumerate(top5, start=1):
        print(f"Rank {rank}: interval={combos[idx,0]}, threshold={combos[idx,1]}, "
              f"mean_reward={mean_rewards[idx]/NORM_CONSTANT:.6f}, done_any={any_dones_per_combo[idx]}")

    # If you’d like a histogram for a single combo or the best combo:
    if plot_hist:
        # e.g., histogram of all episodes of the best combo
        plt.hist(rewards[best_idx], bins=20)
        plt.title(f"Histogram of Rewards for Best Combo: (i={best_interval}, t={best_threshold})")
        plt.xlabel("Total reward")
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":
    main(plot_hist=False)