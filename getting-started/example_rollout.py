"""
Example script to run a rollout in the JAXMARL framework.

It uses the LogWrapper and CTRolloutManager to manage the environment and
collect metrics during the rollout.

This can be used as a starting point for implementing heuristics
"""

import jax
import jax.numpy as jnp
import flax
import chex
from typing import Any

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager


@flax.struct.dataclass
class Runner:
    key: chex.PRNGKey
    env_state: Any
    obs: chex.Array


MAP_NAME = "ToyExample-v2"
# MAP_NAME = "Cologne-v1"
NUM_ENVS = 1_000
NUM_TIMESTEPS = 50
reward_normalization = 1e6

# Initialise environment
env = make("road_env", map_name=MAP_NAME)
logger_env = LogWrapper(env)
ctrm_env = CTRolloutManager(
    logger_env, batch_size=NUM_ENVS, preprocess_obs=False
)  # preprocess_obs=True adds one-hot encoded agent IDs

key = jax.random.PRNGKey(42)
key, key_reset = jax.random.split(key, 2)

obs, env_state = ctrm_env.batch_reset(key_reset)

# initialise runner
runner = Runner(key=key, env_state=env_state, obs=obs)


def env_step(runner, unused):
    key, env_state, obs = runner.key, runner.env_state, runner.obs
    key, key_step, key_action = jax.random.split(key, 3)

    # choose actions
    # actions (dict): For each agent, an array of size NUM_ENVS
    # (for now, do-nothing)
    actions = {
        agent: jnp.zeros(NUM_ENVS, dtype=jnp.int32)
        for a, agent in enumerate(env.agents)
    }

    next_obs, env_state, _, _, infos = ctrm_env.batch_step(key_step, env_state, actions)

    runner = Runner(key=key, env_state=env_state, obs=next_obs)

    return runner, infos


# rollout for NUM_TIMESTEPS
runner, infos = jax.lax.scan(env_step, runner, None, NUM_TIMESTEPS)

# calculate metrics (_get_greedy_metrics)
# aggregate arrays for which returned_episode is True
metrics = jax.tree.map(
    lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan)),
    infos,
)

# (numpy, do-nothing) ToyExample-v2: -1_080.811
# (numpy, do-nothing) Cologne-v1: -19_984.037
mean_return = metrics["returned_episode_returns"] / reward_normalization
print(f"Map: {MAP_NAME} | Mean return: {mean_return:.2f}")
