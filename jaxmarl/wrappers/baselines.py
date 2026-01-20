# This file was adapted from the original in the process of creating the imp-act adaption of JaxMARL under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
"""Wrappers for use with jaxmarl baselines."""

import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial

# from gymnax.environments import environment, spaces
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from typing import Dict, Optional, List, Tuple, Union
from jaxmarl.environments.overcooked_v2.common import DynamicObject
from jaxmarl.environments.spaces import Box, Discrete, MultiDiscrete
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.num_agents, -1))


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done[0]
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


@struct.dataclass
class OvercookedV2LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_episode_recipe_returns: Dict[str, float]


class OvercookedV2LogWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

        self.recipe_dict = {
            f"{recipe[0]}_{recipe[1]}_{recipe[2]}": DynamicObject.get_recipe_encoding(
                recipe
            )
            for recipe in env.possible_recipes
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)

        recipe_returns = {
            r: jnp.zeros((self._env.num_agents,)) for r in self.recipe_dict
        }

        state = OvercookedV2LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            recipe_returns,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: OvercookedV2LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done[0]
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1

        updated_recipe_returns = {
            id: jax.lax.select(
                (state.env_state.recipe == self.recipe_dict[id]) & ep_done,
                new_episode_return,
                old_episode_return,
            )
            for id, old_episode_return in state.returned_episode_recipe_returns.items()
        }

        state = OvercookedV2LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=jax.lax.select(
                ep_done, new_episode_return, state.returned_episode_returns
            ),
            returned_episode_lengths=jax.lax.select(
                ep_done, new_episode_length, state.returned_episode_lengths
            ),
            returned_episode_recipe_returns=updated_recipe_returns,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        info["returned_episode_recipe_returns"] = state.returned_episode_recipe_returns
        return obs, state, reward, done, info


class MPELogWrapper(LogWrapper):
    """Times reward signal by number of agents within the environment,
    to match the on-policy codebase."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        rewardlog = reward * self._env.num_agents  # As per on-policy codebase
        ep_done = done[0]
        new_episode_return = state.episode_returns + rewardlog
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


@struct.dataclass
class SMAXLogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    won_episode: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_won_episode: int


class SMAXLogWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = SMAXLogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SMAXLogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done[0]
        batch_reward = reward
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        new_won_episode = (batch_reward >= 1.0).astype(jnp.float32)
        state = SMAXLogEnvState(
            env_state=env_state,
            won_episode=new_won_episode * (1 - ep_done),
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            returned_won_episode=state.returned_won_episode * (1 - ep_done)
            + new_won_episode * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_won_episode"] = state.returned_won_episode
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


def get_space_dim(space):
    # get the proper action/obs space from Discrete-MultiDiscrete-Box spaces
    if isinstance(space, (DiscreteGymnax, Discrete)):
        return space.n
    elif isinstance(space, (BoxGymnax, Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        print(space)
        raise NotImplementedError(
            "Current wrapper works only with Discrete/MultiDiscrete/Box action and obs spaces"
        )


class CTRolloutManager(JaxMARLWrapper):
    """
    Rollout Manager for Centralized Training of with Parameters Sharing. Used by JaxMARL Q-Learning Baselines.
    - Batchify multiple environments (the number of parallel envs is defined by batch_size in __init__).
    - Pads the observations of the agents in order to have all the same length.
    - Adds an agent id (one hot encoded) to the observation vectors.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        batch_size: int,
        training_agents: List = None,
        preprocess_obs: bool = True,
    ):

        super().__init__(env)

        self.batch_size = batch_size

        # the agents to train could differ from the total trainable agents in the env (f.i. if using pretrained agents)
        # it's important to know it in order to compute properly the default global rewards and state
        self.training_agents = (
            self.agents if training_agents is None else training_agents
        )
        self.preprocess_obs = preprocess_obs

        # TOREMOVE: this is because overcooked doesn't follow other envs conventions
        if len(env.observation_spaces) == 0:
            self.observation_spaces = {
                agent: self.observation_space() for agent in self.agents
            }
        if len(env.action_spaces) == 0:
            self.action_spaces = {agent: env.action_space() for agent in self.agents}

        # batched action sampling
        self.batch_samplers = {
            agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0))
            for agent in self.agents
        }

        # assumes the observations are flattened vectors
        self.max_obs_length = max(
            list(map(lambda x: get_space_dim(x), self.observation_spaces.values()))
        )
        self.max_action_space = max(
            list(map(lambda x: get_space_dim(x), self.action_spaces.values()))
        )
        self.obs_size = self.max_obs_length
        if self.preprocess_obs:
            self.obs_size += len(self.agents)

        # agents ids
        self.agents_one_hot_array = jnp.eye(len(self.agents))
        # valid actions
        self.valid_actions = {a: jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh = {
            a: jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n)))
            for a, u in self.action_spaces.items()
        }

        # custom valid actions for specific envs
        if "smax" in env.name.lower():
            self.get_valid_actions = lambda state: jax.vmap(env.get_avail_actions)(
                state
            )
        elif "hanabi" in env.name.lower():
            self.get_valid_actions = lambda state: jax.vmap(env.get_legal_moves)(state)

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        obs, state = jax.vmap(self.wrapped_reset, in_axes=0)(keys)
        obs = jnp.swapaxes(obs, 0, 1)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        obs, state, reward, done, info = jax.vmap(
            self.wrapped_step, in_axes=(0, 0, 1)
        )(keys, states, actions)
        obs = jnp.swapaxes(obs, 0, 1)
        reward = jnp.swapaxes(reward, 0, 1)
        done = jnp.swapaxes(done, 0, 1)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.vmap(self._preprocess_obs, in_axes=(0, 0))(
                obs, self.agents_one_hot_array
            )
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs, state, reward, done, info = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.vmap(self._preprocess_obs, in_axes=(0, 0))(
                obs, self.agents_one_hot_array
            )
            obs = jnp.where(done[:, None], 0.0, obs)
        return obs, state, reward, done, info

    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](
            jax.random.split(key, self.batch_size)
        ).astype(int)

    @partial(jax.jit, static_argnums=0)
    def get_valid_actions(self, state):
        # default is to return the same valid actions one hot encoded for each env
        actions = jnp.stack([self.valid_actions_oh[a] for a in self.agents], axis=0)
        return jnp.tile(actions[:, None, :], (1, self.batch_size, 1))

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [
            (0, max(0, self.max_obs_length - arr.shape[-1]))
        ]
        arr = jnp.pad(arr, pad_width, mode="constant", constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
