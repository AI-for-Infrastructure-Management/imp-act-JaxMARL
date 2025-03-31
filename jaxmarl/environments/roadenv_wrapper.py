"""
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

import jax
import jax.numpy as jnp
from typing import Dict
import chex
from functools import partial
from flax import struct
from typing import Tuple, Optional
from jaxmarl.environments.spaces import Box, Discrete

# imp-act imports
from imp_act import make
from imp_act.environments.jax_environment import EnvState


@struct.dataclass
class State:
    done: chex.Array
    step: int


class RoadEnvironment_Wrapper(object):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(self, map_name) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.env = make(f"{map_name}-jax")
        self.num_agents = self.env.total_num_segments
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        num_damage_states = self.env.num_damage_states
        num_component_actions = len(self.env.action_map)
        self.observation_spaces = {
            i: Box(low=0, high=1, shape=(num_damage_states + 2,)) for i in self.agents
        }
        self.action_spaces = {i: Discrete(num_component_actions) for i in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""

        _, state = self.env.reset(key)
        obs = self.get_obs(state).astype(jnp.float32)
        obs = {agent: obs[i] for i, agent in enumerate(self.agents)}

        state = self.convert_state_to_float32(state)

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly.
        """

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        #! TODO: add infos
        return obs, states, rewards, dones, {}

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        # convert actions dict to array
        array_actions = jnp.array(
            jax.tree_util.tree_leaves(actions), dtype=jnp.int32
        ).squeeze()

        state = self.convert_state_to_float64(state)

        _, next_state, reward, done, info = self.env.step_env(key, state, array_actions)
        obs = self.get_obs(state).astype(jnp.float32)
        reward = reward.astype(jnp.float32)

        # make obs a dict again
        obs = {agent: obs[i] for i, agent in enumerate(self.agents)}
        reward = {agent: reward for i, agent in enumerate(self.agents)}

        # modify the done signal to include the "__all__" key
        dones = {a: done for i, a in enumerate(self.agents)}
        dones.update({"__all__": done})

        next_state = self.convert_state_to_float32(next_state)

        return obs, next_state, reward, dones, info

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Applies observation function to state.
        Returns the observation for each agent as an array.
        Shape: (num_agents, num_damage_states + 2)
        The last two dimensions are the normalized timestep and budget.
        """
        N = self.env.total_num_segments
        _timestep = jnp.full((N, 1), state.timestep / self.env.max_timesteps)
        _budget = jnp.full((N, 1), state.budget_remaining / self.env.budget_amount)
        return jnp.concatenate([state.belief, _timestep, _budget], axis=1)

    @staticmethod
    def convert_state_to_float32(state):
        """
        Converts all attributes of state object to float32 by creating
        a new object.
        """

        env_state = EnvState(
            damage_state=state.damage_state.astype(jnp.int32),
            observation=state.observation.astype(jnp.int32),
            belief=state.belief.astype(jnp.float32),
            base_travel_time=state.base_travel_time.astype(jnp.float32),
            capacity=state.capacity.astype(jnp.float32),
            worst_obs_counter=state.worst_obs_counter.astype(jnp.int32),
            deterioration_rate=state.deterioration_rate.astype(jnp.int32),
            timestep=state.timestep,
            budget_remaining=state.budget_remaining,
            episode_return=state.episode_return,
        )

        return env_state

    @staticmethod
    def convert_state_to_float64(state):
        """
        Converts all attributes of state object to float64 by creating
        a new object.
        """

        env_state = EnvState(
            damage_state=state.damage_state.astype(jnp.int64),
            observation=state.observation.astype(jnp.int64),
            belief=state.belief.astype(jnp.float64),
            base_travel_time=state.base_travel_time.astype(jnp.float64),
            capacity=state.capacity.astype(jnp.float64),
            worst_obs_counter=state.worst_obs_counter.astype(jnp.int64),
            deterioration_rate=state.deterioration_rate.astype(jnp.int64),
            timestep=state.timestep,
            budget_remaining=state.budget_remaining,
            episode_return=state.episode_return,
        )

        return env_state

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent=None):
        """Observation space for a given agent."""
        return self.observation_spaces[self.agents[0]]

    def action_space(self, agent=None):
        """Action space for a given agent."""
        return self.action_spaces[self.agents[0]]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State = None) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        num_component_actions = len(self.env.action_map)
        return {i: list(range(num_component_actions)) for i in range(self.num_agents)}

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
