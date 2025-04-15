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

    def __init__(self, map_name, encoding_type="binary") -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.env = make(f"{map_name}-jax")
        self.map_name = map_name
        self.num_agents = self.env.total_num_segments
        self.agents = [f"agent_{a}" for a in range(self.num_agents)]
        num_damage_states = self.env.num_damage_states
        num_component_actions = len(self.env.action_map)
        self.world_state_size = self.num_agents * num_damage_states + 2
        self.action_spaces = {
            agent: Discrete(num_component_actions) for agent in self.agents
        }
        if encoding_type == "sinusoidal":
            self.agent_encodings = self.sinusoidal_encoding(
                jnp.arange(self.num_agents), 16
            )
        elif encoding_type == "binary":
            self.agent_encodings = self.generate_binary_agent_ids(self.num_agents)
        elif encoding_type == "one-hot":
            self.agent_encodings = jnp.eye(self.num_agents, dtype=jnp.float32)
        elif encoding_type == "none":
            self.agent_encodings = jnp.zeros((self.num_agents, 0), dtype=jnp.float32)
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")
        agent_encoding_dim = self.agent_encodings.shape[1]
        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(num_damage_states + 2 + agent_encoding_dim,))
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""

        _, state = self.env.reset(key)
        obs = self.get_obs(state).astype(jnp.float32)
        global_state = self.get_global_state(obs, state).astype(jnp.float32)
        obs = {agent: obs[i] for i, agent in enumerate(self.agents)}
        obs.update({"__all__": global_state})

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
        states = jax.lax.cond(dones["__all__"],lambda: states_re, lambda: states_st)

        obs = jax.lax.cond(dones["__all__"],lambda: obs_re,lambda: obs_st)
        #! TODO: add infos
        return obs, states, rewards, dones, {}

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        # convert actions dict to array
        array_actions = jnp.stack(
            [actions[agent] for agent in self.agents], dtype=jnp.int32
        )

        state = self.convert_state_to_float64(state)

        _, next_state, reward, done, info = self.env.step_env(key, state, array_actions)
        next_obs = self.get_obs(next_state).astype(jnp.float32)
        global_state = self.get_global_state(next_obs, next_state).astype(jnp.float32)
        reward = reward.astype(jnp.float32)

        # make next_obs, reward, done dicts
        # modify the done signal to include the "__all__" key
        next_obs = {agent: next_obs[a] for a, agent in enumerate(self.agents)}
        rewards = {agent: reward for a, agent in enumerate(self.agents)}
        rewards['__all__'] = reward
        dones = {agent: done for a, agent in enumerate(self.agents)}
        next_obs.update({"__all__": global_state})
        dones.update({"__all__": done})

        next_state = self.convert_state_to_float32(next_state)

        return next_obs, next_state, rewards, dones, info

    def get_obs(self, state: State) -> chex.Array:
        """
        Applies observation function to state.
        Returns the observation for each agent as an array.
        Shape: (num_agents, num_damage_states + 2)
        The last two dimensions are the normalized timestep and budget.
        """
        N = self.env.total_num_segments
        _timestep = jnp.full((N, 1), state.timestep / self.env.max_timesteps)
        _budget = jnp.full((N, 1), state.budget_remaining / self.env.budget_amount)
        return jnp.concatenate([state.belief, _timestep, _budget, self.agent_encodings], axis=1)

    def get_global_state(self, obs, state: State) -> Dict[str, chex.Array]:
        _timestep = jnp.array([state.timestep / self.env.max_timesteps], dtype=jnp.float32)
        _budget = jnp.array([state.budget_remaining / self.env.budget_amount], dtype=jnp.float32)
        return jnp.concatenate([state.belief.flatten(), _timestep, _budget], axis=0)

    def observation_space(self, agent=None):
        """
        Observation space for a given agent.
        All agents have the same observation space.
        """
        return self.observation_spaces[self.agents[0]]

    def action_space(self, agent=None):
        """
        Action space for a given agent.
        All agents have the same action space.
        """
        return self.action_spaces[self.agents[0]]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State = None) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        num_component_actions = len(self.env.action_map)
        return {agent: list(range(num_component_actions)) for agent in self.agents}

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

    @staticmethod
    def sinusoidal_encoding(position, d_model):
        """
        position: int or [N] array of positions (e.g. agent ids or time steps)
        d_model: dimension of the embedding
        """
        position = jnp.atleast_1d(position).astype(jnp.float32)  # shape: [N]
        i = jnp.arange(d_model)[None, :]  # shape: [1, d_model]

        angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = position[:, None] * angle_rates  # shape: [N, d_model]

        # Apply sin to even indices and cos to odd indices
        angle_rads = jnp.where(i % 2 == 0, jnp.sin(angle_rads), jnp.cos(angle_rads))
        return angle_rads  # shape: [N, d_model]
    
    @staticmethod
    def generate_binary_agent_ids(num_agents):
        num_bits = int(jnp.ceil(jnp.log2(num_agents + 1)))  # +1 to avoid zero
        ids = jnp.arange(num_agents) + 1 # start from 1 to avoid zero
        # Create mask for each bit (from highest to lowest)
        bit_masks = 1 << jnp.arange(num_bits - 1, -1, -1)
        binary_ids = ((ids[:, None] & bit_masks) > 0).astype(jnp.int32)
        return binary_ids
    
    @property
    def name(self) -> str:
        """Environment name."""
        return "road_env"
