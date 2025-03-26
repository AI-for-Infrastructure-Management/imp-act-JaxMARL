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
            f"agent_{i}": Box(low=0, high=1, shape=(num_damage_states,))
            for i in range(self.num_agents)
        }
        self.action_spaces = {
            f"agent_{i}": Discrete(num_component_actions)
            for i in range(self.num_agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""

        obs, state = self.env.reset(key)
        obs = {f"agent_{i}": obs[i] for i in range(self.num_agents)}
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
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        # convert actions dict to array
        array_actions = jnp.array(jax.tree_util.tree_leaves(actions))

        obs, next_state, reward, done, info = self.env.step_env(
            key, state, array_actions
        )

        # make obs a dict again
        obs = {f"agent_{i}": obs[i] for i in range(self.num_agents)}

        # modify the done signal to include the "__all__" key
        dones = {a: done for i, a in enumerate(self.agents)}
        dones.update({"__all__": done})

        return obs, next_state, reward, dones, info

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        return self.env.get_obs(state)

    def observation_space(self, agent: str = "agent_0"):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str = "agent_0"):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State = None) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        num_component_actions = len(self.env.action_map)
        return {i: list(range(num_component_actions)) for i in range(self.num_agents)}

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
