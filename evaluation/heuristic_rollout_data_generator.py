import jax
import jax.numpy as jnp

from heuristic_evaluation_road_env import make_get_rollout_data

from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    LogWrapper,
)


class HeuristicRolloutDataGenerator:

    def __init__(self, config):

        # Environment
        env = make("road_env", map_name=config["map"], record_rollout=True)
        env = LogWrapper(env)
        self.env = env
        self.road_env = env.env

        # Rollout data generator
        self.get_rollout_data = make_get_rollout_data(config)

    def generate_rollout_data(self, key, num_rollouts, verbose=False):
        """
        Keys and their shapes
        # shape (E, T, N, _)
        # E = number of environments
        # T = number of timesteps
        # N = number of agents
        # A = number of actions

        Example
        -------
        action (10, 50, 60)
        applied_actions (10, 50, 60)
        budget_constraints_applied (10, 50)
        budget_remaining (10, 51)
        edge_beliefs (10, 51, 60, 5)
        edge_observations (10, 51, 60)
        edge_states (10, 51, 60)
        episode_cost (10, 1)
        forced_replace_constraint_applied (10, 50)
        maintenance_reward (10, 50)
        reward (10, 50)
        terminal_reward (10, 50)
        timesteps (10, 51)
        total_travel_time (10, 50)
        traffic_volumes (10, 50, 60)
        travel_time_reward (10, 50)
        travel_times (10, 50, 60)
        """

        # Run jitted function to get rollout data
        rollout_keys = jax.random.split(key, num_rollouts)
        initial_env_state, env_state, actions, rewards, infos = self.get_rollout_data(
            rollout_keys
        )

        ############### Make episode data a single dict ################
        episode_data = {
            "timesteps": None,  #! shape: (N, T+1)
            "edge_states": None,  #! shape: (N, T+1)
            "edge_observations": None,  #! shape: (N, T+1)
            "edge_beliefs": None,  #! shape: (N, T+1, S)
            "budget_remaining": None,  #! shape: (N, T+1)
        }

        # observations
        episode_data["timesteps"] = jnp.concatenate(
            [
                initial_env_state.env_state.timestep[:, None],
                env_state.env_state.timestep,
            ],
            axis=1,
        )
        episode_data["edge_states"] = jnp.concatenate(
            [
                initial_env_state.env_state.damage_state[:, None, :],
                env_state.env_state.damage_state,
            ],
            axis=1,
        )

        episode_data["edge_observations"] = jnp.concatenate(
            [
                initial_env_state.env_state.observation[:, None, :],
                env_state.env_state.observation,
            ],
            axis=1,
        )
        episode_data["edge_beliefs"] = jnp.concatenate(
            [
                initial_env_state.env_state.belief[:, None, :],
                env_state.env_state.belief,
            ],
            axis=1,
        )

        # actions
        episode_data["action"] = jnp.stack(
            [actions[a] for a in self.env.agents]
        ).transpose((1, 2, 0))
        episode_data["applied_actions"] = infos["applied_actions"]
        episode_data["forced_replace_constraint_applied"] = infos[
            "forced_replace_constraint_applied"
        ]

        # traffic
        episode_data["travel_times"] = infos["travel_times"]
        episode_data["traffic_volumes"] = infos["traffic_volumes"]
        episode_data["total_travel_time"] = infos["total_travel_time"]

        # budget
        episode_data["budget_remaining"] = jnp.concatenate(
            [
                initial_env_state.env_state.budget_remaining[:, None],
                env_state.env_state.budget_remaining,
            ],
            axis=1,
        )
        episode_data["budget_constraints_applied"] = infos["budget_constraints_applied"]
        episode_data["forced_repair_flags"] = infos["forced_repair_flags"]

        # rewards
        episode_data["reward"] = rewards["__all__"]
        episode_data["episode_cost"] = (
            (infos["returned_episode_returns"] * infos["returned_episode"])
            .mean(axis=-1)
            .sum(axis=-1, keepdims=True)
        )
        episode_data["travel_time_reward"] = infos["reward_elements"][
            "travel_time_reward"
        ]
        episode_data["maintenance_reward"] = infos["reward_elements"][
            "maintenance_reward"
        ]
        episode_data["terminal_reward"] = infos["reward_elements"]["terminal_reward"]

        ###################### Sanity check ############################
        # ensure last timestep is the same as the env.max_timesteps
        max_timesteps = self.road_env.max_timesteps
        assert (
            episode_data["timesteps"][0, -1] == max_timesteps
        ), f"Last timestep is not the same as the env.max_timesteps: {episode_data['timesteps'][0, -1]} != {max_timesteps}. Make sure to disable the automatic reset of the environment in the last step of the rollout."

        if verbose:
            for key in sorted(episode_data.keys()):
                print(f"{key:<35}", episode_data[key].shape)

        return episode_data

    @staticmethod
    def get_single_episode_data(episode_data, episode_idx=0):
        """Get a single episode data from the rollout data"""
        return jax.tree_map(lambda x: x[episode_idx], episode_data)
