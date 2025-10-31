import copy
import os
import pandas as pd
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import CTRolloutManager

from safetensors.flax import save_file, load_file


class RolloutDataGenerator:

    def __init__(
        self,
        env_name,
        alg,
        test_num_envs=10,
        checkpoint_id=None,
        checkpoint_dir=None,
    ):
        self.env_name = env_name
        self.alg = alg
        vmapped_seed = 0

        # Load checkpoint
        if checkpoint_dir is None and checkpoint_id is None:
            self.get_checkpoint_dir_and_id()
        else:
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_id = checkpoint_id
        checkpoint_path = f"../outputs/{env_name}/{self.alg}/{self.checkpoint_dir}"

        # train config
        train_config = OmegaConf.to_container(
            OmegaConf.load(os.path.join(checkpoint_path, "config.yaml"))
        )

        if self.alg in ["mappo_rnn", "mappo"]:

            from mappo_rnn_road_env import load_checkpoint_agent, make_get_rollout_data

            # Initialise environment
            env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])
            _, _, loaded_params, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["ippo_rnn", "ippo"]:
            from ippo_rnn_road_env import load_checkpoint_agent, make_get_rollout_data

            # Initialise environment
            env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])

            _, loaded_params, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["vdn_rnn", "vdn"]:

            from vdn_rnn_road_env import (
                load_checkpoint_agent,
                make_get_rollout_data,
                env_from_config,
            )

            # Initialise environment
            env, _ = env_from_config(copy.deepcopy(train_config))
            env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

            train_config["MAX_ACTION_SPACE"] = env.max_action_space
            loaded_params, _, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        # ToDo
        elif self.alg in ["qmix_rnn", "qmix"]:
            from qmix_rnn_road_env import (
                load_checkpoint_agent,
                make_get_rollout_data,
                env_from_config,
            )

            # Initialise environment
            env, _ = env_from_config(copy.deepcopy(train_config))
            env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

            train_config["MAX_ACTION_SPACE"] = env.max_action_space
            loaded_params, _, _, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        elif self.alg in ["pqn_rnn", "pqn"]:
            from pqn_rnn_road_env import (
                load_checkpoint_agent,
                make_get_rollout_data,
                env_from_config,
            )

            # Initialise environment
            env, _ = env_from_config(copy.deepcopy(train_config))
            env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

            train_config["MAX_ACTION_SPACE"] = env.max_action_space
            loaded_params, _ = load_checkpoint_agent(
                checkpoint_path, self.checkpoint_id, vmapped_seed, train_config
            )

        else:
            raise NotImplementedError(f"Algorithm {self.alg} not supported.")

        self.jit_get_rollout_data = jax.jit(
            make_get_rollout_data(train_config, test_num_envs)
        )
        self.loaded_params = loaded_params
        self.env = env
        self.road_env = env.env

    def get_checkpoint_dir_and_id(self):
        inference_df = pd.read_csv("inference_results.csv")

        _runs = inference_df[
            (inference_df["algorithm"] == self.alg)
            & (inference_df["map_name"] == self.env_name)
        ]

        print("Selected best checkpoint dir and id")
        _runs = (
            _runs.sort_values(by=["mean"], ascending=False)
            .reset_index(drop=True)
            .iloc[0]
        )

        self.checkpoint_dir = _runs["checkpoint_dir_name"]
        self.checkpoint_id = _runs["checkpoint_id"]
        print(
            f"Checkpoint dir: {self.checkpoint_dir}, checkpoint id: {self.checkpoint_id}"
        )

    def generate_rollout_data(self, key, verbose=False):
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
        initial_env_state, env_state, env_act, rewards, infos = (
            self.jit_get_rollout_data(key, self.loaded_params)
        )

        ############### Make episode data a single dict ################
        episode_data = {
            "timesteps": None,  #! shape: (T+1, N)
            "edge_states": None,  #! shape: (T+1, N)
            "edge_observations": None,  #! shape: (T+1, N)
            "edge_beliefs": None,  #! shape: (T+1, S, N)
            "budget_remaining": None,  #! shape: (T+1,)
        }

        # observations
        episode_data["timesteps"] = jnp.concatenate(
            [
                initial_env_state.env_state.timestep[None, :],
                env_state.env_state.timestep,
            ],
            axis=0,
        )
        episode_data["edge_states"] = jnp.concatenate(
            [
                initial_env_state.env_state.damage_state[None, :],
                env_state.env_state.damage_state,
            ],
            axis=0,
        )
        episode_data["edge_observations"] = jnp.concatenate(
            [
                initial_env_state.env_state.observation[None, :],
                env_state.env_state.observation,
            ],
            axis=0,
        )
        episode_data["edge_beliefs"] = jnp.concatenate(
            [initial_env_state.env_state.belief[None, :], env_state.env_state.belief],
            axis=0,
        )

        # actions
        episode_data["action"] = jnp.stack(
            [env_act[a] for a in self.env.agents]
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
                initial_env_state.env_state.budget_remaining[None, :],
                env_state.env_state.budget_remaining,
            ],
            axis=0,
        )
        episode_data["budget_constraints_applied"] = infos["budget_constraints_applied"]
        episode_data["forced_repair_flags"] = infos["forced_repair_flags"]

        # rewards
        episode_data["reward"] = rewards["__all__"]
        episode_data["episode_cost"] = (
            (infos["returned_episode_returns"] * infos["returned_episode"])
            .sum(axis=0)
            .mean(axis=1)
        )[None, :]
        episode_data["travel_time_reward"] = infos["reward_elements"][
            "travel_time_reward"
        ]
        episode_data["maintenance_reward"] = infos["reward_elements"][
            "maintenance_reward"
        ]
        episode_data["terminal_reward"] = infos["reward_elements"]["terminal_reward"]

        episode_data = jax.tree.map(
            lambda x: x.swapaxes(1, 0), episode_data
        )  #! swap axes

        ###################### Sanity check ############################
        # ensure last timestep is the same as the env.max_timesteps
        max_timesteps = self.road_env.max_timesteps
        assert (
            episode_data["timesteps"][0, -1] == max_timesteps
        ), f"Last timestep is not the same as the env.max_timesteps: {episode_data['timesteps'][0, -1]} != {max_timesteps}. Make sure to disable the automatic reset of the environment in the last step of the rollout."

        if verbose:
            for key, value in episode_data.items():
                print(f"{key:<35}", value.shape)

        return episode_data

    @staticmethod
    def get_single_episode_data(episode_data, episode_idx=0):
        """Get a single episode data from the rollout data"""
        return jax.tree_map(lambda x: x[episode_idx], episode_data)

    def save_rollout_data(self, episode_data):
        """Save the rollout data to .safetensors file"""

        os.makedirs(
            f"./rollout_data/{self.env_name}/{self.alg}/{self.checkpoint_dir}",
            exist_ok=True,
        )

        # Save the rollout data
        save_file(
            episode_data,
            f"./rollout_data/{self.env_name}/{self.alg}/{self.checkpoint_dir}/rollout_data_{self.checkpoint_id}.safetensors",
        )
        print(
            f"Rollout data saved to {self.env_name}/{self.alg}/{self.checkpoint_dir}/rollout_data_{self.checkpoint_id}.safetensors"
        )

    def load_rollout_data(self):
        """Load the rollout data from .safetensors file"""

        try:
            # Load the rollout data
            rollout_data = load_file(
                f"./rollout_data/{self.env_name}/{self.alg}/{self.checkpoint_dir}/rollout_data_{self.checkpoint_id}.safetensors"
            )

            print(
                f"Rollout data loaded from {self.env_name}/{self.alg}/{self.checkpoint_dir}/rollout_data_{self.checkpoint_id}.safetensors"
            )
        except FileNotFoundError:
            print(
                f"Rollout data not found in {self.env_name}/{self.alg}/{self.checkpoint_dir}/rollout_data_{self.checkpoint_id}.safetensors. Generating new rollout data."
            )
        return rollout_data
