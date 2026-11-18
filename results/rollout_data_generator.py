import copy
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import CTRolloutManager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # *_road_env.py modules


class RolloutDataGenerator:

    def __init__(self, env_name, alg, checkpoint_path, checkpoint_id, test_num_envs=10):
        """Roll out the checkpoint `checkpoint_id` of the run directory `checkpoint_path`.

        The run directory holds config.yaml and
        checkpoints/<vmapped_seed_key>/checkpoint_<id>.safetensors (the layout of the
        released checkpoints).
        """
        self.env_name = env_name
        self.alg = alg
        self.checkpoint_id = checkpoint_id
        vmapped_seed = 0

        # train config
        train_config = OmegaConf.to_container(
            OmegaConf.load(os.path.join(checkpoint_path, "config.yaml"))
        )
        # Saved configs omit NUM_UPDATES, which sets the zero-padding of checkpoint names
        train_config.setdefault(
            "NUM_UPDATES",
            train_config["TOTAL_TIMESTEPS"]
            // train_config["NUM_STEPS"]
            // train_config["NUM_ENVS"],
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
