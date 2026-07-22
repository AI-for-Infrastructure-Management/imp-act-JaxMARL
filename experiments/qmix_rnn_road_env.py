"""
This file was adapted from the original in the process of creating the 
imp-act adaption of JaxMARL under the 
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

Original file: baselines/QLearning/qmix_rnn.py
"""
import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
import logging
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
    save_params,
    resolve_episode_horizon,
    make_store_eval_returns,
)

# Get Hydra's logger
log = logging.getLogger(__name__)

class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        hidden = hidden.astype(jnp.float32)
        obs = obs.astype(jnp.float32)
        dones = dones.astype(jnp.float32)

        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals


class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""

    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return x


class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """

    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):

        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0))  # (time_steps, batch_size, n_agents)

        # hypernetwork
        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale
        )(states)

        # monotonicity and reshaping
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()  # (time_steps, batch_size)


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

@chex.dataclass
class RunningStats:
    count: jnp.ndarray  # or float32
    mean: jnp.ndarray
    M2: jnp.ndarray

def init_running_stats():
    return RunningStats(
        count=jnp.array(0.0, dtype=jnp.float32),
        mean=jnp.array(0.0, dtype=jnp.float32),
        M2=jnp.array(0.0, dtype=jnp.float32),
    )

def update_running_stats(stats: RunningStats, x: jnp.ndarray) -> RunningStats:
    x = x.astype(jnp.float32)        # if x might be float64
    batch_count = x.size
    batch_sum = jnp.sum(x)
    batch_mean = batch_sum / batch_count
    batch_var = jnp.mean((x - batch_mean) ** 2)
    batch_M2 = batch_var * batch_count

    new_count = stats.count + jnp.asarray(batch_count, dtype=jnp.float32)
    new_mean = (stats.count * stats.mean + batch_sum) / new_count
    new_M2 = (
        stats.M2
        + batch_M2
        + (stats.count * batch_count * (stats.mean - batch_mean) ** 2) / new_count
    )
    return RunningStats(count=new_count, mean=new_mean, M2=new_M2)

def get_std(stats: RunningStats) -> jnp.ndarray:
    return jnp.sqrt(stats.M2 / (stats.count + 1e-8))

def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_EVALS"] = int(1 / config["TEST_INTERVAL"])
    config["EVAL_UPDATES"] = jnp.asarray(
        np.linspace(1, config["NUM_UPDATES"], config["NUM_EVALS"], dtype=int)
    )

    episode_horizon = resolve_episode_horizon(config, env)

    if config.get("SAVE_CHECKPOINTS", False):
        config["NUM_CHECKPOINTS"] = int(1 / config["SAVE_CHECKPOINTS_INTERVAL"])
        config["CHECKPOINT_UPDATES"] = jnp.asarray(
            np.linspace(
                1,
                config["NUM_UPDATES"],
                config["NUM_CHECKPOINTS"],
                dtype=int,
            )
        )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"], preprocess_obs=False
        )  # batched env for testing (has different batch size)

        # to initalize some variables is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim

        # INIT NETWORK AND OPTIMIZER
        network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        mixer = MixingNetwork(
            config["MIXER_EMBEDDING_DIM"],
            config["MIXER_HYPERNET_HIDDEN_DIM"],
            config["MIXER_INIT_SCALE"],
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size),
                    dtype=jnp.float32
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1), dtype=jnp.float32),  # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            agent_params = network.init(rng, init_hs, *init_x)

            # init mixer
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros((len(env.agents), 1, 1)) # q vals: agents, time, batch
            state_size = sample_traj.obs["__all__"].shape[
                -1
            ]  # get the state shape from the buffer
            init_state = jnp.zeros((1, 1, state_size)) # (time_step, batch_size, obs_size)
            mixer_params = mixer.init(_rng, init_x, init_state)

            network_params = {'agent':agent_params, 'mixer':mixer_params}

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=network_params,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)

        # INIT BUFFER
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=int(config["BUFFER_SIZE"] // config["NUM_ENVS"]),
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, test_state, eval_raw_returns, rng, rnorm = (
                runner_state
            )

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state.params['agent'],
                    hs,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x:config.get("REW_SCALE", 1)*x, rewards),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng, rnorm = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                # _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params['agent'],
                    init_hs,
                    _obs,
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                # --- REWARD STANDARDIZATION ---
                rewards = minibatch.rewards["__all__"][:-1]
                rewards_flat = jnp.ravel(rewards)
                new_rnorm = update_running_stats(rnorm, rewards_flat)
                std = get_std(new_rnorm)
                rewards_norm = (rewards - new_rnorm.mean) / (std + 1e-8)

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params['agent'],
                        init_hs,
                        _obs,
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    # get the q values of the next state
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    qmix_next = mixer.apply(train_state.target_network_params['mixer'], q_next, minibatch.obs["__all__"])
                    qmix_target = (
                        rewards_norm
                        + (
                            1 - minibatch.dones["__all__"][:-1]
                        )  # use next done because last done was saved for rnn re-init
                        * config["GAMMA"]
                        * qmix_next[1:]  # sum over agents
                    )

                    qmix = mixer.apply(params['mixer'], chosen_action_q_vals, minibatch.obs["__all__"])[:-1]
                    loss = jnp.mean(
                        (qmix - jax.lax.stop_gradient(qmix_target)) ** 2
                    )

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng, new_rnorm), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng, rnorm), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng, rnorm: jax.lax.scan(
                    _learn_phase, (train_state, rng, rnorm), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng, rnorm: (
                    (train_state, rng, rnorm),
                    (
                        jnp.zeros(config["NUM_EPOCHS"], dtype=jnp.float32),
                        jnp.zeros(config["NUM_EPOCHS"], dtype=jnp.float32),
                    ),
                ),  # do nothing
                train_state,
                _rng,
                rnorm,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                "rnorm_mean": rnorm.mean,
                "rnorm_std": get_std(rnorm),
                "rnorm_count": rnorm.count,
            }

            log_wrapper_infos = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                {
                    "returned_episode": infos["returned_episode"], 
                    "returned_episode_lengths": infos["returned_episode_lengths"],
                    "returned_episode_returns": infos["returned_episode_returns"],
                },
            )

            metrics.update(log_wrapper_infos)

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state, eval_raw_returns = jax.lax.cond(
                    jnp.any(train_state.n_updates == config["EVAL_UPDATES"]),
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: (test_state, eval_raw_returns),
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

                # STORE RAW EVAL RETURNS
                if config.get("STORE_EVAL_RETURNS", False):
                    jax.lax.cond(
                        jnp.any(train_state.n_updates == config["EVAL_UPDATES"]),
                        lambda: jax.debug.callback(
                            store_eval_returns,
                            original_seed,
                            eval_raw_returns,
                            train_state.n_updates,
                        ),
                        lambda: None,
                    )

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                        )
                    metrics_conversion = {k:float(v) for k,v in metrics.items()}
                    try:
                         metrics_conversion["gpu_stats"] = jax.devices()[0].memory_stats()
                    except IndexError:
                         pass
                    wandb.log(metrics_conversion, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            # CHECKPOINTING
            if config.get("SAVE_CHECKPOINTS", False):
                jax.lax.cond(
                    jnp.any(train_state.n_updates == config["CHECKPOINT_UPDATES"]),
                    lambda _: jax.debug.callback(
                        checkpoint_model,
                        original_seed,
                        train_state,
                        train_state.n_updates,
                        rnorm,
                    ),
                    lambda _: None,
                    operand=None,
                )

            runner_state = (
                train_state,
                buffer_state,
                test_state,
                eval_raw_returns,
                rng,
                rnorm,
            )

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None, None

            params = train_state.params['agent']
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )

            # Raw per-episode returns for one agent (all agents share the same
            # reward). Episodes are synchronized across envs by the fixed
            # horizon, so they can be sliced at those boundaries instead of
            # masked, avoiding a dynamic shape.
            raw_returns = infos["returned_episode_returns"][
                episode_horizon - 1 :: episode_horizon, :, 0
            ].reshape(-1)

            return metrics, raw_returns

        store_eval_returns = make_store_eval_returns(config)

        def checkpoint_model(vmapped_seed, train_state, step, rnorm):
            save_dir = os.path.join(
                config["HYDRA_PATH"] ,
                'checkpoints',
                str(vmapped_seed)
            )
            os.makedirs(save_dir, exist_ok=True)

            update_step_length = int(np.ceil(np.log10(config["NUM_UPDATES"])))

            save_path = os.path.join(save_dir,f'checkpoint_{step:0{update_step_length}}.safetensors')            

            params = train_state.params

            metadata = {
                "rnorm": 
                {
                    "mean": rnorm.mean.item(),
                    "M2": rnorm.M2.item(),
                    "count": rnorm.count.item(),
                }
            }

            params_to_save = {
                "params": params,
                "metadata": metadata,
            }

            log.info(f"Saving checkpoint {save_path}")
            save_params(params_to_save, save_path)

        rng, _rng = jax.random.split(rng)
        test_state, eval_raw_returns = get_greedy_metrics(_rng, train_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            buffer_state,
            test_state,
            eval_raw_returns,
            _rng,
            init_running_stats(),
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": None}

    return train


def env_from_config(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    return env, config["ENV_NAME"]


def single_run(config):
    alg_name = config.get("ALG_NAME", "qmix_rnn")

    map_name = config["ENV_KWARGS"].get("map_name", "default")

    config["HYDRA_PATH"] = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    wandb.init(
        entity=config["ENTITY"],
        project=f"{config['PROJECT']}_{map_name}",
        tags=[
            alg_name.upper(),
            f"jax_{jax.__version__}",
        ],
        config=config,
        mode=config["WANDB_MODE"],
        dir=config.get("WANDB_DIR", None),
    )

    # update the default params in case of overriding
    for k, v in dict(wandb.config).items():
        config[k] = v

    env, env_name = env_from_config(copy.deepcopy(config))

    if config["SEED"] == "random":
        config["SEED"] = np.random.randint(0, 2**32 - 1)
    

    map_name = config["ENV_KWARGS"].get("map_name", "default")

    wandb.run.name = f"{alg_name}_{env_name}_{map_name}_{config['SEED']}"
    wandb.config.update(config, allow_val_change=True)

    print("Config:\n", OmegaConf.to_yaml(config))

    # Save actual config
    config["WANDB_RUN_ID"] = wandb.run.id
    config["WANDB_RUN_URL"] = wandb.run.url
    config["WANDB_RUN_NAME"] = wandb.run.name

    OmegaConf.save(
        config,
        os.path.join(
            config["HYDRA_PATH"] ,
            'config.yaml'
        ),
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    env_name = default_config["ENV_NAME"]
    alg_name = default_config.get("ALG_NAME", "qmix_rnn")
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        if config["SEED"] == "random":
            seed = np.random.randint(0, 2**32 - 1)
            config["SAMPLED_SEED"] = seed
        else:
            seed = config["SEED"]
        
        wandb.config.update(config)

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(seed)
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    map_name = default_config["ENV_KWARGS"].get("map_name", "default")

    sweep_config = {
        "name": f"{alg_name}_{env_name}_{map_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    1e-1,
                    1e-2,
                    1e-3,
                    1e-4,
                ]
            },
            "NUM_ENVS": {"values": [1, 4, 8, 16]},
            "BUFFER_BATCH_SIZE": {
                "values": [16, 32, 64, 128],
            },
            "BUFFER_SIZE": {
                "values": [
                    5000,
                    2000,
                ]
            },
            "EPS_FINISH": {
                "values": [0.05, 0.01, 0.001, 0],
            },
            "NUM_EPOCHS": {
                "values": [1, 2, 4, 8],
            },
            "HIDDEN_SIZE": {
                "values": [32, 64, 128],
            },
            "GAMMA": {
                "values": [0.99, 0.999, 1.0],
            },
            "TARGET_UPDATE_INTERVAL" : {
                "values": [8, 16, 32, 64],
            },
            "MIXER_EMBEDDING_DIM": {
                "values": [32, 64, 128],
            },
            "MIXER_HYPERNET_HIDDEN_DIM": {
                "values": [32, 64, 128],
            },
            "MIXER_INIT_SCALE": {
                "values": [0.0001, 0.001, 0.01],
            },
            "REW_SCALE": {
                "values": [1e-8, 1e-9, 1e-10],
            },
            "TAU": {
                "values": [1, 0.99, 0.9],
            }
        }
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)


@hydra.main(version_base=None, config_path="./config", config_name="qmix_rnn_road_env")
def main(config):
    config = OmegaConf.to_container(config)

    print(jax.devices())

    if config.get("DOUBLE_PRECISION_MODE", False):
        jax.config.update("jax_enable_x64", True)
        log.info("64 bit precision enabled")

    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
