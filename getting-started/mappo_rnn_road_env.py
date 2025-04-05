"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.

Adapted from: baselines/MAPPO/mappo_rnn_smax.py
"""

import os
import copy
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
import wandb
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
from functools import partial

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper


class RoadEnvWorldStateWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)
    """

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state_fn(self, obs, state):
        return obs["__all__"][None].repeat(self._env.num_agents, axis=0)

    def world_state_size(self):
        return self._env.world_state_size


class ScannedRNN(nn.Module):
    @functools.partial(
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

        # Ensure inputs are cast to float32
        ins = ins.astype(jnp.float32)
        resets = resets.astype(jnp.float32)

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )

        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)

        # Ensure outputs are cast to float32
        new_rnn_state = new_rnn_state.astype(jnp.float32)
        y = y.astype(jnp.float32)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def running_mean_std(mean, var, count, new_data):
    """
    Update running mean and variance with new data.

    TODO: For now 1D array, Generalize to N-D array

    Args:
    - mean (jnp.array): Current running mean.
    - var (jnp.array): Current running variance.
    - count (float): Current sample count.
    - new_data (jnp.array): Incoming batch of data.

    Returns:
    - new_mean (jnp.array): Updated mean.
    - new_var (jnp.array): Updated variance.
    - new_count (float): Updated sample count.

    Adapted from: https://github.com/uoe-agents/epymarl/blob/main/src/components/standarize_stream.py
    """
    batch_mean = jnp.mean(new_data, axis=0)
    batch_var = jnp.var(new_data, axis=0)
    batch_count = new_data.shape[0]

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + (delta * batch_count) / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + (delta**2) * (count * batch_count) / tot_count
    new_var = m_2 / tot_count

    return new_mean, new_var, tot_count


def make_train(config):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = RoadEnvWorldStateWrapper(env)
    env = LogWrapper(env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(lr, ent_coeff, rng):

        # add lr to config
        config["LR"] = lr
        config["ENT_COEF"] = ent_coeff

        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        _ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(
            _rng_actor, _ac_init_hstate, ac_init_x
        )
        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    env.world_state_size(),
                )
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        _cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, _cr_init_hstate, cr_init_x
        )

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # INIT REWARD STANDARDIZATION
        reward_mean, reward_var, reward_count = (
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            0,
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps, eval_metrics = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    hstates,
                    reward_standardization,
                    rng,
                ) = runner_state

                reward_mean, reward_var, reward_count = reward_standardization

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                # print('env step ac in', ac_in)
                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))

                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                rewards_ = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                if config["REWARD_STANDARDIZATION"]:
                    reward_mean, reward_var, reward_count = running_mean_std(
                        reward_mean, reward_var, reward_count, rewards_
                    )
                    rewards_ = (rewards_ - reward_mean) / jnp.sqrt(reward_var + 1e-8)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    rewards_,
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    (reward_mean, reward_var, reward_count),
                    rng,
                )
                return runner_state, transition

            reward_standardization = runner_state[-2]
            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                reward_standardization,
                rng,
            ) = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))

            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # NETWORKS UPDATE
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN NETWORK
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # UPDATE METRICS
            train_states = update_state[0]
            metrics = traj_batch.info
            metrics = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metrics["loss"] = loss_info
            rng = update_state[-1]

            # EVALUATION
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)

                def eval_and_store_returns(rng, train_states):
                    val = get_greedy_metrics(rng, train_states)[
                        "returned_episode_returns"
                    ]
                    idx = update_steps // int(
                        config["NUM_UPDATES"] * config["TEST_INTERVAL"]
                    )
                    return eval_metrics.at[idx].set(val)

                eval_metrics = jax.lax.cond(
                    update_steps % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: eval_and_store_returns(_rng, train_states),
                    lambda _: eval_metrics,
                    operand=None,
                )
                # metrics.update({"test_" + k: v for k, v in eval_metrics.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    # Filter the dictionary to only include keys in METRICS_TO_LOG
                    filtered_metrics = {
                        key: metrics[key]
                        for key in config["METRICS_TO_LOG"]
                        if key in metrics
                    }
                    wandb.log(filtered_metrics)

                jax.debug.callback(callback, metrics, original_seed)

            metrics["update_steps"] = update_steps
            update_steps = update_steps + 1
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                reward_standardization,
                rng,
            )
            return (runner_state, update_steps, eval_metrics), metrics

        def get_greedy_metrics(rng, train_states):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            config["TEST_NUM_ACTORS"] = config["TEST_NUM_ENVS"] * env.num_agents

            def _greedy_env_step(step_state, unused):
                train_states, env_state, last_obs, last_done, ac_hstate, rng = (
                    step_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["TEST_NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, ac_hstate, ac_in
                )
                action = pi.sample(seed=_rng)
                env_act = unbatchify(
                    action, env.agents, config["TEST_NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["TEST_NUM_ENVS"])
                obsv, env_state, rewards, dones, infos = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(
                    lambda x: x.reshape((config["TEST_NUM_ACTORS"])), infos
                )
                done_batch = batchify(
                    dones, env.agents, config["TEST_NUM_ACTORS"]
                ).squeeze()

                step_state = (train_states, env_state, obsv, done_batch, ac_hstate, rng)
                return step_state, (rewards, dones, infos)

            reset_rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            init_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            ac_init_hstate = ScannedRNN.initialize_carry(
                config["TEST_NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            done_batch = jnp.zeros((config["TEST_NUM_ACTORS"],), dtype=jnp.bool_)
            step_state = (
                train_states,
                env_state,
                init_obs,
                done_batch,
                ac_init_hstate,
                rng,
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
            return metrics

        rng, _rng = jax.random.split(rng)

        # Storing evaluation metrics on CPU to save GPU memory
        num_evals = int(1 / config["TEST_INTERVAL"])
        initial_eval_metrics = jnp.zeros((num_evals,), device="cpu")

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            (reward_mean, reward_var, reward_count),
            _rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step,
            (runner_state, 0, initial_eval_metrics),
            None,
            config["NUM_UPDATES"],
        )
        return {"runner_state": runner_state, "metrics": None}

    return train


def single_run(config):

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    ## Plotting

    import matplotlib.pyplot as plt

    plt.plot(outs["metrics"]["test_returned_episode_returns"].T / 1e6)
    plt.xlabel("Updates")
    plt.ylabel("Returns")
    plt.title("MAPPO with RNN on RoadEnv")
    plt.savefig("mappo_rnn_road_env.png", dpi=300)
    plt.show()

    # save model params
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        env_name = config["ENV_NAME"]
        alg_name = config["ALG_NAME"]
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with vmap over seeds and hyperparameters."""

    import time

    start_time = time.time()

    ## Define the hyperparameters to search
    hypers_to_search = {
        "lr": [0.001, 0.005, 0.0005],
        "ent_coeff": [0.0],
    }
    all_hypers = [jnp.array(v) for v in hypers_to_search.values()]
    # cartesian product of all hyperparameters to create a grid
    # hypers_grid shape: (num_combinations, num_hyperparams)
    _grids = jnp.meshgrid(*all_hypers, indexing="ij")
    hypers_grid = jnp.stack([*_grids], axis=-1).reshape(-1, len(all_hypers))
    hypers_split = [hypers_grid[:, i] for i in range(hypers_grid.shape[1])]

    ## seeds for each hyperparameter
    rng = jax.random.PRNGKey(default_config["SEED"])
    rngs = jax.random.split(rng, default_config["NUM_SEEDS"])

    ## create a vmap over the hyperparameters and seeds
    #! Manually adjust the inputs in the train function
    #! and Nones and 0s in the vmaps to match the input shapes
    vmap_seeds = jax.vmap(make_train(default_config), in_axes=(None, None, 0))
    vmap_hypers = jax.vmap(vmap_seeds, in_axes=(0, 0, None))
    train_vjit = jax.jit(vmap_hypers)
    # outs shape: (num_combinations, num_seeds, num_updates)
    outs = jax.block_until_ready(train_vjit(*hypers_split, rngs))

    test_returns = outs["runner_state"][-1] / 1e6
    num_combinations, num_seeds, num_updates = test_returns.shape

    # Aggregate over seeds
    # shape: (num_combinations, num_updates)
    test_returns_mean = jnp.mean(test_returns, axis=1)
    test_returns_std = jnp.std(test_returns, axis=1)

    # get the best hyperparameter(s)
    test_returns_max = jnp.max(test_returns_mean, axis=1)
    idx_best_comb = jnp.argmax(test_returns_max)
    best_hyper = hypers_grid[idx_best_comb]

    # print results
    print("-" * 50)
    print(f"Best combination: {best_hyper}, index: {idx_best_comb}")
    print(f"Best combination mean: {test_returns_mean[idx_best_comb].mean():.2f}")
    print(f"Best combination std: {test_returns_std[idx_best_comb].mean():.2f}")

    print()
    print("Summary of all combinations:")
    print("-" * 25)
    print(f"# combinations: {num_combinations}")
    print(f"# seeds (per combination): {num_seeds}")
    print("Hyperparameters grid:")
    for i, hyp in enumerate(hypers_grid):
        _label = [f"{k}: {v}" for k, v in zip(hypers_to_search.keys(), hyp)]
        print(
            f"{i}: {_label} | mean: {test_returns_mean[i].mean():.2f}, std: {test_returns_std[i].mean():.2f}"
        )

    # plot
    import matplotlib.pyplot as plt

    env_name = default_config["ENV_NAME"]
    alg_name = default_config["ALG_NAME"]

    for h, hyp in enumerate(hypers_grid):
        _label = [f"{k}: {v}" for k, v in zip(hypers_to_search.keys(), hyp)]
        plt.plot(test_returns_mean[h], label=_label)
        plt.fill_between(
            np.arange(num_updates),
            test_returns_mean[h] - test_returns_std[h],
            test_returns_mean[h] + test_returns_std[h],
            alpha=0.2,
        )
    plt.title(f"Hyperparameter Search | ENV: {env_name}, ALG:{alg_name}")
    plt.xlabel("Eval Chckpts")
    plt.ylabel("Returns")
    plt.legend()
    plt.savefig("hypers.png", dpi=300)
    plt.show()

    total_time = time.time() - start_time

    print(f"Total time:{total_time:.2f}")


@hydra.main(version_base=None, config_path="config", config_name="mappo_rnn_road_env")
def main(config):

    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
