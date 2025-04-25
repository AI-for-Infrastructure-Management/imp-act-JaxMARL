"""
Based on PureJaxRL Implementation of PPO

Adapted from: baselines/IPPO/ippo_rnn_smax.py
"""

import os
import logging
import functools
import copy
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import wandb
import distrax
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, save_params

# Get Hydra's logger
log = logging.getLogger(__name__)


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
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)

        # Ensure inputs are cast to float32
        new_rnn_state = new_rnn_state.astype(jnp.float32)
        y = y.astype(jnp.float32)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
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

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


@chex.dataclass
class EvalMetricsManager:
    # reported with wandb.log during training
    logged_eval_metrics: dict
    # reported with wandb.run.summary at the end of training
    eval_returns: jnp.ndarray


@chex.dataclass
class RunningStats:
    count: jnp.ndarray  # or float32
    mean: jnp.ndarray
    var: jnp.ndarray


def init_running_stats():
    return RunningStats(
        count=jnp.array(0.0, dtype=jnp.float32),
        mean=jnp.array(0.0, dtype=jnp.float32),
        var=jnp.array(0.0, dtype=jnp.float32),
    )


def update_running_stats(stats: RunningStats, x: jnp.ndarray) -> RunningStats:
    x = x.astype(jnp.float32)  # if x might be float64
    batch_count = x.size
    batch_mean = jnp.mean(x)
    batch_var = jnp.var(x)

    delta = batch_mean - stats.mean
    new_count = stats.count + batch_count
    new_mean = stats.mean + (delta * batch_count) / new_count
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + (delta**2) * (stats.count * batch_count) / new_count
    new_var = m_2 / new_count

    return RunningStats(count=new_count, mean=new_mean, var=new_var)


def apply_normalization(stats: RunningStats, x):
    return (x - stats.mean) / jnp.sqrt(stats.var + 1e-8)


def make_train(config):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["TOTAL_TIMESTEPS"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] * config["NUM_UPDATES"]
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

    def train(rng):

        original_seed = rng[0]

        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps, metrics_manager = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rnorm,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                rewards_flat = batchify(
                    reward, env.agents, config["NUM_ACTORS"]
                ).squeeze()
                if config["REWARD_STANDARDIZATION"]:
                    new_rnorm = update_running_stats(rnorm, rewards_flat)
                    rewards_flat = apply_normalization(new_rnorm, rewards_flat)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    rewards_flat,
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    hstate,
                    new_rnorm,
                    rng,
                )
                return runner_state, transition

            rnorm = runner_state[-2]
            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rnorm,
                rng,
            ) = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (
                    init_hstate,
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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # UPDATE METRICS
            train_state = update_state[0]
            rng = update_state[-1]
            metrics = {
                "update_steps": update_steps,
                "env_step": (
                    (update_steps + 1) * config["NUM_ENVS"] * config["NUM_STEPS"]
                ),
                # loss info
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
                # reward normalization
                "rnorm_mean": rnorm.mean,
                "rnorm_std": jnp.sqrt(rnorm.var),
                "rnorm_count": rnorm.count,
            }
            log_wrapper_infos = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        traj_batch.info["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                {
                    "returned_episode": traj_batch.info["returned_episode"],
                    "returned_episode_lengths": traj_batch.info[
                        "returned_episode_lengths"
                    ],
                    "returned_episode_returns": traj_batch.info[
                        "returned_episode_returns"
                    ],
                },
            )
            metrics.update(log_wrapper_infos)

            # EVALUATION
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)

                def eval_and_store_returns(rng, train_state):
                    eval_metrics = get_greedy_metrics(rng, train_state)
                    idx = update_steps // int(
                        config["NUM_UPDATES"] * config["TEST_INTERVAL"]
                    )
                    _metrics_manager = EvalMetricsManager(
                        logged_eval_metrics=eval_metrics,
                        eval_returns=metrics_manager.eval_returns.at[idx].set(
                            eval_metrics["returned_episode_returns"]
                        ),
                    )
                    return _metrics_manager

                metrics_manager = jax.lax.cond(
                    update_steps % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: eval_and_store_returns(_rng, train_state),
                    lambda _: metrics_manager,
                    operand=None,
                )
                metrics.update(
                    {
                        "test_" + k: v
                        for k, v in metrics_manager.logged_eval_metrics.items()
                    }
                )

            # CHECKPOINTING
            if config.get("SAVE_CHECKPOINTS", False):

                jax.lax.cond(
                    update_steps
                    % int(config["NUM_UPDATES"] * config["SAVE_CHECKPOINTS_INTERVAL"])
                    == 0,
                    lambda _: jax.debug.callback(
                        checkpoint_model,
                        original_seed,
                        train_state,
                        update_steps,
                    ),
                    lambda _: None,
                    operand=None,
                )

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
                    metrics_conversion = {k: float(v) for k, v in metrics.items()}
                    wandb.log(metrics_conversion, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            update_steps = update_steps + 1
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rnorm,
                rng,
            )
            return (runner_state, update_steps, metrics_manager), metrics

        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            config["TEST_NUM_ACTORS"] = config["TEST_NUM_ENVS"] * env.num_agents

            def _greedy_env_step(step_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = step_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["TEST_NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, _ = network.apply(train_state.params, hstate, ac_in)
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
                infos = jax.tree.map(
                    lambda x: x.reshape((config["TEST_NUM_ACTORS"])), infos
                )
                done_batch = batchify(
                    dones, env.agents, config["TEST_NUM_ACTORS"]
                ).squeeze()

                step_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return step_state, (rewards, dones, infos)

            reset_rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            init_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            init_hstate = ScannedRNN.initialize_carry(
                config["TEST_NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            done_batch = jnp.zeros((config["TEST_NUM_ACTORS"],), dtype=jnp.bool_)
            step_state = (
                train_state,
                env_state,
                init_obs,
                done_batch,
                init_hstate,
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

        def checkpoint_model(vmapped_seed, train_state, step):
            save_dir = os.path.join(
                config["HYDRA_PATH"],
                "checkpoints",
                str(vmapped_seed),
            )
            os.makedirs(save_dir, exist_ok=True)

            update_step_length = int(np.ceil(np.log10(config["NUM_UPDATES"])))

            save_path = os.path.join(
                save_dir, f"checkpoint_{step:0{update_step_length}}.safetensors"
            )

            params = jax.tree.map(lambda x: x, train_state.params)
            log.info(f"Saving checkpoint {save_path}")
            save_params(params, save_path)

        rng, _rng = jax.random.split(rng)

        # Metrics Manager
        num_evals = int(1 / config["TEST_INTERVAL"])
        metrics_manager = EvalMetricsManager(
            logged_eval_metrics=get_greedy_metrics(_rng, train_state),
            eval_returns=jnp.zeros((num_evals,), device="cpu"),
        )

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            init_running_stats(),
            _rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step,
            (runner_state, 0, metrics_manager),
            None,
            config["NUM_UPDATES"],
        )
        return {"runner_state": runner_state, "metrics": None}

    return train


def single_run(config):

    alg_name = config.get("ALG_NAME", "mappo_rnn")

    env_name = config.get("env_name", "default")
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
    )

    # update the default params in case of overriding
    for k, v in dict(wandb.config).items():
        config[k] = v

    # embedding size for the GRU, must be same as the GRU hidden size
    config["FC_DIM_SIZE"] = config["GRU_HIDDEN_DIM"]
    config["TOTAL_TIMESTEPS"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] * config["NUM_UPDATES"]
    )

    if config["SEED"] == "random":
        config["SEED"] = np.random.randint(0, 2**32 - 1)

    wandb.run.name = f"{alg_name}_{env_name}_{map_name}_{config['SEED']}"
    wandb.config.update(config, allow_val_change=True)

    print("Config:\n", OmegaConf.to_yaml(config))

    # Save actual config
    config["WANDB_RUN_ID"] = wandb.run.id
    config["WANDB_RUN_URL"] = wandb.run.get_url()
    config["WANDB_RUN_NAME"] = wandb.run.name

    OmegaConf.save(config, os.path.join(config["HYDRA_PATH"], "config.yaml"))

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # wandb summary
    metrics_manager = outs["runner_state"][-1]
    wandb.run.summary["eval_returns"] = list(metrics_manager.eval_returns)

    # save params
    if config.get("SAVE_CHECKPOINTS", False):
        model_state = outs["runner_state"][0][0]
        save_dir = os.path.join(config["HYDRA_PATH"], "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        for s in range(config["NUM_SEEDS"]):
            params = jax.tree.map(lambda x: x[s], model_state.params)
            save_path = os.path.join(
                save_dir,
                str(rngs[s][0].item()),
                f"checkpoint_final.safetensors",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            log.info(f"Saving final checkpoint {save_path}")
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    alg_name = default_config["ALG_NAME"]
    map_name = default_config["ENV_KWARGS"]["map_name"]

    def wrapped_make_train():
        run = wandb.init(project=default_config["PROJECT"], tags=[alg_name, map_name])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        config["FC_DIM_SIZE"] = config["GRU_HIDDEN_DIM"]

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{map_name}",
        "method": "grid",
        "metric": {
            "name": "test_eval_return",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.001, 0.0005]},
            # "NUM_ENVS": {"values": [8, 32, 64, 128]},
            # "ENT_COEF": {"values": [0, 0.01]},
            # FC_DIM_SIZE is set to GRU_HIDDEN_DIM in above
            # "GRU_HIDDEN_DIM": {"values": [32, 64, 128]},
            # "NUM_MINIBATCHES": {"values": [2, 4, 8]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 6]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=50)


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_road_env")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()


def load_checkpoint_agent(hydra_path, rng, checkpoint_id):
    from jaxmarl.wrappers.baselines import load_params

    # load YAML
    config = OmegaConf.load(os.path.join(hydra_path, "config.yaml"))
    config = OmegaConf.to_container(config)

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)

    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    params = load_params(
        os.path.join(
            hydra_path,
            "checkpoints",
            str(rng),
            f"checkpoint_{str(checkpoint_id)}.safetensors",
        )
    )
    params = jax.tree.map(lambda x: x, params)

    return network, params
