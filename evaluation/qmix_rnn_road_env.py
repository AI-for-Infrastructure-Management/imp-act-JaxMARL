import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
import logging
from functools import partial
from typing import Any
import yaml

import chex
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf

from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
    load_params
)

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


@chex.dataclass
class RunningStats:
    count: jnp.ndarray
    mean: jnp.ndarray
    M2: jnp.ndarray

def init_running_stats():
    return RunningStats(
        count=jnp.array(0.0, dtype=jnp.float32),
        mean=jnp.array(0.0, dtype=jnp.float32),
        M2=jnp.array(0.0, dtype=jnp.float32),
    )

def update_running_stats(stats: RunningStats, x: jnp.ndarray) -> RunningStats:
    x = x.astype(jnp.float32)
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

def env_from_config(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    return env, config["ENV_NAME"]

def load_checkpoint_agent(checkpoint_path, step, vmapped_seed, config):
   
    update_step_length = int(np.ceil(np.log10(config["NUM_UPDATES"])))

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    load_path = os.path.join(
                checkpoint_path,
                'checkpoints',
                str(rngs[vmapped_seed][0].item()),
                f'checkpoint_{step:0{update_step_length}}.safetensors',
            )

    loaded_params = load_params(load_path)

    metadata = loaded_params["metadata"]

    rnorm = RunningStats(
        count=jnp.array(metadata["rnorm"]["count"], dtype=jnp.int32),
        mean=jnp.array(metadata["rnorm"]["mean"], dtype=jnp.float32),
        M2=jnp.array(metadata["rnorm"]["M2"], dtype=jnp.float32),
    )

    network = RNNQNetwork(
        action_dim=config["MAX_ACTION_SPACE"],
        hidden_dim=config["HIDDEN_SIZE"],
    )

    mixer = MixingNetwork(
        config["MIXER_EMBEDDING_DIM"],
        config["MIXER_HYPERNET_HIDDEN_DIM"],
        config["MIXER_INIT_SCALE"],
    )

    return loaded_params, rnorm, network, mixer


def make_get_greedy_metrics(train_config, test_num_envs, test_num_steps):
    env, env_name = env_from_config(copy.deepcopy(train_config))
    env = CTRolloutManager(env, batch_size=test_num_envs, preprocess_obs=False)

    network = RNNQNetwork(
        action_dim=env.max_action_space,
        hidden_dim=train_config["HIDDEN_SIZE"],
    )

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def get_greedy_metrics(rng, loaded_params): 
            params = loaded_params["params"]['agent']  
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
                valid_actions = env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((test_num_envs), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                train_config["HIDDEN_SIZE"], len(env.agents), test_num_envs
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
                _greedy_env_step, step_state, None, test_num_steps
            )

            return infos
    
    return get_greedy_metrics


def evaluate_checkpoint(config_eval):
    rng = jax.random.PRNGKey(config_eval.get("SEED"))
    checkpoint_path = config_eval["CHECKPOINT_PATH"]
    step = config_eval["STEP"]
    vmapped_seed = config_eval.get("VMAPPED_SEED", 0)

    # load YAML
    config = OmegaConf.load(os.path.join(checkpoint_path, "config.yaml"))
    config = OmegaConf.to_container(config)

    env, env_name = env_from_config(copy.deepcopy(config))

    env = CTRolloutManager(env, batch_size=config_eval["TEST_NUM_ENVS"], preprocess_obs=False)

    config["MAX_ACTION_SPACE"] = env.max_action_space
    
    loaded_params, rnorm, network, mixer = load_checkpoint_agent(checkpoint_path, step, vmapped_seed, config)

    get_greedy_metrics = make_get_greedy_metrics(
        config, config_eval["TEST_NUM_ENVS"], config_eval["TEST_NUM_STEPS"]
    )

    rng, _rng = jax.random.split(rng)
    infos = get_greedy_metrics(_rng, loaded_params)
    
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
    metrics["episodes"] = config_eval["TEST_NUM_ENVS"]
    
    log.info(f"Evaluation metrics for checkpoint at step {step} with {config_eval['TEST_NUM_ENVS']} envs:")
    log.info(f"  Environment: {env_name}")
    for k, v in metrics.items():
        log.info(f"  {k}: {float(v)}")
    
    return metrics
    
@hydra.main(version_base=None, config_path="./config", config_name="eval_qmix_rnn_road_env")
def main(config):
    config = OmegaConf.to_container(config)
    
    print("Config:")
    print(OmegaConf.to_yaml(config))
    
    metrics = evaluate_checkpoint(config)

    # Save metrics to a file
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    metrics_file_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_file_path, 'w') as f:
        yaml.dump({k: float(v) for k, v in metrics.items()}, f)

    return metrics

if __name__ == "__main__":
    main()