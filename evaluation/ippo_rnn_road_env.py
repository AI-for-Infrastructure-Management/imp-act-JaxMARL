import os
import logging
import yaml
import jax
import chex
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Dict
import functools
import distrax
import hydra
from omegaconf import OmegaConf
from functools import partial

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper, load_params

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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


@chex.dataclass
class RunningStats:
    count: jnp.ndarray  # or float32
    mean: jnp.ndarray
    var: jnp.ndarray


def load_checkpoint_agent(checkpoint_path, step, vmapped_seed, config):

    update_step_length = int(np.ceil(np.log10(config["NUM_UPDATES"])))

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    load_path = os.path.join(
        checkpoint_path,
        "checkpoints",
        str(rngs[vmapped_seed][0].item()),
        f"checkpoint_{step:0{update_step_length}}.safetensors",
    )

    loaded_params = load_params(load_path)
    params = loaded_params["params"]
    metadata = loaded_params["metadata"]

    rnorm = RunningStats(
        count=jnp.array(metadata["rnorm"]["count"], dtype=jnp.int32),
        mean=jnp.array(metadata["rnorm"]["mean"], dtype=jnp.float32),
        var=jnp.array(metadata["rnorm"]["var"], dtype=jnp.float32),
    )

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)

    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    return network, params, rnorm


def evaluate_checkpoint(config_eval):
    rng = jax.random.PRNGKey(config_eval.get("SEED"))
    checkpoint_path = config_eval["CHECKPOINT_PATH"]
    step = config_eval["STEP"]
    vmapped_seed = config_eval.get("VMAPPED_SEED", 0)

    # load YAML
    config = OmegaConf.load(os.path.join(checkpoint_path, "config.yaml"))
    config = OmegaConf.to_container(config)

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)

    network, params, rnorm = load_checkpoint_agent(
        checkpoint_path, step, vmapped_seed, config
    )

    def get_greedy_metrics(rng, params):

        config["TEST_NUM_ACTORS"] = config["TEST_NUM_ENVS"] * env.num_agents

        def _greedy_env_step(step_state, unused):
            env_state, last_obs, last_done, hstate, rng = step_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = batchify(last_obs, env.agents, config["TEST_NUM_ACTORS"])
            ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
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

            step_state = (env_state, obsv, done_batch, hstate, rng)
            return step_state, (rewards, dones, infos)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
        init_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["TEST_NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        done_batch = jnp.zeros((config["TEST_NUM_ACTORS"],), dtype=jnp.bool_)
        step_state = (
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

    rng, _rng = jax.random.split(rng)
    metrics = get_greedy_metrics(_rng, params)

    log.info(
        f"Evaluation metrics for checkpoint at step {step} with {config_eval['TEST_NUM_ENVS']} envs:"
    )
    log.info(f"  Environment: {config['ENV_NAME']}")
    for k, v in metrics.items():
        log.info(f"  {k}: {float(v)}")

    return metrics


@hydra.main(
    version_base=None, config_path="./config", config_name="eval_ippo_rnn_road_env"
)
def main(config):
    config = OmegaConf.to_container(config)

    print("Config:")
    print(OmegaConf.to_yaml(config))

    metrics = evaluate_checkpoint(config)

    # Save metrics to a file
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    metrics_file_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_file_path, "w") as f:
        yaml.dump({k: float(v) for k, v in metrics.items()}, f)

    return metrics


if __name__ == "__main__":
    main()
