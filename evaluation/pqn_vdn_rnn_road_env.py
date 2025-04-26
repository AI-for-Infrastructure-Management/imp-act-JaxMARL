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
# from flax.linen.initializers import constant, orthogonal
# from gymnax.wrappers.purerl import LogWrapper
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


class QNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = False

    @nn.compact
    def __call__(self, hidden, x, dones, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        rnn_in = (x, dones)
        hidden, x = ScannedRNN()(hidden, rnn_in)

        if self.dueling:
            adv = nn.Dense(self.action_dim)(x)
            val = nn.Dense(1)(x)
            q_vals = val + adv - jnp.mean(adv, axis=-1, keepdims=True)
        else:
            q_vals = nn.Dense(self.action_dim)(x)

        return hidden, q_vals

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

    params = loaded_params["params"]
    batch_stats = loaded_params["batch_stats"]

    # rnorm = RunningStats(
    #     count=jnp.array(metadata["rnorm"]["count"], dtype=jnp.int32),
    #     mean=jnp.array(metadata["rnorm"]["mean"], dtype=jnp.float32),
    #     M2=jnp.array(metadata["rnorm"]["M2"], dtype=jnp.float32),
    # )

    network = QNetwork(
        action_dim=config["MAX_ACTION_SPACE"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        dueling=config.get("DUELING", False),
    )

    return params, batch_stats, network

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
    
    params, batch_stats, network = load_checkpoint_agent(checkpoint_path, step, vmapped_seed, config)

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def get_greedy_metrics(rng, params, batch_stats): 

            def _greedy_env_step(step_state, unused):
                params, bach_stats, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(
                    partial(network.apply), in_axes=(None, 0, 0, 0, None)
                )(
                    {
                        "params": params,
                        "batch_stats": batch_stats,
                    },
                    hstate,
                    _obs,
                    _dones,
                    False,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, bach_stats, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config_eval["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config_eval["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params,
                batch_stats,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config_eval["TEST_NUM_STEPS"]
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
            metrics["episodes"] = config_eval["TEST_NUM_ENVS"]
            return metrics
    
    rng, _rng = jax.random.split(rng)
    metrics = get_greedy_metrics(_rng, params, batch_stats)
    
    log.info(f"Evaluation metrics for checkpoint at step {step} with {config_eval['TEST_NUM_ENVS']} envs:")
    log.info(f"  Environment: {env_name}")
    for k, v in metrics.items():
        log.info(f"  {k}: {float(v)}")
    
    return metrics
    
@hydra.main(version_base=None, config_path="./config", config_name="eval_pqn_rnn_road_env")
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