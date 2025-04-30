import time
import sys
import importlib.util
from pathlib import Path

import os
import yaml
import pandas as pd
import jax
import numpy as np
import jax.numpy as jnp

from scipy.stats import bootstrap

import jaxmarl
from jaxmarl.wrappers.baselines import load_params


MAP_NAME = "ToyExample-v2"
ALGORITHMS = ["vdn_rnn", "qmix_rnn", "pqn_rnn", "mappo_rnn", "ippo_rnn"]
TEST_NUM_ENVS = 100
TEST_NUM_STEPS = 50
VMAPPED_SEED = 0
EVAL_SEED = 0
REWARD_SCALE = 1e6
BASE_PATH = os.getcwd()

# read YAML file with all checkpoints dir names
with open(f"{BASE_PATH}/inference/all_checkpoint_dirs.yaml", "r") as f:
    all_checkpoint_dirs = yaml.safe_load(f)


def import_function_from_path(script_path: str, function_name: str):
    script_path = Path(script_path).resolve()
    module_name = script_path.stem

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return getattr(module, function_name)


def compute_episode_return_stats(episode_returns):

    bst = bootstrap(
        data=[episode_returns],
        statistic=np.mean,
        confidence_level=0.95,
    )
    bst_std_err = bst.standard_error
    lower_ci = bst.confidence_interval.low
    upper_ci = bst.confidence_interval.high
    mean = jnp.mean(episode_returns)
    std_err = jnp.std(episode_returns) / np.sqrt(len(episode_returns))

    inference_stats = {
        "mean": mean,
        "std_err": std_err,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "bst_std_err": bst_std_err,
    }

    return inference_stats


############################ EVALUATION ################################

all_eval_stats = []

time_main_0 = time.time()

#! ALGORITHMS
# for a, alg in enumerate(ALGORITHMS):
for a, alg in enumerate(["mappo_rnn"]):  #! ALGORITHM: "mappo_rnn"
    chkpt_dirs_alg = all_checkpoint_dirs[MAP_NAME][alg]

    make_get_greedy_metrics = import_function_from_path(
        f"{BASE_PATH}/evaluation/{alg}_road_env.py",
        "make_get_greedy_metrics",
    )

    #! SEEDS
    # for i, chkpt_dir in enumerate(chkpt_dirs_alg):
    for i, chkpt_dir_name in enumerate([chkpt_dirs_alg[0]]):  #! SEED: 0

        train_config_path = (
            f"{BASE_PATH}/outputs/{MAP_NAME}/{alg}/{chkpt_dir_name}/config.yaml"
        )
        train_config = yaml.safe_load(open(train_config_path, "r"))
        rngs = jax.random.split(
            jax.random.PRNGKey(train_config["SEED"]), train_config["NUM_SEEDS"]
        )
        checkpoint_path = f"{BASE_PATH}/outputs/{MAP_NAME}/{alg}/{chkpt_dir_name}/checkpoints/{rngs[VMAPPED_SEED][0]}"

        env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])

        jit_get_greedy_metrics = jax.jit(
            make_get_greedy_metrics(train_config, TEST_NUM_ENVS, TEST_NUM_STEPS)
        )

        #! CHECKPOINTS
        for i, safetensor_name in enumerate(sorted(os.listdir(checkpoint_path))):
            time0 = time.time()

            checkpoint_id = safetensor_name.split(".")[0].split("_")[-1]
            safetensor_path = os.path.join(checkpoint_path, safetensor_name)
            loaded_params = load_params(safetensor_path)

            rng = jax.random.PRNGKey(EVAL_SEED)
            infos = jit_get_greedy_metrics(rng, loaded_params)

            # infos["returned_episode_returns"] | shape: (NUM_TIME_STEPS, NUM_ENVS, NUM_AGENTS)
            # This selects the episode returns for which returned_episode is True
            # and reshapes it to (NUM_EPISODES, NUM_AGENTS). This is required when NUM_ENVS
            # cannot fit in memory, and we must process them in batches. Since all agents
            # have same rewards, we only take the first agent's rewards.
            _episode_returns = infos["returned_episode_returns"][
                infos["returned_episode"]
            ]
            episode_returns = _episode_returns.reshape(-1, env.num_agents)[:, 0]

            inference_stats = compute_episode_return_stats(episode_returns)

            mean = inference_stats["mean"] / REWARD_SCALE
            _95_ci = (
                inference_stats["lower_ci"] / REWARD_SCALE,
                inference_stats["upper_ci"] / REWARD_SCALE,
            )

            all_eval_stats.append(
                {
                    "map_name": MAP_NAME,
                    "algorithm": alg,
                    "checkpoint_dir_name": chkpt_dir_name,
                    "WANDB_RUN_ID": train_config["WANDB_RUN_ID"],
                    "checkpoint_id": checkpoint_id,
                    "VMAPPED_SEED": VMAPPED_SEED,
                    "eval_seed": EVAL_SEED,
                    "mean": float(inference_stats["mean"]),
                    "std_err": float(inference_stats["std_err"]),
                    "lower_ci": float(inference_stats["lower_ci"]),
                    "upper_ci": float(inference_stats["upper_ci"]),
                    "bst_std_err": float(inference_stats["bst_std_err"]),
                }
            )

            print(
                f" {i+1:<3} | Checkpoint: {checkpoint_id:<4} | time: {time.time()-time0:.2f} | mean: {mean:.2f} | 95% CI: ({_95_ci[0]:.2f}, {_95_ci[1]:.2f})"
            )

print(f"Total time: {time.time()-time_main_0:.2f}")

df = pd.DataFrame(all_eval_stats)
df.to_csv("inference/inference_results.csv", index=False)
print("Saved inference stats to inference/inference_results.csv")
print(df.head())
print(df)
