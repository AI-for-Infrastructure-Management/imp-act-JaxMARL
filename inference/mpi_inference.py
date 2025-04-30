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

from itertools import chain
from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

# get SLURM_ARRAY_TASK_ID
if "SLURM_ARRAY_TASK_ID" in os.environ:
    TASK_ARRAY_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])

MAP_NAME = "Cologne-v1"
ALGORITHMS = ["vdn_rnn", "qmix_rnn", "pqn_rnn", "mappo_rnn", "ippo_rnn"]
NUM_CHECKPOINTS = 100
VMAPPED_SEED = 0
EVAL_SEED = 0
REWARD_SCALE = 1e6
BASE_PATH = "/home/pbhustali/prateek/imp-act-JaxMARL"
MODELS_PATH = f"/scratch/pbhustali/impact-JaxMARL-models/outputs"
RESULTS_PATH = f"{BASE_PATH}/inference/{MAP_NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

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

if MAP_NAME == "ToyExample-v2":
    TEST_NUM_ENVS = 250
    TEST_NUM_STEPS = 50 * 40

elif MAP_NAME == "Cologne-v1":
    TEST_NUM_ENVS = 200
    TEST_NUM_STEPS = 50 * 50

assert TEST_NUM_ENVS * TEST_NUM_STEPS == 50 * 10_000, "Total timesteps should be 50 * 10_000"

if rank == 0:
    print(f"MAP_NAME: {MAP_NAME}")
    print(f"VMAPPED_SEED: {VMAPPED_SEED}")
    print(f"EVAL_SEED: {EVAL_SEED}")
    print(f"TEST_NUM_ENVS: {TEST_NUM_ENVS}")
    print(f"TEST_NUM_STEPS: {TEST_NUM_STEPS}")
    print(f"TOTAL TIMESTEPS: {TEST_NUM_ENVS * TEST_NUM_STEPS}")

time_main_0 = time.time()

#! ALGORITHMS
for alg in [ALGORITHMS[TASK_ARRAY_ID]-1]:

    all_eval_stats_process = []
    all_eval_stats = []

    chkpt_dirs_alg = all_checkpoint_dirs[MAP_NAME][alg]

    make_get_greedy_metrics = import_function_from_path(
        f"{BASE_PATH}/evaluation/{alg}_road_env.py",
        "make_get_greedy_metrics",
    )

    #! SEEDS
    for j, chkpt_dir_name in enumerate(chkpt_dirs_alg):

        train_config_path = (
            f"{MODELS_PATH}/{MAP_NAME}/{alg}/{chkpt_dir_name}/config.yaml"
        )
        train_config = yaml.safe_load(open(train_config_path, "r"))
        rngs = jax.random.split(
            jax.random.PRNGKey(train_config["SEED"]), train_config["NUM_SEEDS"]
        )
        checkpoint_path = f"{MODELS_PATH}/{MAP_NAME}/{alg}/{chkpt_dir_name}/checkpoints/{rngs[VMAPPED_SEED][0]}"

        env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])

        jit_get_greedy_metrics = jax.jit(
            make_get_greedy_metrics(train_config, TEST_NUM_ENVS, TEST_NUM_STEPS)
        )
        all_safetensor_names = list(sorted(os.listdir(checkpoint_path)))

        #! CHECKPOINTS (MPI)
        for k in range(rank, NUM_CHECKPOINTS, num_procs):
            # for k in range(rank, 50, num_procs):

            time0 = time.time()

            safetensor_name = all_safetensor_names[k]
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

            all_eval_stats_process.append(
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
                f" Algorithm: {alg:<9} | Seed ({j+1:>2}): {rngs[VMAPPED_SEED][0]:<10} | Checkpoint ({k+1:>3}): {checkpoint_id:<4} | time: {time.time()-time0:.2f} | mean: {mean:.2f} | 95% CI: ({_95_ci[0]:.2f}, {_95_ci[1]:.2f})"
            )

    # Gather alg inference data at the end of all checkpoints x seeds
    all_eval_stats_process_0 = comm.gather(all_eval_stats_process, root=0)
    if rank == 0:

        all_eval_stats.extend(list(chain(*all_eval_stats_process_0)))

        df = pd.DataFrame(all_eval_stats)
        df.to_csv(
            f"{BASE_PATH}/inference/{MAP_NAME}/inference_results_{alg}.csv", index=False
        )
        print(f"Saved inference stats to inference_results_{alg}.csv")

        print(f"Total time: {time.time()-time_main_0:.2f}")
