import os
import numpy as np
import logging
import yaml
import hydra
from omegaconf import OmegaConf
import time 
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import bootstrap
from pathlib import Path

import jaxmarl
from jaxmarl.wrappers.baselines import load_params

import qmix_rnn_road_env
import vdn_rnn_road_env
import mappo_rnn_road_env
import ippo_rnn_road_env
import pqn_rnn_road_env

from utils.wandb_utils import download_run, store_run_data, get_run_from_link
from utils.run_data_handling import Run

log = logging.getLogger(__name__)


def get_greedy_metric_fn(algorithm):
    if algorithm == "qmix_rnn":
        return qmix_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "vdn_rnn":
        return vdn_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "mappo_rnn":
        return mappo_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "ippo_rnn":
        return ippo_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "pqn_rnn":
        return pqn_rnn_road_env.make_get_greedy_metrics
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


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
def evaluate_checkpoints(config):
    # read YAML file with all checkpoints dir names
    config['BASE_PATH'] = config.get('BASE_PATH') or os.getcwd()

    config['TEST_NUM_STEPS'] = 50 * int(np.ceil(config['TEST_NUM_EPISODES'] / config['TEST_NUM_ENVS']))

    base_path = Path(config['BASE_PATH'])

    with open(f"{config['BASE_PATH']}/inference/all_checkpoint_dirs.yaml", "r") as f:
        all_checkpoint_dirs = yaml.safe_load(f)
    
    time_main_0 = time.time()

    #! ALGORITHMS
    for alg in config['ALGORITHMS']:
        log.info(f"Evaluating algorithm: {alg}")
        all_eval_stats = []
        
        chkpt_dirs_alg = all_checkpoint_dirs[config['MAP_NAME']][alg]

        make_get_greedy_metrics = get_greedy_metric_fn(alg)

        #! SEEDS
        for j, chkpt_dir_name in enumerate(chkpt_dirs_alg):
            train_config_path = (
                f"{config['BASE_PATH']}/outputs/{config['MAP_NAME']}/{alg}/{chkpt_dir_name}/config.yaml"
            )
            train_config = yaml.safe_load(open(train_config_path, "r"))
            rngs = jax.random.split(
                jax.random.PRNGKey(train_config["SEED"]), train_config["NUM_SEEDS"]
            )
            checkpoint_path = f"{config['BASE_PATH']}/outputs/{config['MAP_NAME']}/{alg}/{chkpt_dir_name}/checkpoints/{rngs[config['VMAPPED_SEED']][0]}"

            env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])

            jit_get_greedy_metrics = jax.jit(
                make_get_greedy_metrics(train_config, config['TEST_NUM_ENVS'], config['TEST_NUM_STEPS'])
            )
            
            all_safetensor_names = list(sorted(os.listdir(checkpoint_path)))

            top_k = config.get('TOP_K_CHECKPOINTS') or len(all_safetensor_names)
            evaluate_checkpoints = all_safetensor_names
            if top_k < len(all_safetensor_names):
                
                entity, project, run_id = get_run_from_link(train_config["WANDB_RUN_URL"])

                run_store_path = base_path / "evaluation/data" / train_config["ENV_KWARGS"]["map_name"] / alg / run_id

                if not run_store_path.exists():
                    log.info(f"Downloading run data for {run_id}...")
                    run_store_path.mkdir(parents=True, exist_ok=True)
                    run_str = f"{entity}/{project}/{run_id}"
                    run_data = download_run(run_str)
                    store_run_data(run_data, run_store_path.parent)
                else:
                    log.info(f"Run data for {run_id} already exists, skipping download.")

                run = Run(run_store_path)

                all_checkpoint_ids = [int(safetensor_name.split(".")[0].split("_")[-1])for safetensor_name in all_safetensor_names]
                checkpoint_returns = run.history['test_returned_episode_returns'][all_checkpoint_ids]

                top_k_checkpoints = np.argsort(checkpoint_returns)[-top_k:]
                evaluate_checkpoints = [all_safetensor_names[i] for i in top_k_checkpoints]

            #! CHECKPOINTS
            for k, safetensor_name in enumerate(evaluate_checkpoints):
                time0 = time.time()

                checkpoint_id = safetensor_name.split(".")[0].split("_")[-1]
                safetensor_path = os.path.join(checkpoint_path, safetensor_name)
                loaded_params = load_params(safetensor_path)

                rng = jax.random.PRNGKey(config['EVAL_SEED'])
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

                mean = inference_stats["mean"] / config['REWARD_SCALE']
                _95_ci = (
                    inference_stats["lower_ci"] / config['REWARD_SCALE'],
                    inference_stats["upper_ci"] / config['REWARD_SCALE'],
                )

                all_eval_stats.append(
                    {
                        "map_name": config['MAP_NAME'],
                        "algorithm": alg,
                        "checkpoint_dir_name": chkpt_dir_name,
                        "WANDB_RUN_ID": train_config["WANDB_RUN_ID"],
                        "checkpoint_id": checkpoint_id,
                        "VMAPPED_SEED": config['VMAPPED_SEED'],
                        "eval_seed": config['EVAL_SEED'],
                        "mean": float(inference_stats["mean"]),
                        "std_err": float(inference_stats["std_err"]),
                        "lower_ci": float(inference_stats["lower_ci"]),
                        "upper_ci": float(inference_stats["upper_ci"]),
                        "bst_std_err": float(inference_stats["bst_std_err"]),
                    }
                )

                log.info(
                    f" Algorithm: {alg:<9} | Seed ({j+1:>2}): {rngs[config['VMAPPED_SEED']][0]:<10} | Checkpoint ({k+1:>3}): {checkpoint_id:<4} | time: {time.time()-time0:.2f} | mean: {mean:.2f} | 95% CI: ({_95_ci[0]:.2f}, {_95_ci[1]:.2f})"
                )

        # Save algorithm-specific results
        results_dir = f"{config['BASE_PATH']}/inference/{config['MAP_NAME']}"
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(all_eval_stats)
        df.to_csv(f"{results_dir}/inference_results_{alg}.csv", index=False)
        log.info(f"Saved inference stats to {results_dir}/inference_results_{alg}.csv")

    # Combine all algorithm results into one CSV
    all_results_files = [f"{results_dir}/inference_results_{alg}.csv" for alg in config['ALGORITHMS']]
    all_results = pd.concat([pd.read_csv(f) for f in all_results_files if os.path.exists(f)])
    all_results.to_csv(f"{config['BASE_PATH']}/inference/inference_results.csv", index=False)
    log.info(f"Saved combined inference stats to {config['BASE_PATH']}/inference/inference_results.csv")
    
    log.info(f"Total time: {time.time()-time_main_0:.2f}")
    log.info(all_results.head())
    
    return all_results


@hydra.main(version_base=None, config_path="./config", config_name="evaluate_runs")
def main(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    print(f"Configuration:\n{yaml.dump(config_dict, default_flow_style=False)}")
    return evaluate_checkpoints(config_dict)


if __name__ == "__main__":
    main()