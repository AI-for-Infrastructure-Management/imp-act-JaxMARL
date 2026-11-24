"""Re-run greedy inference over saved checkpoints and write raw eval returns.

The producer half of evaluation (JAX, GPU): every run directory found under
EVALUATION_PATH has each of its checkpoints evaluated, and the per-episode returns
written to `<run>/eval_returns/<algorithm>/seed<EVAL_SEED>/eval_returns_<step>.csv`
— the same artifact training writes under `STORE_EVAL_RETURNS` — plus an
`eval_meta.yaml` sidecar recording what produced them. No statistics are computed
here: `compute_eval_statistics.py` builds the results table from the returns, and
runs automatically at the end unless WRITE_RESULTS_TABLE is false.

    python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/{ENV}/" --config-name "{ALG}"
    python evaluation/generate_eval_returns.py EVALUATION_PATH="data/models/Cologne-v1" TEST_NUM_ENVS=1000
"""

import logging
import os
import time
from pathlib import Path

import hydra
import ippo_rnn_road_env
import jax
import mappo_rnn_road_env
import numpy as np
import pqn_rnn_road_env
import qmix_rnn_road_env
import vdn_ba_rnn_road_env
import vdn_rnn_road_env
import yaml
from compute_eval_statistics import DEFAULT_OUTPUT as DEFAULT_RESULTS_SUBDIR
from compute_eval_statistics import EPISODE_HORIZON
from compute_eval_statistics import main as compute_statistics_main
from eval_returns_format import (
    eval_returns_complete,
    eval_returns_dir,
    resolve_algorithm,
    write_eval_meta,
    write_eval_returns,
)
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import load_params

log = logging.getLogger(__name__)


def get_greedy_metric_fn(algorithm):
    if algorithm == "qmix_rnn":
        return qmix_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "vdn_rnn":
        return vdn_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "vdn_ba_rnn":
        return vdn_ba_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "mappo_rnn":
        return mappo_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "ippo_rnn":
        return ippo_rnn_road_env.make_get_greedy_metrics
    elif algorithm == "pqn_rnn":
        return pqn_rnn_road_env.make_get_greedy_metrics
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


############################ EVALUATION ################################


def evaluate_checkpoint(path, config):
    """Write raw per-episode eval returns for every checkpoint of one run.

    Produces the same artifact training does under `STORE_EVAL_RETURNS`, so that
    `compute_eval_statistics.py` is the only thing that computes a statistic.
    Returns the directory written, or None if it was already complete.
    """
    train_config = yaml.safe_load(open(path / "config.yaml", "r"))

    rngs = jax.random.split(
        jax.random.PRNGKey(train_config["SEED"]), train_config["NUM_SEEDS"]
    )
    vmapped_seed_key = str(rngs[config["VMAPPED_SEED"]][0])
    checkpoint_path = path / "checkpoints" / vmapped_seed_key

    algorithm = resolve_algorithm(train_config["ALG_NAME"], config["EVAL_VDN_BA"])
    out_dir = eval_returns_dir(path, algorithm, config["EVAL_SEED"])

    # Filter on the extension rather than taking the directory listing whole: a
    # real run's checkpoint directory can hold entries that are not checkpoints, and
    # handing one to load_params dies with "header too large" only after every real
    # checkpoint has already been evaluated. It also miscounts the completeness check.
    checkpoint_files = sorted(
        name for name in os.listdir(checkpoint_path) if name.endswith(".safetensors")
    )
    if not checkpoint_files:
        # Bail before anything is written: an empty directory that got a sidecar
        # would count as complete (0 == 0) and be skipped on every later run.
        log.warning(f"{checkpoint_path} holds no .safetensors checkpoints, skipping")
        return None

    # The artifact is its own cache: a complete directory is skippable and a partial
    # one is visibly partial.
    if not config["FORCE"] and eval_returns_complete(out_dir, len(checkpoint_files)):
        log.info(
            f"{out_dir} already complete ({len(checkpoint_files)} evals), skipping"
        )
        return None

    env = jaxmarl.make(train_config["ENV_NAME"], **train_config["ENV_KWARGS"])

    # resolve_algorithm already folded EVAL_VDN_BA into the name, so the variant's
    # own module is picked up here.
    make_get_greedy_metrics = get_greedy_metric_fn(algorithm)

    jit_get_greedy_metrics = jax.jit(
        make_get_greedy_metrics(
            train_config, config["TEST_NUM_ENVS"], config["TEST_NUM_STEPS"]
        )
    )

    log.info(f"Evaluating {len(checkpoint_files)} checkpoints -> {out_dir}")
    num_episodes = None
    for k, safetensor_name in enumerate(checkpoint_files):
        time0 = time.time()

        checkpoint_id = safetensor_name.split(".")[0].split("_")[-1]
        safetensor_path = os.path.join(checkpoint_path, safetensor_name)
        loaded_params = load_params(safetensor_path)

        rng = jax.random.PRNGKey(config["EVAL_SEED"])
        infos = jit_get_greedy_metrics(rng, loaded_params)

        # infos["returned_episode_returns"] | shape: (NUM_TIME_STEPS, NUM_ENVS, NUM_AGENTS)
        # This selects the episode returns for which returned_episode is True
        # and reshapes it to (NUM_EPISODES, NUM_AGENTS). This is required when NUM_ENVS
        # cannot fit in memory, and we must process them in batches. Since all agents
        # have same rewards, we only take the first agent's rewards.
        _episode_returns = infos["returned_episode_returns"][infos["returned_episode"]]
        episode_returns = _episode_returns.reshape(-1, env.num_agents)[:, 0]

        # float64 on the way out, so the file carries full precision and the
        # consumer is not handed a float32 accumulation.
        episode_returns = np.asarray(episode_returns, dtype=np.float64)
        num_episodes = len(episode_returns)

        # Same filename the trainer uses, keeping the checkpoint's own id and
        # zero-padding so the join back to checkpoint_<id>.safetensors is exact.
        write_eval_returns(
            out_dir / f"eval_returns_{checkpoint_id}.csv", episode_returns
        )

        log.info(
            f" Algorithm: {algorithm:<12} | Seed: {vmapped_seed_key:<10}"
            f" | Checkpoint ({k+1:>3}/{len(checkpoint_files)}): {checkpoint_id:<6}"
            f" | time: {time.time()-time0:.2f}s"
            f" | mean: {episode_returns.mean():.4e} over {num_episodes} episodes"
        )

    # Written last: its presence is what marks the directory complete.
    write_eval_meta(
        out_dir,
        algorithm=algorithm,
        eval_seed=config["EVAL_SEED"],
        vmapped_seed=config["VMAPPED_SEED"],
        vmapped_seed_key=vmapped_seed_key,
        map_name=train_config["ENV_KWARGS"]["map_name"],
        train_alg_name=train_config["ALG_NAME"],
        test_num_envs=config["TEST_NUM_ENVS"],
        test_num_steps=config["TEST_NUM_STEPS"],
        num_episodes=num_episodes,
        num_checkpoints=len(checkpoint_files),
    )
    return out_dir


def evaluate_checkpoints(config):
    config["BASE_PATH"] = config.get("BASE_PATH") or os.getcwd()
    base_path = Path(config["BASE_PATH"])

    # The per-algorithm configs under config/final_run_evaluations/<env>/ are
    # standalone — hydra loads one as the whole config with no merge against
    # generate_eval_returns.yaml — so they carry none of these keys. Default
    # them here rather than indexing them directly downstream.
    config.setdefault("EVAL_VDN_BA", False)
    config.setdefault("FORCE", False)
    config.setdefault("WRITE_RESULTS_TABLE", True)
    config.setdefault("RESULTS_PATH", None)

    # Every env yields one episode per EPISODE_HORIZON steps, so the episode count
    # must fill the env/step grid exactly or the eval would hold a different number
    # of episodes than requested.
    if config["TEST_NUM_EPISODES"] % config["TEST_NUM_ENVS"] != 0:
        raise ValueError(
            f"TEST_NUM_EPISODES ({config['TEST_NUM_EPISODES']}) must be a multiple "
            f"of TEST_NUM_ENVS ({config['TEST_NUM_ENVS']})"
        )
    config["TEST_NUM_STEPS"] = EPISODE_HORIZON * (
        config["TEST_NUM_EPISODES"] // config["TEST_NUM_ENVS"]
    )

    def find_run_directories_recursively(path):
        # detect if this is a run directory

        if (path / "checkpoints").exists():
            run_paths = [path]
        else:
            run_paths = []
            for subdir in path.iterdir():
                if subdir.is_dir():
                    run_paths.extend(find_run_directories_recursively(subdir))

        return run_paths

    evaluation_path = Path(config.get("EVALUATION_PATH") or base_path / "outputs")
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation path {evaluation_path} does not exist")

    log.info(f"Searching for run directories in {evaluation_path}")
    run_paths = find_run_directories_recursively(evaluation_path)
    log.info(f"Found {len(run_paths)} run directories")

    time_main_0 = time.time()

    written, skipped = [], 0
    for run_path in run_paths:
        out_dir = evaluate_checkpoint(run_path, config)
        if out_dir is None:
            skipped += 1
        else:
            written.append(out_dir)

    log.info(
        f"Wrote eval returns for {len(written)} runs ({skipped} skipped) "
        f"in {time.time()-time_main_0:.2f}s"
    )

    if not config["WRITE_RESULTS_TABLE"]:
        log.info(
            "WRITE_RESULTS_TABLE is false — run compute_eval_statistics.py "
            "over the tree to build the results table"
        )
        return written

    results_path = Path(config["RESULTS_PATH"] or base_path / DEFAULT_RESULTS_SUBDIR)
    log.info(f"Building the results table from {evaluation_path} -> {results_path}")
    compute_statistics_main([str(evaluation_path), "-o", str(results_path)])
    return written


@hydra.main(
    version_base=None, config_path="./config", config_name="generate_eval_returns"
)
def main(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    print(f"Configuration:\n{yaml.dump(config_dict, default_flow_style=False)}")
    return evaluate_checkpoints(config_dict)


if __name__ == "__main__":
    main()
