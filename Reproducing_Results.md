# Reproducing the results

This guide describes how to reproduce the results of the paper "IMP-act: Benchmarking MARL for Infrastructure Management Planning at Scale with JAX". It covers the training of the MARL algorithms, the hyperparameter tuning, the evaluation of the released checkpoints, the heuristic baselines, and the figures and tables of the paper. Before starting, follow the setup instructions in the [README](README.md) to install the required packages.

# Training

## Final runs

All final runs of an algorithm use a single configuration file in `experiments/config/final_runs/`, which contains the hyperparameters selected in the hyperparameter tuning. We train each algorithm with 10 seeds (200–209) in each of the 8 environment configurations, and select the environment configuration through `ENV_KWARGS.map_name`:

```bash
python experiments/{SCRIPT} --config-path config/final_runs --config-name {CONFIG} \
    ENV_KWARGS.map_name={MAP} SEED={SEED}
```

where `{SCRIPT}` and `{CONFIG}` depend on the algorithm,

| Algorithm | `{SCRIPT}` | `{CONFIG}` |
|---|---|---|
| VDN | `vdn_rnn_road_env.py` | `vdn_rnn_road_env` |
| QMIX | `qmix_rnn_road_env.py` | `qmix_rnn_road_env` |
| PQN-VDN | `pqn_vdn_rnn_road_env.py` | `pqn_vdn_rnn_road_env` |
| MAPPO | `mappo_rnn_road_env.py` | `mappo_rnn_road_env` |
| IPPO | `ippo_rnn_road_env.py` | `ippo_rnn_road_env` |

`{MAP}` is one of the 8 environment configurations,

| Map | Constrained (global budget) | Unconstrained |
|---|---|---|
| ToyExample | `ToyExample-v2` (budget 6.5M) | `ToyExample-v2-unconstrained` |
| Cologne | `Cologne-v1-moderate-budget` (budget 300M), `Cologne-v1` (budget 200M), `Cologne-v1-critical-budget` (budget 150M) | `Cologne-v1-unconstrained` |
| CologneBonnDusseldorf | `CologneBonnDusseldorf-v1` (budget 800M) | `CologneBonnDusseldorf-v1-unconstrained` |

and `{SEED}` is one of the seeds 200–209. For example, the first seed of VDN in the unconstrained Cologne environment is trained with

```bash
python experiments/vdn_rnn_road_env.py --config-path config/final_runs --config-name vdn_rnn_road_env \
    ENV_KWARGS.map_name=Cologne-v1-unconstrained SEED=200
```

Each run trains for 50M environment steps, saves a checkpoint after every 1% of the training updates, and logs to Weights & Biases (W&B). To train without W&B, add `WANDB_MODE=disabled`. VDN<sub>BA</sub> is not trained separately, since it evaluates the VDN checkpoints with budget-aware action selection (see [Evaluation](#evaluation)).

## Hyperparameter tuning

We tune the hyperparameters of each algorithm on `Cologne-v1` with a random search over 50 hyperparameter combinations, each trained with 3 seeds. The search space of each algorithm is defined in the W&B sweep files in `experiments/config/hyper_parameter_tuning/cologne_v1/`, and a sweep is started with

```bash
wandb sweep experiments/config/hyper_parameter_tuning/cologne_v1/{ALG}_sweep.yaml
wandb agent {SWEEP_ID}
```

where `{ALG}` is one of `vdn`, `qmix`, `pqn`, `mappo` and `ippo`, and `{SWEEP_ID}` is the ID printed by `wandb sweep`. Since the search is random, a new sweep samples different combinations than the ones in the paper. The combinations of the paper, together with their learning curves, are released in `tuning_results.parquet` and analysed in [`results/nb_hyperparameter_tuning.ipynb`](results/nb_hyperparameter_tuning.ipynb).

# Evaluation

## Released checkpoints

For each algorithm and environment configuration, we release the best checkpoint of each of the 10 training seeds on [Zenodo](https://zenodo.org/records/24175731), released under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). The checkpoints are grouped by the rank of their seed, such that `rank01.tar.gz` holds the best seed of every algorithm and environment configuration. Each archive can be downloaded on its own,

```bash
wget "https://zenodo.org/records/24175731/files/rank01.tar.gz?download=1" -O rank01.tar.gz
tar -xzf rank01.tar.gz
```

or, alternatively, `best_checkpoint` in `results/common.py` downloads the archive and extracts the requested checkpoint to `results/data/checkpoints/`:

```bash
cd results
python -c "from common import best_checkpoint; print(best_checkpoint('Cologne-v1', 'vdn_rnn', rank=1))"
```

A checkpoint is then evaluated with the evaluation script of its algorithm,

```bash
python evaluation/{ALG}_road_env.py CHECKPOINT_PATH={CHECKPOINT_PATH} STEP={STEP} TEST_NUM_ENVS=1000
```

where `{ALG}` is one of `vdn_rnn`, `vdn_ba_rnn`, `qmix_rnn`, `pqn_rnn`, `mappo_rnn` and `ippo_rnn`, `{CHECKPOINT_PATH}` is the extracted checkpoint folder, and `{STEP}` is the step in its name. For instance, the best VDN checkpoint on `Cologne-v1` is evaluated with

```bash
python evaluation/vdn_rnn_road_env.py \
    CHECKPOINT_PATH=results/data/checkpoints/Cologne-v1__vdn_rnn__rank01__seed200__step21662 \
    STEP=21662 TEST_NUM_ENVS=1000
```

The script prints the mean return over `TEST_NUM_ENVS` episodes and saves it to `{CHECKPOINT_PATH}/eval/{TIMESTAMP}/metrics.yaml`. Similarly, `vdn_ba_rnn` evaluates a VDN checkpoint with budget-aware action selection.

## All checkpoints of a training run

To evaluate every checkpoint of your own training runs, as we did to obtain `results.csv`, we provide `evaluation/evaluate_runs.py` together with one evaluation config per map and algorithm in `evaluation/config/final_run_evaluations/`:

```bash
python evaluation/evaluate_runs.py --config-path config/final_run_evaluations/{ENV} --config-name {ALG} \
    EVALUATION_PATH={EVALUATION_PATH}
```

where `{ENV}` is one of `toy_example_v2`, `cologne_v1` and `cologne_bonn_dusseldorf_v1`, `{ALG}` is one of `vdn_rnn`, `vdn_ba_rnn`, `qmix_rnn`, `pqn_rnn`, `mappo_rnn` and `ippo_rnn`, and `{EVALUATION_PATH}` is the folder with the training runs. By default, all 10,000 evaluation episodes run in parallel, and lowering `TEST_NUM_ENVS` reduces the GPU memory usage.

# Heuristics

## Evaluating the heuristics

We compare the MARL algorithms with four heuristic policies, and evaluate each of them in every environment configuration with

```bash
python evaluation/heuristic_evaluation_road_env.py --config-path config/heuristics --config-name {CONFIG}_{HEURISTIC}
```

where `{CONFIG}` is one of `toy_example_v2`, `toy_example_v2_unconstrained`, `cologne_v1`, `cologne_v1_moderate_budget`, `cologne_v1_critical_budget`, `cologne_v1_unconstrained`, `cologne_bonn_dusseldorf_v1` and `cologne_bonn_dusseldorf_v1_unconstrained`, and `{HEURISTIC}` is one of

- `heuristic`: the prioritized segment-based heuristic H<sub>PS</sub>,
- `do_nothing`: always routine inspections,
- `random`: random actions, and
- `always_repair`: always minor repair.

## Optimizing the heuristics

The parameters of H<sub>PS</sub> are optimized separately for each environment configuration with

```bash
python experiments/heuristics_optimize_road_env.py --config-path config/heuristics/optimization --config-name {CONFIG}_heuristic
```

where `{CONFIG}` is one of the 8 environment configurations listed above.

# Figures and tables

The figures and tables of the paper can be reproduced with the code in [`results/`](results/README.md) using the released data, which is downloaded automatically when missing:

```bash
python results/create_figures.py
python results/create_tables.py
```

See [`results/README.md`](results/README.md) for the scripts, notebooks and released data files.
