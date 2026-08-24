# Introduction

This guide is for reproducing the results in the paper "IMP-act: Benchmarking MARL for Infrastructure Management Planning at Scale with JAX"

Follow the main setup instructions in the [README](README.md) to install the required packages and set up the environment.

# Training
## Repeating single runs
To repeat a single evaluation run with the same hyperparameters and seed as used in the paper, use the following commands:

```bash 
python experiments/{ALG}_road_env.py --config-path config/final_runs/{ENV}/ SEED={SEED}
```

Where
- `{ALG}` is the algorithm you want to run
    - `vdn_rnn`
    - `qmix_rnn`
    - `pqn_rnn`
    - `mappo_rnn`
    - `ippo_rnn`
- `{ENV}` is the environment you want to run the evaluation for. The options are:
    - `toy_example_v2`
    - `cologne_v1`
    - `cologne_bonn_dusseldorf_v1`
- and `{SEED}` is the seed you want to use. The seeds used in the paper are:

| Seed # | ToyExample-v2 | Cologne-v1 | CologneBonnDusseldorf-v1 |
|--------|:-------------:|:----------:|:------------------------:|
| 1      | 2849413441    | 1221700768 | 2411725836               |
| 2      | 1696433054    | 3410157204 | 1769144590               |
| 3      | 3346946419    | 1116695186 | 1729133645               |
| 4      | 3076387228    | 2006757533 | 1039433210               |
| 5      | 1975447076    | 1989696838 | 1506812145               |
| 6      | 2982199793    | 3979383719 | 2519822500               |
| 7      | 1363760044    | 1191826828 | 2760380712               |
| 8      | 3057989503    | 1362152812 | 1670876600               |
| 9      | 3530972771    | 1950933211 | 4061526404               |
| 10     | 4165291849    | 3846780909 | 1033305736               |

Here are some examples of how to run the algorithms with the same hyperparameters and seeds as used in the paper:

<details>
<summary> Show Commands </summary>

```bash
# VDN algorithm
python experiments/vdn_rnn_road_env.py --config-path config/final_runs/toy_example_v2/ SEED=2849413441
python experiments/vdn_rnn_road_env.py --config-path config/final_runs/cologne_v1/ SEED=1221700768
python experiments/vdn_rnn_road_env.py --config-path config/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# QMIX algorithm
python experiments/qmix_rnn_road_env.py --config-path config/final_runs/toy_example_v2/ SEED=2849413441
python experiments/qmix_rnn_road_env.py --config-path config/final_runs/cologne_v1/ SEED=1221700768
python experiments/qmix_rnn_road_env.py --config-path config/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# PQN algorithm
python experiments/pqn_rnn_road_env.py --config-path config/final_runs/toy_example_v2/ SEED=2849413441
python experiments/pqn_rnn_road_env.py --config-path config/final_runs/cologne_v1/ SEED=1221700768
python experiments/pqn_rnn_road_env.py --config-path config/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# MAPPO algorithm
python experiments/mappo_rnn_road_env.py --config-path config/final_runs/toy_example_v2/ SEED=2849413441
python experiments/mappo_rnn_road_env.py --config-path config/final_runs/cologne_v1/ SEED=1221700768
python experiments/mappo_rnn_road_env.py --config-path config/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# IPPO algorithm
python experiments/ippo_rnn_road_env.py --config-path config/final_runs/toy_example_v2/ SEED=2849413441
python experiments/ippo_rnn_road_env.py --config-path config/final_runs/cologne_v1/ SEED=1221700768
python experiments/ippo_rnn_road_env.py --config-path config/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

```

</details>

## Hyperparameter tuning runs
To run the hyperparameter tuning yourself as described in the paper, use the following commands:

```bash
wandb sweep experiments/config/hyper_parameter_tuning/{ENV}/{ALG}_sweep.yaml
wandb agent {SWEEP_ID}
```

Where 
- `{ENV}` is the environment you want to run the hyperparameter tuning for. The options are:
    - `toy_example_v2`
    - `cologne_v1`
    - `cologne_bonn_dusseldorf_v1`
- `{ALG}` is the algorithm you want to run 
    - `vdn`
    - `qmix`
    - `pqn`
    - `mappo`
    - `ippo`
- and `{SWEEP_ID}` is the ID of the sweep which is created when you run the first command.


## Seeded Training Runs
To run the evaluation runs, use the following command:

```bash
wandb sweep experiments/config/final_runs/{ENV}/{ALG}_sweep.yaml
wandb agent {SWEEP_ID}
```

Where
- `{ENV}` is the environment you want to run the evaluation for. The options are:
    - `toy_example_v2`
    - `cologne_v1`
    - `cologne_bonn_dusseldorf_v1`
- `{ALG}` is the algorithm you want to run
    - `vdn`
    - `qmix`
    - `pqn`
    - `mappo`
    - `ippo`
- and `{SWEEP_ID}` is the ID of the sweep which is created when you run the first command.

# Evaluation of checkpoints
First, you need to download the pretrained models used for the evaluation. 

All data is made available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The data is available on Hugging Face at [this link](https://huggingface.co/datasets/AI-for-Infrastructure-Management/imp-act-benchmark-results/tree/main). (DOI: 10.57967/hf/5523)

You can use this script to download all data available in the repository:
```bash
./data/download.sh
```

Then, to evaluate checkpoints, use the commands in the following sections.

### How evaluation is organised

Evaluation has **one artifact and one consumer**:

```
training (STORE_EVAL_RETURNS) ─┐
                               ├─→ eval_returns_<step>.csv ─→ compute_eval_statistics.py ─→ inference_results.csv
generate_eval_returns.py ──────┘      raw episode returns          statistics + splits
```

`generate_eval_returns.py` re-runs greedy inference and **writes returns**; it computes
no statistics. `compute_eval_statistics.py` is the only thing that bootstraps or emits
a results row, whichever producer the returns came from. So there is exactly one
implementation of the table, and the expensive step (inference, GPU) can be re-run
independently of the cheap one (~1.6 s of bootstrap per checkpoint, CPU).

| Section | Reads | Produces | Requires |
|---|---|---|---|
| [By algorithm](#evaluating-checkpoints-by-algorithm) / [multiple checkpoints](#evaluating-multiple-checkpoints) | saved checkpoints | raw returns under `<run>/eval_returns/`, then the table | JAX, GPU |
| [Computing the results table](#computing-the-results-table) | raw returns from either producer | `inference_results.csv` + per-map/algorithm splits | numpy, scipy, pyyaml |
| [Single checkpoints](#evaluating-single-checkpoints) | one checkpoint | metrics logged to the console | JAX, GPU |

If a run was trained with `STORE_EVAL_RETURNS: True` you need no GPU at all — the
evaluation already happened during training, so go straight to the second section. Use the
first for runs without stored returns, or to evaluate a **variant** such as budget-aware
VDN, which re-runs inference with a different action rule over the same checkpoints.

## Evaluating checkpoints by algorithm
To evaluate a folder containing checkpoints of a single algorithm for one environment, use the following command:

```bash
python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/{ENV}/" --config-name "{ALG}"
```

Where `{ENV}` is the environment you want to run the evaluation for. The options are:
- `toy_example_v2`
- `cologne_v1`
- `cologne_bonn_dusseldorf_v1`

And `{ALG}` is the algorithm you want to run Also budget aware $\text{VDN}_\text{BA}$ is available.
- `vdn_rnn`
- `qmix_rnn`
- `pqn_rnn`
- `mappo_rnn`
- `ippo_rnn`
* `vdn_ba_rnn`


## Evaluating multiple checkpoints
To evaluate a folder possibly containing multiple algorithms and checkpoints, use the following command:

```bash
python evaluation/generate_eval_returns.py EVALUATION_PATH="{EVALUATION_PATH}" TEST_NUM_ENVS={TEST_NUM_ENVS}
```
Where
- `{EVALUATION_PATH}` is the path to the folder containing the checkpoints you want to evaluate.

Depending on the available GPU VRAM you can set the amount of parallel environments TEST_NUM_ENVS to reduce the memory usage. The default value is 10000, so all evaluation episodes are run in parallel.

### Where the returns go, and re-running

Both commands above write raw returns to `<run>/eval_returns/<algorithm>/seed<EVAL_SEED>/`,
alongside an `eval_meta.yaml` recording what produced them. Keying on the algorithm variant
and the eval seed means a budget-aware pass cannot collide with a plain one, nor with the
returns training wrote for itself.

A directory that already holds a complete set is **skipped**, so an interrupted run can
simply be repeated; `FORCE=true` re-evaluates anyway.

The results table is then built by [`compute_eval_statistics.py`](#computing-the-results-table),
which runs automatically at the end. On a cluster, set `WRITE_RESULTS_TABLE=false` in the
array task and run one table pass over the whole tree afterwards — otherwise every task
rebuilds the same table.


## Computing the results table
This is what turns raw returns into the results table, whether they were written during
training (`STORE_EVAL_RETURNS: True`) or by `generate_eval_returns.py`. No inference is
re-run and no GPU is needed:

```bash
python evaluation/compute_eval_statistics.py
python evaluation/compute_eval_statistics.py {RUN_ROOT} --workers 32
```

With no arguments it reads `outputs/`, where hydra writes runs, and writes to
`evaluation/results/`. `{RUN_ROOT}` may be any directory above the runs — the run
directories are found by locating the eval returns and walking up to the nearest
`config.yaml`, so the layout does not matter.

Useful options:
- `--workers {N}` — parallel processes. The bootstrap dominates, at roughly 1.6 s and
  0.4 GB per checkpoint for 10 000-episode evaluations.
- `--resume` — skip checkpoints already present in an existing `inference_results.csv`.
- `--dry-run` — report what was found without computing anything.

Provenance comes from an `eval_meta.yaml` sidecar where one exists (returns produced by
`generate_eval_returns.py`), and otherwise from the run's own `config.yaml`. Returns
written during training have no fixed eval seed — their draws come from the training RNG —
so `eval_seed` is empty on those rows and set on post-hoc ones. That does not affect a
per-checkpoint mean or confidence interval, but comparisons relying on the same eval draws
across runs need a post-hoc pass at a fixed `EVAL_SEED`.

A run can hold several evaluations at once — the one training wrote, plus one per
(algorithm variant, eval seed) — and they come out as separate rows, distinguished by the
`algorithm` and `eval_seed` columns.


## Evaluating single checkpoints
```bash
python evaluation/{ALG}_road_env.py CHECKPOINT_PATH={CHECKPOINT_PATH} STEP={STEP}
```

Where
- `{ALG}` is the algorithm you want to evaluate
    - `vdn_rnn`
    - `qmix_rnn`
    - `pqn_rnn`
    - `mappo_rnn`
    - `ippo_rnn`
- `{CHECKPOINT_PATH}` is the path to the checkpoint you want to evaluate.

# Heuristics
## Evaluating heuristics
To evaluate the baseline heuristics used in the paper, use the following command:

```bash
python evaluation/heuristic_evaluation_road_env.py --config-path "evaluation/config/heuristics/" --config-name "{ENV}_heuristic"
``` 

Where 
- `{ENV}` is the environment you want to run the evaluation for. The options are:
    - `toy_example_v2`
    - `cologne_v1`
    - `cologne_bonn_dusseldorf_v1` 

## Optimizing heuristics
To optimize the heuristics used in the paper, use the following command:

```bash
python experiments/heuristics_optimize_road_env.py --config-path "config/heuristics/optimization/" --config-name "{ENV}_heuristic"
```

Where 
- `{ENV}` is the environment you want to run the evaluation for. The options are:
    - `toy_example_v2`
    - `cologne_v1`
    - `cologne_bonn_dusseldorf_v1` 


# Plots

The code to generate Figure 3 based on the results of the training are located in `evaluation/plotting/figure_3.ipynb` with instructions on how to download the data and create the plot.
