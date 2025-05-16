# Introduction

This guide is for reproducing the results in the paper "IMP-act: Benchmarking MARL for Infrastructure Management Planning at Scale with JAX"

Follow the main setup instructions in the [README](README.md) to install the required packages and set up the environment.

# Training
## Repeating single runs
To repeat a single evaluation run with the same hyperparameters and seed as used in the paper, use the following commands:

```bash 
python experiments/{ALG}_road_env.py --config-path experiments/final_runs/{ENV}/ SEED={SEED}
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
python experiments/vdn_rnn_road_env.py --config-path experiments/final_runs/toy_example_v2/ SEED=2849413441
python experiments/vdn_rnn_road_env.py --config-path experiments/final_runs/cologne_v1/ SEED=1221700768
python experiments/vdn_rnn_road_env.py --config-path experiments/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# QMIX algorithm
python experiments/qmix_rnn_road_env.py --config-path experiments/final_runs/toy_example_v2/ SEED=2849413441
python experiments/qmix_rnn_road_env.py --config-path experiments/final_runs/cologne_v1/ SEED=1221700768
python experiments/qmix_rnn_road_env.py --config-path experiments/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# PQN algorithm
python experiments/pqn_rnn_road_env.py --config-path experiments/final_runs/toy_example_v2/ SEED=2849413441
python experiments/pqn_rnn_road_env.py --config-path experiments/final_runs/cologne_v1/ SEED=1221700768
python experiments/pqn_rnn_road_env.py --config-path experiments/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# MAPPO algorithm
python experiments/mappo_rnn_road_env.py --config-path experiments/final_runs/toy_example_v2/ SEED=2849413441
python experiments/mappo_rnn_road_env.py --config-path experiments/final_runs/cologne_v1/ SEED=1221700768
python experiments/mappo_rnn_road_env.py --config-path experiments/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

# IPPO algorithm
python experiments/ippo_rnn_road_env.py --config-path experiments/final_runs/toy_example_v2/ SEED=2849413441
python experiments/ippo_rnn_road_env.py --config-path experiments/final_runs/cologne_v1/ SEED=1221700768
python experiments/ippo_rnn_road_env.py --config-path experiments/final_runs/cologne_bonn_dusseldorf_v1/ SEED=2411725836

```

</details>

## Hyperparameter tuning runs
To run the hyperparameter tuning yourself as described in the paper, use the following commands:

```bash
wandb sweep experiments/hyperparameter_tuning/{ENV}/{ALG}_sweep.yaml
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
wandb sweep experiments/final_runs/{ENV}/{ALG}_sweep.yaml
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
The data is available on Hugging Face at [this link](https://huggingface.co/datasets/AI-for-Infrastructure-Management/imp-act-benchmark-results/tree/main).

You can use this script to download all data available in the repository:
```bash
./data/download.sh
```

Then, to evaluate checkpoints, use the commands in the following sections.

## Evaluating checkpoints by algorithm
To evaluate a folder containing checkpoints of a single algorithm for one environment, use the following command:

```bash
python evaluation/evaluate_runs.py --config-path "config/final_run_evaluations/{ENV}/" --config-name "{ALG}"
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
python evaluation/evaluate_runs.py EVALUATION_PATH="{EVALUATION_PATH}" TEST_NUM_ENVS={TEST_NUM_ENVS}
```
Where
- `{EVALUATION_PATH}` is the path to the folder containing the checkpoints you want to evaluate.

Depending on the available GPU VRAM you can set the amount of parallel environments TEST_NUM_ENVS to reduce the memory usage. The default value is 10000, so all evaluation episodes are run in parallel.


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
python experiments/heuristic_evaluation_road_env.py --config-path "config/heuristics/best_parameters/" --config-name "{ENV}_heuristic"
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