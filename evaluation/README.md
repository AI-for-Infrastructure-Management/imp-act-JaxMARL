# Evaluation and Statistics computation

Getting the results table has two steps: (1) produce raw per-episode returns and (2) use them to compute statistics.

```
(A) training (STORE_EVAL_RETURNS) ─┐
                                   ├─→ eval_returns_<step>.csv ─→ compute_eval_statistics.py
(B) generate_eval_returns.py ──────┘
                                         raw episode returns          the results table
└──── 1. evaluate checkpoints ─────┘                              └──── 2. statistics ─────┘
```

**Step 1 — evaluate checkpoints.** Two ways to do it:

- **A**: set `STORE_EVAL_RETURNS: True` before training and the returns computed during training are stored.
- **B**: run `generate_eval_returns.py` on saved checkpoints afterwards — for runs that didn't set the flag, or for variants like budget-aware VDN.

**Step 2 — compute statistics.** `compute_eval_statistics.py` takes returns from either method, computes statistics (like mean and CIs), and writes `inference_results.csv`

Run everything from the repo root.

# 1. Evaluate checkpoints

## Method A — during training (`STORE_EVAL_RETURNS`)

Set `STORE_EVAL_RETURNS: True` in the training config (already set in `experiments/config/final_runs/`). Each periodic greedy evaluation then dumps its raw returns during training, to

```
<run>/checkpoints/<vmapped_seed>/eval_returns_<step>.csv
```

Go straight to step 2.

## Method B — after training (`generate_eval_returns.py`)

Re-runs greedy inference over saved checkpoints. Use it for runs trained without `STORE_EVAL_RETURNS`, or to evaluate a **variant** such as budget-aware VDN, which applies a different action rule to the same checkpoints. Requires JAX and a GPU.

Each (environment, algorithm) pair has a ready-made config under `config/final_run_evaluations/`, so the usual invocation just names one:

```bash
python evaluation/generate_eval_returns.py \
  --config-path "config/final_run_evaluations/<env>/" --config-name "<algorithm>"
```

The config sets `EVALUATION_PATH` (and `TEST_NUM_ENVS`, and `EVAL_VDN_BA` where needed); everything else comes from `config/generate_eval_returns.yaml`. Any key can still be overridden on the command line by appending `KEY=value`:

| Override | Meaning |
|---|---|
| `EVALUATION_PATH` | **Where the checkpoints are read from.** The tree is walked recursively for run directories, so this can be a single algorithm's folder or a whole environment root. |
| `TEST_NUM_ENVS` | Episodes evaluated in parallel. Default `10000`, and `5000` in the CologneBonnDusseldorf-v1 configs; lower it to fit GPU memory. `TEST_NUM_EPISODES` must stay a multiple of it. |
| `EVAL_VDN_BA` | `true` re-evaluates `vdn_rnn` checkpoints with budget-constrained greedy action selection. Applies to every run found under the path. |
| `FORCE` | `true` re-evaluates directories that already hold a complete set of returns. |

### Where the output goes

`EVALUATION_PATH` is the input. Returns are written back **in place**, into each run directory found beneath it:

```
<run>/eval_returns/<algorithm>/seed<EVAL_SEED>/eval_returns_<step>.csv
```

alongside an `eval_meta.yaml` recording what produced them — a different location from Method A's, so the two never collide. Keying on the algorithm variant and the eval seed also means a budget-aware pass cannot collide with a plain one. A directory that already holds a complete set is skipped, so an interrupted run can simply be repeated.

Step 2 runs automatically at the end unless you pass `WRITE_RESULTS_TABLE=false`.

### One algorithm, one environment

| Env | Algorithm | Command |
|---|---|---|
| **ToyExample-v2** | VDN | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "vdn_rnn"` |
|  | VDN-BA | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "vdn_ba_rnn"` |
|  | QMIX | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "qmix_rnn"` |
|  | PQN | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "pqn_rnn"` |
|  | MAPPO | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "mappo_rnn"` |
|  | IPPO | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "ippo_rnn"` |
| **Cologne-v1** | VDN | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_v1/" --config-name "vdn_rnn"` |
|  | VDN-BA | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_v1/" --config-name "vdn_ba_rnn"` |
|  | QMIX | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_v1/" --config-name "qmix_rnn"` |
|  | PQN | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_v1/" --config-name "pqn_rnn"` |
|  | MAPPO | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_v1/" --config-name "mappo_rnn"` |
|  | IPPO | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_v1/" --config-name "ippo_rnn"` |
| **CologneBonnDusseldorf-v1** | VDN | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_bonn_dusseldorf_v1/" --config-name "vdn_rnn"` |
|  | VDN-BA | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_bonn_dusseldorf_v1/" --config-name "vdn_ba_rnn"` |
|  | QMIX | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_bonn_dusseldorf_v1/" --config-name "qmix_rnn"` |
|  | PQN | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_bonn_dusseldorf_v1/" --config-name "pqn_rnn"` |
|  | MAPPO | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_bonn_dusseldorf_v1/" --config-name "mappo_rnn"` |
|  | IPPO | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/cologne_bonn_dusseldorf_v1/" --config-name "ippo_rnn"` |

### A whole environment at once

The tree walk means an environment root covers all six algorithms in one pass. There is a config for this only for ToyExample-v2; for the other two, point `EVALUATION_PATH` at the environment root yourself.

| Env | Command |
|---|---|
| ToyExample-v2 | `python evaluation/generate_eval_returns.py --config-path "config/final_run_evaluations/toy_example_v2/" --config-name "generate_eval_returns"` |
| Cologne-v1 | `python evaluation/generate_eval_returns.py EVALUATION_PATH="data/models/Cologne-v1"` |
| CologneBonnDusseldorf-v1 | `python evaluation/generate_eval_returns.py EVALUATION_PATH="data/models/CologneBonnDusseldorf-v1" TEST_NUM_ENVS=5000` |
| Any folder | `python evaluation/generate_eval_returns.py EVALUATION_PATH="<path>" TEST_NUM_ENVS=<n>` |

Note that an environment root evaluates VDN plainly, not budget-aware; `EVAL_VDN_BA=true` applies to every run found under the path, so use the `vdn_ba_rnn` config instead.

# 2. Compute eval statistics

`compute_eval_statistics.py` computes statistics (like mean and CIs), whichever method (A or B) the returns came from. Needs only numpy, scipy, and pyyaml — no JAX, no GPU.

```bash
python evaluation/compute_eval_statistics.py <root> [-o <output-dir>]
```

`<root>` is anything above the run directories; it defaults to `outputs/`, where hydra puts them. Output defaults to `evaluation/results/` and holds `inference_results.csv` plus per-map/algorithm splits.

| Option | Meaning |
|---|---|
| `-o`, `--output-dir` | Destination for `inference_results.csv` and the splits (default `evaluation/results/`). |
| `--workers` | Worker processes (default `1`). |
| `--n-resamples` | Bootstrap resamples (default `1000`; scipy's own default is `9999`). |
| `--bootstrap-seed` | Seed for reproducible CI bounds; a negative value leaves it unseeded (default `0`). |
| `--resume` | Append to an existing `inference_results.csv`, skipping rows already in it. |
| `--dry-run` | Report what was discovered, compute nothing. |

# Notes

- **`vdn_ba_rnn.yaml` points at the `vdn_rnn` checkpoints on purpose.** Budget-aware VDN
  is not a separately trained algorithm — it re-evaluates the same `vdn_rnn` checkpoints
  with budget-constrained greedy action selection, which is why the config pairs that path
  with `EVAL_VDN_BA: True`. `resolve_algorithm()` renames it to `vdn_ba_rnn` in the
  returns tree and the results table, so it cannot collide with a plain VDN pass.
- **The CologneBonnDusseldorf-v1 configs set `TEST_NUM_ENVS: 5000`** — the default of
  `10000` does not fit in memory for the largest environment. Pass it yourself if you
  evaluate that environment without one of its configs.
- **`eval_returns_format.py` is not something you run** — it's the shared definition of
  the on-disk format: the CSV reader and writer, the `eval_returns/<algorithm>/seed<n>/`
  layout, and the `eval_meta.yaml` sidecar. `generate_eval_returns.py` imports the writer
  half and `compute_eval_statistics.py` the reader half, so the format lives in one place.
  Method A is the exception: training writes its returns from
  `jaxmarl/wrappers/baselines.py`, which duplicates the `np.savetxt` call rather than
  importing it — the two have to be kept byte-compatible by hand. Worth knowing only if
  you change the format.

# See also

- [`../Reproducing_Results.md`](../Reproducing_Results.md) — how these steps fit into the full pipeline, from training through to the figures.
