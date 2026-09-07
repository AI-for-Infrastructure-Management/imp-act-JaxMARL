# Hyperparameter Tuning on SLURM

This guide describes how the hyperparameter sweeps for all five algorithms
(**MAPPO, IPPO, QMIX, VDN, PQN**) are run on a SLURM cluster, and how to
reproduce them. The same pipeline is used for every algorithm and map; the
examples below use MAPPO on Cologne-v1.

The pipeline is a **random search** run as independent SLURM jobs, one job per
hyperparameter combination:

```
sweep config (yaml)
      │  1. sample combos into a manifest CSV     slurm_scripts/sample_random_search.py
      ▼
      │  2. submit one sbatch job per combo       slurm_scripts/launch_hypers_sweep.sh
      ▼
training runs on /scratch
      │  3. sync offline W&B runs (if offline)    slurm_scripts/wand_sync_hypers_sweep.sh
      ▼
W&B project (compare runs by test_returned_episode_returns)
```

## Files involved

| File | Purpose |
|------|---------|
| `experiments/config/hyper_parameter_tuning/{env}/{alg}_sweep.yaml` | Search space per algorithm and map |
| `experiments/config/hyper_parameter_tuning/{env}/{alg}_road_env.yaml` | Base config the overrides are applied to |
| `slurm_scripts/launch_hypers_sweep.sh` | Samples combos and submits the SLURM jobs |
| `slurm_scripts/sample_random_search.py` | Random sampling into the manifest CSV (called by `launch_hypers_sweep.sh`) |
| `slurm_scripts/wand_sync_hypers_sweep.sh` | Syncs offline W&B runs to the cloud |

## Naming

`ALG` (accepted aliases in parentheses):

| `ALG` | Training script | Sweep yaml |
|-------|-----------------|------------|
| `mappo_rnn` (`mappo`) | `experiments/mappo_rnn_road_env.py` | `mappo_sweep.yaml` |
| `ippo_rnn` (`ippo`) | `experiments/ippo_rnn_road_env.py` | `ippo_sweep.yaml` |
| `qmix_rnn` (`qmix`) | `experiments/qmix_rnn_road_env.py` | `qmix_sweep.yaml` |
| `vdn_rnn` (`vdn`) | `experiments/vdn_rnn_road_env.py` | `vdn_sweep.yaml` |
| `pqn_rnn` (`pqn`, `pqn_vdn_rnn`) | `experiments/pqn_vdn_rnn_road_env.py` | `pqn_sweep.yaml` |

`CONFIG_DIR` (relative to `experiments/config/`), one per map:

- `hyper_parameter_tuning/toy_example_v2` (ToyExample-v2)
- `hyper_parameter_tuning/cologne_v1` (Cologne-v1)
- `hyper_parameter_tuning/cologne_bonn_dusseldorf_v1` (CologneBonnDusseldorf-v1)

## 0. One-time setup

The sweep scripts expect a Python 3.10 venv at `.venv` in the repo root, with
the project (CUDA JAX build) and the `imp-act` submodule installed. Using
[uv](https://docs.astral.sh/uv/), from the repo root:

```bash
uv venv --python 3.10 "/scratch/${USER}/imp-act-jaxmarl/.venv"  # venv on scratch (fast I/O)
ln -s "/scratch/${USER}/imp-act-jaxmarl/.venv" .venv            # symlink into the repo
source .venv/bin/activate

uv pip install -e ".[algs]"
uv pip install "jax[cuda12]==0.4.30"   # swap in the CUDA build (same version as pyproject.toml)
uv pip install -r imp-act/requirements/requirements.txt
uv pip install -e ./imp-act
```

Plain `python -m venv` + `pip` works the same way if uv is not available.
Check GPU visibility on a compute node with
`python -c 'import jax; print(jax.devices())'`.

## 1. Define the search space

Each sweep yaml (e.g.
`experiments/config/hyper_parameter_tuning/cologne_v1/mappo_sweep.yaml`)
follows the W&B sweep format. Only the `parameters:` block (and optional
`SWEEP_SEEDS`) is read by the sampler:

- `value: x` — fixed for all runs.
- `values: [a, b, c]` — searched over; the random search samples one entry per combo.
- `SWEEP_SEEDS: [101, 102, 103]` (top level) — each sampled combo is submitted
  once per seed, labelled `C0001_1`, `C0001_2`, … This is the standard setup:
  three seeds per combination (so `N_COMBOS=50` submits 150 jobs), and the
  best combination is chosen by averaging over seeds — group runs by
  `COMBO_ID` in W&B. If omitted, each combo runs once with the `SEED` from
  the base config.

Do not change the fixed values (e.g. `NUM_UPDATES`, `NUM_ENVS`) unless you
intend to change the experiment: the SLURM time/memory tables in
`launch_hypers_sweep.sh` were measured for these settings.

## 2. Launch the sweep

`launch_hypers_sweep.sh` takes no arguments — edit the *Manual Configuration* block
at the top, then run it:

```bash
bash slurm_scripts/launch_hypers_sweep.sh
```

| Variable | Meaning |
|----------|---------|
| `ALG` | Algorithm, e.g. `mappo_rnn` (see table above) |
| `CONFIG_DIR` | Map config dir, e.g. `hyper_parameter_tuning/cologne_v1` |
| `N_COMBOS` | Number of **new** combos to sample and submit per invocation |
| `RANDOM_SEED` | Seed for the *sampler* (not the training seed) — keep at `0` for reproducibility |
| `PARTITION`, `ACCOUNT` | **Cluster-specific** — your cluster's GPU partition and project account |
| `TIME_LIMIT`, `CPUS_PER_TASK`, `MEM_PER_CPU` | Leave empty to use the built-in per-(map, algorithm) resource tables; set to override |
| `GPUS_PER_TASK` | GPUs per job (default `1`) |
| `WANDB_MODE` | `offline` (default) if compute nodes have no internet; `online` if they do — then step 3 is not needed |
| `SCRATCH_ROOT` | Where manifests, run outputs, and checkpoints go (default `/scratch/$USER/imp-act-jaxmarl`) |

What it does:

1. Samples `N_COMBOS` new, unique combos from the sweep yaml and appends them
   to the **manifest**
   `${SCRATCH_ROOT}/${CONFIG_DIR}/${SWEEP_NAME}/sampled_configs.csv`
   (for the example: `.../hyper_parameter_tuning/cologne_v1/mappo_sweep/sampled_configs.csv`).
   The manifest is the reproducibility record of the sweep — **keep it**.
2. Resolves the SLURM time limit and memory from the per-(map, algorithm)
   resource tables inside the script.
3. Shows the parsed search space (searched vs. fixed parameters, total number
   of combinations — check it to catch typos in the sweep yaml) and a summary,
   and asks for confirmation. Then it submits one `sbatch` job per new
   manifest row. Each job writes everything (Hydra output,
   checkpoints, W&B files, SLURM logs) to its own directory
   `${SCRATCH_ROOT}/${CONFIG_DIR}/${SWEEP_NAME}/${run_label}/`.

Each invocation only submits the newly added combos, so to grow a sweep just
run the script again — e.g. two runs with `N_COMBOS=50` give 100 distinct
combos. It does *not* detect failed jobs; resubmit those manually if needed.

## 3. Sync results to W&B (offline runs only)

Skip this step if you ran with `WANDB_MODE=online`. Otherwise, from a node
with internet access (typically the login node), after `wandb login`:

```bash
bash slurm_scripts/wand_sync_hypers_sweep.sh
```

Edit its *Manual Configuration* block first: `ALG` and `CONFIG_DIR` must match
the sweep you launched, and `PROJECT` is the W&B project the runs are uploaded
into (suggested convention: `{alg}-hypers-sweep-{map}`, e.g.
`mappo-hypers-sweep-cologne-v1`). The script finds all `offline-run-*`
directories under the sweep's scratch directory and syncs them in parallel
(`JOBS` at a time). Already-synced runs are skipped, so it is safe to re-run
while jobs are still finishing.

The best combination per algorithm and map is then selected by comparing runs
in the W&B project on `test_returned_episode_returns` (each run's `COMBO_ID`
and `RUN_LABEL` are stored in its config).

## Porting to another cluster — checklist

1. Adjust the venv path in step 0 if your cluster's scratch space is not `/scratch/$USER`.
2. `slurm_scripts/launch_hypers_sweep.sh`: set `PARTITION` and `ACCOUNT`; set `SCRATCH_ROOT` to match step 1; set `WANDB_MODE=online` if your compute nodes have internet access.
3. The resource tables in `launch_hypers_sweep.sh` were measured on NVIDIA V100 nodes.
   On slower/faster GPUs adjust `TIME_LIMIT` (or the table entries), not the experiment configs.
4. `slurm_scripts/wand_sync_hypers_sweep.sh` (offline mode only): set `SCRATCH_ROOT` and your W&B `ENTITY`/`PROJECT`.

## Reproducibility notes

- The manifest CSV (`sampled_configs.csv`) fully determines a sweep: run
  label, seed, and every Hydra override per job. Archive it alongside the
  results.
- Sampling is deterministic given the sweep yaml, `RANDOM_SEED`, and the
  existing manifest contents.
- The exact training commands are reconstructible from a manifest row: each
  field `KEY=value` is passed as a Hydra override to the training script with
  `--config-path experiments/config/${CONFIG_DIR} --config-name {alg}_road_env`.
- The chosen best hyperparameters per algorithm/map are frozen in
  `experiments/config/final_runs/`; see
  [Reproducing_Results.md](Reproducing_Results.md) for the final seeded runs.
