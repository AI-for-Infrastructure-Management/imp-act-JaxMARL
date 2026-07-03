#!/bin/bash

set -euo pipefail

# Random-search hyperparameter sweep launcher.
#
# Samples N_COMBOS new hyperparameter combinations from the sweep yaml
# (experiments/config/${CONFIG_DIR}/${SWEEP_NAME}.yaml), appends them to a
# manifest CSV on scratch, and submits one SLURM job per new combination
# after showing a summary and asking for confirmation.
#
# Key points:
#   - The manifest (${SCRATCH_ROOT}/${CONFIG_DIR}/${SWEEP_NAME}/sampled_configs.csv)
#     records one row per run: run label, optional SEED, and all Hydra
#     overrides of that trial. It is the reproducibility record of the sweep.
#   - Only the rows added by the current invocation are submitted, so to grow
#     a sweep simply run the script again: two runs with N_COMBOS=50 give
#     100 distinct combinations.
#   - Each job writes its SLURM logs, checkpoints, and (offline) W&B run to
#     its own directory ${SCRATCH_ROOT}/${CONFIG_DIR}/${SWEEP_NAME}/${run_label}/.
#   - SLURM time and memory are resolved from the per-(map, algorithm) tables
#     below unless TIME_LIMIT / CPUS_PER_TASK / MEM_PER_CPU are set manually.
#
# Required:
#   Edit the variables in the "Manual Configuration" section before running.
#   This script does not accept command-line arguments.
#
# Usage:
#   bash slurm_scripts/launch_hypers_sweep.sh
#
# See hyperparameter_tuning.md for the full workflow (setup and syncing).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

############################
# Manual Configuration
############################

# Algorithm: mappo_rnn | ippo_rnn | qmix_rnn | vdn_rnn | pqn_rnn
ALG="mappo_rnn"
# Map config dir, relative to experiments/config/:
#   hyper_parameter_tuning/{toy_example_v2 | cologne_v1 | cologne_bonn_dusseldorf_v1}
CONFIG_DIR="hyper_parameter_tuning/cologne_v1"
# Number of NEW combinations to sample and submit in this invocation.
N_COMBOS="10"
# Seed for the combination sampler (not the training seed; that comes from the
# base config or SWEEP_SEEDS in the sweep yaml). Keep at 0 for reproducibility.
RANDOM_SEED="0"

# SLURM settings — cluster-specific, adjust for your cluster.
PARTITION="gpu-v100"
ACCOUNT="research-abe-aet"
# Leave the next three empty to resolve them from the resource tables below.
TIME_LIMIT=""
CPUS_PER_TASK=""
MEM_PER_CPU=""
GPUS_PER_TASK="1"

# Use "online" if your compute nodes have internet access; with "offline",
# sync the runs afterwards with slurm_scripts/wand_sync_hypers_sweep.sh.
WANDB_MODE="offline"
# Where manifests, run outputs, and checkpoints are written.
SCRATCH_ROOT="/scratch/${USER}/imp-act-jaxmarl"

############################
# Algorithm Selection
############################
# Normalizes ALG and maps it to the training script and config file names.
# The normalized ALG is also the algorithm key into the resource tables below.
case "${ALG}" in
  mappo|mappo_rnn)
    ALG="mappo_rnn"
    SWEEP_NAME="mappo_sweep"
    PROGRAM="experiments/mappo_rnn_road_env.py"
    CONFIG_NAME="mappo_rnn_road_env"
    ;;
  ippo|ippo_rnn)
    ALG="ippo_rnn"
    SWEEP_NAME="ippo_sweep"
    PROGRAM="experiments/ippo_rnn_road_env.py"
    CONFIG_NAME="ippo_rnn_road_env"
    ;;
  qmix|qmix_rnn)
    ALG="qmix_rnn"
    SWEEP_NAME="qmix_sweep"
    PROGRAM="experiments/qmix_rnn_road_env.py"
    CONFIG_NAME="qmix_rnn_road_env"
    ;;
  vdn|vdn_rnn)
    ALG="vdn_rnn"
    SWEEP_NAME="vdn_sweep"
    PROGRAM="experiments/vdn_rnn_road_env.py"
    CONFIG_NAME="vdn_rnn_road_env"
    ;;
  pqn|pqn_rnn|pqn_vdn_rnn)
    ALG="pqn_rnn"
    SWEEP_NAME="pqn_sweep"
    PROGRAM="experiments/pqn_vdn_rnn_road_env.py"
    CONFIG_NAME="pqn_rnn_road_env"
    ;;
  *)
    echo "Unsupported ALG=${ALG}" >&2
    exit 1
    ;;
esac

SWEEP_FILE="${ROOT_DIR}/experiments/config/${CONFIG_DIR}/${SWEEP_NAME}.yaml"
DEFAULT_CONFIG_FILE="${ROOT_DIR}/experiments/config/${CONFIG_DIR}/${CONFIG_NAME}.yaml"

[ -f "${SWEEP_FILE}" ] || { echo "Missing sweep config: ${SWEEP_FILE}" >&2; exit 1; }
[ -f "${DEFAULT_CONFIG_FILE}" ] || { echo "Missing default config: ${DEFAULT_CONFIG_FILE}" >&2; exit 1; }
[ -x "${PYTHON_BIN}" ] || { echo "Missing python executable: ${PYTHON_BIN}" >&2; exit 1; }

CONFIG_BASENAME="${CONFIG_DIR##*/}"

############################
# Resource Tables
############################
# Measured on NVIDIA V100 nodes, with a safety buffer already included. On
# slower/faster hardware adjust the entries (or set TIME_LIMIT etc. manually
# above), not the experiment configs.

# SLURM time limit per (map, algorithm).
declare -A TIME_LIMITS

# ToyExample-v2
TIME_LIMITS["toy_example_v2,vdn_rnn"]="01:03:00"
TIME_LIMITS["toy_example_v2,qmix_rnn"]="06:36:00"
TIME_LIMITS["toy_example_v2,pqn_rnn"]="03:21:00"
TIME_LIMITS["toy_example_v2,mappo_rnn"]="01:30:00"
TIME_LIMITS["toy_example_v2,ippo_rnn"]="01:30:00"

# Cologne-v1
TIME_LIMITS["cologne_v1,vdn_rnn"]="04:30:00"
TIME_LIMITS["cologne_v1,qmix_rnn"]="02:00:00"
TIME_LIMITS["cologne_v1,pqn_rnn"]="17:00:00"
TIME_LIMITS["cologne_v1,mappo_rnn"]="04:00:00"
TIME_LIMITS["cologne_v1,ippo_rnn"]="04:00:00"

# CologneBonnDusseldorf-v1
TIME_LIMITS["cologne_bonn_dusseldorf_v1,vdn_rnn"]="19:30:00"
TIME_LIMITS["cologne_bonn_dusseldorf_v1,qmix_rnn"]="22:30:00"
TIME_LIMITS["cologne_bonn_dusseldorf_v1,pqn_rnn"]="1-18:00:00"
TIME_LIMITS["cologne_bonn_dusseldorf_v1,mappo_rnn"]="10:00:00"
TIME_LIMITS["cologne_bonn_dusseldorf_v1,ippo_rnn"]="12:00:00"

# Total memory (GB) per (map, algorithm), safety buffer included. Requested
# as cpus-per-task × 3G per CPU.
declare -A MEMORY_GB

# ToyExample-v2
MEMORY_GB["toy_example_v2,vdn_rnn"]="2"
MEMORY_GB["toy_example_v2,qmix_rnn"]="6"
MEMORY_GB["toy_example_v2,pqn_rnn"]="1"
MEMORY_GB["toy_example_v2,mappo_rnn"]="3"
MEMORY_GB["toy_example_v2,ippo_rnn"]="3"

# Cologne-v1
MEMORY_GB["cologne_v1,vdn_rnn"]="24"
MEMORY_GB["cologne_v1,qmix_rnn"]="31"
MEMORY_GB["cologne_v1,pqn_rnn"]="9"
MEMORY_GB["cologne_v1,mappo_rnn"]="6"
MEMORY_GB["cologne_v1,ippo_rnn"]="6"

# CologneBonnDusseldorf-v1
MEMORY_GB["cologne_bonn_dusseldorf_v1,vdn_rnn"]="48"
MEMORY_GB["cologne_bonn_dusseldorf_v1,qmix_rnn"]="41"
MEMORY_GB["cologne_bonn_dusseldorf_v1,pqn_rnn"]="9"
MEMORY_GB["cologne_bonn_dusseldorf_v1,mappo_rnn"]="9"
MEMORY_GB["cologne_bonn_dusseldorf_v1,ippo_rnn"]="9"

resource_key="${CONFIG_BASENAME},${ALG}"

############################
# Resource Resolution
############################
# Fill TIME_LIMIT / CPUS_PER_TASK / MEM_PER_CPU from the tables above,
# unless they were set manually in the configuration section.

if [ -z "${TIME_LIMIT}" ]; then
  TIME_LIMIT="${TIME_LIMITS[${resource_key}]:-}"
  [ -n "${TIME_LIMIT}" ] || { echo "No sweep time entry for ${resource_key}" >&2; exit 1; }
fi

if [ -z "${CPUS_PER_TASK}" ] || [ -z "${MEM_PER_CPU}" ]; then
  mem_gb="${MEMORY_GB[${resource_key}]:-}"
  [ -n "${mem_gb}" ] || { echo "No sweep memory entry for ${resource_key}" >&2; exit 1; }
  mem_per_cpu_gb=3
  [ -n "${CPUS_PER_TASK}" ] || CPUS_PER_TASK=$(( (mem_gb + mem_per_cpu_gb - 1) / mem_per_cpu_gb ))
  [ -n "${MEM_PER_CPU}" ] || MEM_PER_CPU="${mem_per_cpu_gb}G"
fi

############################
# Manifest Sampling
############################
# Append N_COMBOS new unique combinations to the manifest CSV. Rows already
# present (from previous invocations) are never re-submitted; only the rows
# added here are picked up by the submission loop below.

SCRATCH_PATH="${SCRATCH_ROOT}/${CONFIG_DIR}/${SWEEP_NAME}"
MANIFEST_FILE="${SCRATCH_PATH}/sampled_configs.csv"
mkdir -p "${SCRATCH_PATH}"

# Rows already in the manifest before sampling; everything after this line
# number is what the current invocation added and will submit.
existing_count=0
if [ -f "${MANIFEST_FILE}" ]; then
  existing_count="$(wc -l < "${MANIFEST_FILE}")"
fi

# The sampler prints the parsed search space (searched vs. fixed parameters
# and total combinations) — review it in the confirmation step below.
"${PYTHON_BIN}" "${ROOT_DIR}/slurm_scripts/sample_random_search.py" \
  --sweep-config "${SWEEP_FILE}" \
  --output-file "${MANIFEST_FILE}" \
  --num-samples "${N_COMBOS}" \
  --random-seed "${RANDOM_SEED}"

[ -f "${MANIFEST_FILE}" ] || { echo "Missing sampled config file: ${MANIFEST_FILE}" >&2; exit 1; }

mapfile -t new_configs < <(tail -n "+$((existing_count + 1))" "${MANIFEST_FILE}")
if [ "${#new_configs[@]}" -eq 0 ]; then
  echo "No new sampled configs to submit."
  exit 0
fi

############################
# Summary and Confirmation
############################
# Show what would be submitted and ask before calling sbatch. Declining rolls
# the manifest back to its previous state, as if this invocation never ran.

echo ""
echo "==================== Sweep summary ===================="
echo "Algorithm      : ${ALG} (${PROGRAM})"
echo "Config dir     : experiments/config/${CONFIG_DIR}"
echo "Sweep config   : ${SWEEP_FILE}"
echo "Manifest       : ${MANIFEST_FILE}"
echo "Runs           : ${existing_count} existing, ${#new_configs[@]} new (combos x seeds)"
echo "Jobs to submit : ${#new_configs[@]}"
echo "Partition      : ${PARTITION} (account: ${ACCOUNT})"
echo "Resources/job  : time=${TIME_LIMIT} cpus=${CPUS_PER_TASK} mem-per-cpu=${MEM_PER_CPU} gpus=${GPUS_PER_TASK}"
echo "W&B mode       : ${WANDB_MODE}"
echo "Output dir     : ${SCRATCH_PATH}"
echo "========================================================"

read -r -p "Submit ${#new_configs[@]} jobs? [y/N] " reply
case "${reply}" in
  y|Y|yes|YES)
    ;;
  *)
    # Roll back the newly sampled manifest rows so the next run re-samples them.
    if [ "${existing_count}" -gt 0 ]; then
      head -n "${existing_count}" "${MANIFEST_FILE}" > "${MANIFEST_FILE}.tmp"
      mv "${MANIFEST_FILE}.tmp" "${MANIFEST_FILE}"
    else
      rm -f "${MANIFEST_FILE}"
    fi
    echo "Aborted. No jobs submitted; manifest restored."
    exit 0
    ;;
esac

############################
# Job Submission
############################
# One sbatch job per new manifest row. A row is parsed as: run label
# (e.g. C0007), an optional SEED=... field, and the Hydra overrides of the
# trial, which are appended to the training command.

for line in "${new_configs[@]}"; do
  line="${line%$'\r'}"
  IFS=',' read -ra fields <<< "${line}"
  run_label=""
  seed=""
  trial_overrides=()

  for field in "${fields[@]}"; do
    if [[ "${field}" != *=* ]]; then
      run_label="${field}"
    elif [[ "${field}" == SEED=* ]]; then
      seed="${field#*=}"
    else
      trial_overrides+=("${field}")
    fi
  done

  [ -n "${run_label}" ] || { echo "Malformed manifest row: ${line}" >&2; exit 1; }

  # Run labels are C<combo>_<seed_id> when SWEEP_SEEDS is used, so stripping
  # the seed suffix yields the combo id shared by all seeds of a combination.
  combo_id="${run_label%%_*}"
  run_scratch_path="${SCRATCH_PATH}/${run_label}"
  job_name="${SWEEP_NAME}_${run_label}"
  mkdir -p "${run_scratch_path}"

  # Build the training command for sbatch --wrap (%q quotes each part for the
  # shell). WANDB_DIR, hydra.run.dir, and SAVE_PATH all point at the run's own
  # directory so every output of the trial lands in one place; +COMBO_ID and
  # +RUN_LABEL are stored in the run config to identify it in W&B later.
  printf -v command \
    'cd %q && export WANDB_DIR=%q && %q %q --config-path %q --config-name %q hydra.run.dir=%q hydra.output_subdir=null WANDB_MODE=%q HYP_TUNE=False SAVE_PATH=%q' \
    "${ROOT_DIR}" \
    "${run_scratch_path}" \
    "${PYTHON_BIN}" \
    "${ROOT_DIR}/${PROGRAM}" \
    "${ROOT_DIR}/experiments/config/${CONFIG_DIR}" \
    "${CONFIG_NAME}" \
    "${run_scratch_path}" \
    "${WANDB_MODE}" \
    "${run_scratch_path}"

  if [ -n "${seed}" ]; then
    printf -v command '%s %q' "${command}" "SEED=${seed}"
  fi

  printf -v command '%s %q' "${command}" "+COMBO_ID=${combo_id}"
  printf -v command '%s %q' "${command}" "+RUN_LABEL=${run_label}"

  for override in "${trial_overrides[@]}"; do
    printf -v command '%s %q' "${command}" "${override}"
  done

  echo "Submitting: ${run_label} ${seed:+SEED=${seed} }${trial_overrides[*]}"
  sbatch \
    --job-name="${job_name}" \
    --output="${run_scratch_path}/slurm-%j.out" \
    --error="${run_scratch_path}/slurm-%j.err" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --ntasks=1 \
    --time="${TIME_LIMIT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus-per-task="${GPUS_PER_TASK}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --wrap="${command}"
done
