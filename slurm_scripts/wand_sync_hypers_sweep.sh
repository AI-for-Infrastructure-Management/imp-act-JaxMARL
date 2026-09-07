#!/bin/bash

set -euo pipefail

# Uploads the offline W&B runs of a hyperparameter sweep to the W&B cloud.
#
# launch_hypers_sweep.sh runs each trial with WANDB_MODE=offline, so every job
# leaves an offline-run-* directory in its run directory on scratch. This
# script finds all of them under the sweep's scratch directory and syncs them
# to the given W&B entity/project, several in parallel.
#
# Key points:
#   - Not needed if the sweep ran with WANDB_MODE=online.
#   - Run it from a node with internet access (typically the login node),
#     after `wandb login`.
#   - Safe to re-run at any time, also while jobs are still finishing:
#     already-synced runs are skipped.
#
# Required:
#   Edit the variables in the "Manual Configuration" section before running.
#   This script does not accept command-line arguments.
#
# Usage:
#   bash slurm_scripts/wand_sync_hypers_sweep.sh
#
# See hyperparameter_tuning.md for the full workflow.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/.venv/bin/activate"

############################
# Manual Configuration
############################

# Algorithm and map of the sweep to sync — must match the values used in
# launch_hypers_sweep.sh.
ALG="mappo_rnn"
CONFIG_DIR="hyper_parameter_tuning/cologne_v1"
# Must match SCRATCH_ROOT in launch_hypers_sweep.sh.
SCRATCH_ROOT="/scratch/${USER}/imp-act-jaxmarl"
# Number of runs to sync in parallel.
JOBS="10"
# W&B entity (team/user) and project to upload the runs into.
# Suggested project convention: {alg}-hypers-sweep-{map},
# e.g. "mappo-hypers-sweep-cologne-v1".
ENTITY="imp-act"
PROJECT="hyper_parameter_tuning"

[ -n "${PROJECT}" ] || { echo "Set PROJECT before running." >&2; exit 1; }

############################
# Algorithm Selection
############################
# Maps ALG to the sweep directory name on scratch — must stay in sync with
# the SWEEP_NAME mapping in launch_hypers_sweep.sh.

case "${ALG}" in
  mappo|mappo_rnn)
    SWEEP_NAME="mappo_sweep"
    ;;
  ippo|ippo_rnn)
    SWEEP_NAME="ippo_sweep"
    ;;
  qmix|qmix_rnn)
    SWEEP_NAME="qmix_sweep"
    ;;
  vdn|vdn_rnn)
    SWEEP_NAME="vdn_sweep"
    ;;
  pqn|pqn_rnn|pqn_vdn_rnn)
    SWEEP_NAME="pqn_sweep"
    ;;
  *)
    echo "Unsupported ALG=${ALG}" >&2
    exit 1
    ;;
esac

############################
# Run Discovery
############################
# Collect the offline-run-* directories of all trials under the sweep
# directory (one per run, created by wandb in each job's WANDB_DIR).

SWEEP_DIR="${SCRATCH_ROOT}/${CONFIG_DIR}/${SWEEP_NAME}"
[ -d "${SWEEP_DIR}" ] || { echo "Missing sweep dir: ${SWEEP_DIR}" >&2; exit 1; }

mapfile -d '' -t offline_runs < <(
  find "${SWEEP_DIR}" -type d -name 'offline-run-*' -print0 | sort -z
)

[ "${#offline_runs[@]}" -gt 0 ] || {
  echo "No offline wandb runs found under ${SWEEP_DIR}" >&2
  exit 1
}

############################
# Sync
############################
# Upload JOBS runs at a time. --mark-synced flags a run directory as done and
# --no-include-synced skips flagged ones, which is what makes re-runs cheap.

printf 'Syncing %s offline runs from %s with %s parallel jobs\n' \
  "${#offline_runs[@]}" "${SWEEP_DIR}" "${JOBS}"

printf '%s\0' "${offline_runs[@]}" |
  xargs -0 -r -P "${JOBS}" -I{} \
    wandb sync "{}" \
      --entity "${ENTITY}" \
      --project "${PROJECT}" \
      --mark-synced \
      --no-include-synced

echo "Finished syncing all runs from ${SWEEP_DIR}"
