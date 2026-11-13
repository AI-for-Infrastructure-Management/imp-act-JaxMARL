"""Fraction of timesteps in which the budget constraint downgrades actions.

Validates the claim in the results section: "In evaluations of the best VDN policy
in Cologne, budget overruns triggered this mechanism in approximately X% of
timesteps on average."

Usage (from evaluation/):
    python validate_budget_overrun_rate.py [--map Cologne-v1] [--alg vdn_rnn] [--episodes 1000]
"""

import argparse
import os

import jax
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from vdn_rnn_road_env import make_get_rollout_data

from jaxmarl.wrappers.baselines import load_params

CHECKPOINT_ROOT = "data/best_checkpoints"

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="Cologne-v1")
parser.add_argument("--alg", default="vdn_rnn")
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

manifest = pd.read_csv(f"{CHECKPOINT_ROOT}/best_checkpoints.csv")
best = manifest[
    (manifest["map_name"] == args.map) & (manifest["algorithm"] == args.alg)
].iloc[0]
run_dir = os.path.join(CHECKPOINT_ROOT, best["run_dir"])
print(f"{args.map} / {args.alg}: {best['run_dir']}, checkpoint {best['checkpoint_id']}")

config = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
params = load_params(os.path.join(CHECKPOINT_ROOT, best["local_path"]))

rollout = jax.jit(make_get_rollout_data(config, args.episodes))
_, _, _, _, infos = rollout(jax.random.PRNGKey(args.seed), params)

applied = np.asarray(infos["budget_constraints_applied"], dtype=bool)  # (T, episodes)
per_episode = applied.mean(axis=0)
se = per_episode.std(ddof=1) / np.sqrt(len(per_episode))
print(
    f"Budget constraint applied in {100 * applied.mean():.2f}% of timesteps "
    f"(±{100 * 1.96 * se:.2f}% 95% CI, {args.episodes} episodes x {applied.shape[0]} steps)"
)
