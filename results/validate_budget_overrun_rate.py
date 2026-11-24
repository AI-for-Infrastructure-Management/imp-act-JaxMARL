"""Fraction of timesteps in which the budget constraint downgrades actions.

Validates the claim in the results section: "In evaluations of the best VDN policy
in Cologne, budget overruns triggered this mechanism in approximately X% of
timesteps on average."

Usage (from results/):
    python validate_budget_overrun_rate.py [--map Cologne-v1] [--alg vdn_rnn] [--episodes 1000]
"""

import argparse
import sys
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf
from common import best_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # *_road_env.py modules
from vdn_rnn_road_env import make_get_rollout_data  # noqa: E402

from jaxmarl.wrappers.baselines import load_params

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="Cologne-v1")
parser.add_argument("--alg", default="vdn_rnn")
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# The best final-run checkpoint (rank 1), from the Zenodo release
best, run_dir = best_checkpoint(args.map, args.alg)
print(f"{args.map} / {args.alg}: seed {best['seed']}, checkpoint {best['checkpoint_step']}")

config = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"))
params = load_params(str(next((run_dir / "checkpoints").glob("*/checkpoint_*.safetensors"))))

rollout = jax.jit(make_get_rollout_data(config, args.episodes))
_, _, _, _, infos = rollout(jax.random.PRNGKey(args.seed), params)

applied = np.asarray(infos["budget_constraints_applied"], dtype=bool)  # (T, episodes)
per_episode = applied.mean(axis=0)
se = per_episode.std(ddof=1) / np.sqrt(len(per_episode))
print(
    f"Budget constraint applied in {100 * applied.mean():.2f}% of timesteps "
    f"(±{100 * 1.96 * se:.2f}% 95% CI, {args.episodes} episodes x {applied.shape[0]} steps)"
)
