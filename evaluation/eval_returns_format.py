"""The artifact shared by the two halves of evaluation.

Evaluation has one artifact and one consumer:

    training (STORE_EVAL_RETURNS) ─┐
                                   ├─→ eval_returns_<step>.csv ─→ compute_eval_statistics.py
    generate_eval_returns.py ──────┘        raw episode returns          the results table

`generate_eval_returns.py` re-runs greedy inference (JAX, GPU) and writes returns; it
computes no statistics. `compute_eval_statistics.py` reads returns from either producer
and is the only thing that bootstraps or emits a results row.

What is genuinely shared is the *format*, and someone changing it needs the writer
and the reader in front of them at once.
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import yaml

# Directory under a run that holds post-hoc evaluations, one subtree per
# (algorithm variant, eval seed). Training writes its own returns elsewhere.
EVAL_RETURNS_DIRNAME = "eval_returns"
EVAL_META_FILENAME = "eval_meta.yaml"
EVAL_RETURNS_PREFIX = "eval_returns_"


def write_eval_returns(path, returns):
    """Write raw per-episode returns in the format the trainer uses.

    Must stay byte-compatible with `make_store_eval_returns` in
    `jaxmarl/wrappers/baselines.py`, which is `np.savetxt` with a single
    `episode_return` header and no comment prefix — hence the default `%.18e`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path, np.asarray(returns), delimiter=",", header="episode_return", comments=""
    )


def read_eval_returns(path):
    """The other half of `write_eval_returns`. float64 regardless of what was written."""
    return np.atleast_1d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64))


def eval_returns_dir(run_dir, algorithm, eval_seed):
    """Where a post-hoc evaluation of `run_dir` writes its returns.

    Keyed on both the algorithm variant and the eval seed, so a budget-aware pass
    cannot collide with a plain one, nor a second eval seed with the first, nor
    either with the returns training wrote for itself.
    """
    return run_dir / EVAL_RETURNS_DIRNAME / str(algorithm) / f"seed{eval_seed}"


def write_eval_meta(eval_dir, **fields):
    """Record what produced these returns.

    A run's own `config.yaml` describes training, so it cannot say which variant was
    evaluated, under which eval seed, over how many episodes. Without this the
    consumer would label a budget-aware pass `vdn_rnn` and give it the training
    run's eval semantics.
    """
    meta = {
        "produced_by": "generate_eval_returns.py",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        **fields,
    }
    with open(eval_dir / EVAL_META_FILENAME, "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    return meta


def read_eval_meta(eval_dir):
    """The sidecar for `eval_dir`, or None when there is none.

    Absence is the normal case for returns written during training, and means the
    consumer should fall back to the run's `config.yaml`.
    """
    path = eval_dir / EVAL_META_FILENAME
    if not path.is_file():
        return None
    with open(path) as fh:
        return yaml.safe_load(fh)


def eval_returns_complete(eval_dir, expected_files):
    """Whether `eval_dir` already holds a finished evaluation.

    The artifact is its own cache: a complete directory is skippable, a partial one
    is visibly partial, and nothing but the files themselves has to be trusted.
    """
    if read_eval_meta(eval_dir) is None:
        return False
    written = sum(
        1 for name in os.listdir(eval_dir) if name.startswith(EVAL_RETURNS_PREFIX)
    )
    return written == expected_files


def resolve_algorithm(train_alg_name, eval_vdn_ba=False):
    """What belongs in the `algorithm` column, and in the returns directory name.

    Budget-aware evaluation is not a separately trained algorithm: `EVAL_VDN_BA`
    re-evaluates a `vdn_rnn` policy with budget-constrained greedy action selection.
    It is reported under its own name because such rows are otherwise identical to a
    plain VDN evaluation in every identifying column, and downstream code — notably
    `plotting/figure_3.ipynb` — groups by `algorithm`.
    """
    if eval_vdn_ba and train_alg_name == "vdn_rnn":
        return "vdn_ba_rnn"
    return train_alg_name
