"""Per-checkpoint eval statistics from raw episode returns.

The only thing in the pipeline that computes a statistic or writes a results row.
It reads `eval_returns_<step>.csv` wherever they came from — written by training
under `STORE_EVAL_RETURNS: True`, or by `generate_eval_returns.py` re-running
inference — and emits `inference_results.csv` plus per-map/algorithm splits.
numpy + scipy + pyyaml only; no JAX, no GPU, no wandb.

Provenance comes from an `eval_meta.yaml` sidecar when the returns were produced
post-hoc, and otherwise from the run's own `config.yaml`. Returns written during
training carry no fixed eval seed — their draws come from the training RNG — so
`eval_seed` is empty for those rows and set for post-hoc ones. That is the only
statistical difference between the two sources: comparisons that need common random
numbers across runs require a post-hoc pass at a fixed `EVAL_SEED`.

Per checkpoint it reports the mean, its standard error and a BCa bootstrap CI:

1. Everything is computed in float64. Training logs float32 means; measured against
   the 100 logged `test_returned_episode_returns` of a real Cologne VDN run, the
   two agree to 1.6e-7 relative (float32 eps is 1.2e-7) — a single-ulp
   representation difference, not accumulated error.
2. The bootstrap RNG is seeded, so CI bounds are reproducible. `--bootstrap-seed -1`
   leaves it unseeded.
3. `BOOTSTRAP_BATCH` bounds BCa's peak memory, which otherwise blows up by
   materialising both an n_resamples x n and an n x (n-1) array. Batching leaves
   the CI bit-identical.

`--n-resamples` defaults to 1000 rather than scipy's 9999: at full eval size
(n=10_000 episodes) the CI bounds are stable to well within the standard error, and
every row records the value used. Measured at that scale (one core): 1.6 s and
0.4 GB per checkpoint, so 160 run/seed cells x 100 evals is ~7 CPU-hours, or about
a quarter-hour on 32 workers.

Nothing about the directory layout is assumed: runs are found by locating the eval
returns and walking up to the nearest `config.yaml`.

    python evaluation/compute_eval_statistics.py
    python evaluation/compute_eval_statistics.py $DSS/final-runs-v2 --workers 32
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import bootstrap

from eval_returns_format import read_eval_meta, read_eval_returns

log = logging.getLogger("compute_eval_statistics")

DEFAULT_ROOT = Path("outputs")  # hydra's default run dir, so a clone works unargued
# One writer now, so results need no per-producer subdirectory.
DEFAULT_OUTPUT = Path("evaluation/results")

EPISODE_HORIZON = 50  # road env max_timesteps; sets the expected episode count

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_METHOD = "BCa"  # scipy's default
BOOTSTRAP_BATCH = 1000  # bounds BCa's peak memory; never changes the result
DEFAULT_N_RESAMPLES = 1000  # scipy's default is 9999; 1000 suffices here

# The 12 columns of the published inference_results schema, in its order, so
# positional readers keep working. Provenance columns follow.
REFERENCE_COLUMNS = [
    "map_name", "algorithm", "checkpoint_dir_name", "WANDB_RUN_ID", "checkpoint_id",
    "VMAPPED_SEED", "eval_seed", "mean", "std_err", "lower_ci", "upper_ci", "bst_std_err",
]
EXTRA_COLUMNS = [
    "checkpoint_step", "num_episodes", "vmapped_seed_key", "sampled_seed",
    "n_resamples", "bootstrap_seed", "source_file",
]
COLUMNS = REFERENCE_COLUMNS + EXTRA_COLUMNS

EVAL_RETURNS_RE = re.compile(r"^eval_returns_(\d+)\.csv$")
# An offline wandb run keeps a copy of the training config under
# `wandb/offline-run-*/files/`, which would otherwise read as a nested run.
PRUNED_DIRS = {"wandb", ".hydra"}

# Identifies a row for --resume. Includes the variant and the eval seed because one
# run can hold several evaluations of the same checkpoint: the one training wrote,
# and a post-hoc pass per (algorithm variant, eval seed).
KEY_COLUMNS = ("checkpoint_dir_name", "VMAPPED_SEED", "checkpoint_id",
               "algorithm", "eval_seed")


def episode_return_stats(returns, n_resamples=DEFAULT_N_RESAMPLES, rng=None):
    """Mean, standard error and a bootstrap CI over episode returns.

    `std_err` uses the population sd (ddof=0), matching the `jnp.std` default behind
    the metrics training logs, so the two stay comparable.
    Pass `rng=None` for scipy's unseeded behaviour.
    """
    bst = bootstrap(
        data=[returns],
        statistic=np.mean,
        confidence_level=CONFIDENCE_LEVEL,
        n_resamples=n_resamples,
        method=BOOTSTRAP_METHOD,
        batch=BOOTSTRAP_BATCH,
        random_state=rng,
    )
    return {
        "mean": float(np.mean(returns)),
        "std_err": float(np.std(returns) / np.sqrt(len(returns))),
        "lower_ci": float(bst.confidence_interval.low),
        "upper_ci": float(bst.confidence_interval.high),
        "bst_std_err": float(bst.standard_error),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def split_combined(combined_path, result_dir):
    """Per-map/algorithm CSVs beside the combined table, one row group each."""
    with open(combined_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups = {}
    for row in rows:
        groups.setdefault((row["map_name"], row["algorithm"]), []).append(row)
    for (map_name, algorithm), group in sorted(groups.items()):
        write_csv(result_dir / map_name / algorithm /
                  f"inference_results_{algorithm}.csv", group)
    log.info(f"{len(rows)} rows across {len(groups)} (map, algorithm) pairs")


def find_run_dir(start, root):
    """Nearest directory at or above `start`, up to `root`, holding a config.yaml.

    Found rather than assumed, because the eval directory is not a fixed depth below
    the run directory: `<run>/checkpoints/<seed_key>/` and `<run>/<seed_key>/` both
    occur in real training runs, post-hoc passes write
    `<run>/eval_returns/<algorithm>/seed<eval_seed>/`.
    """
    current, root = start.resolve(), root.resolve()
    while not (current / "config.yaml").is_file():
        if current == root or current.parent == current:
            return None
        current = current.parent
    return current


def expected_episodes(config):
    """How many episodes an eval should hold, or None if the config can't say."""
    try:
        envs, steps = int(config["TEST_NUM_ENVS"]), int(config["TEST_NUM_STEPS"])
    except (KeyError, TypeError, ValueError):
        return None
    return envs * (steps // EPISODE_HORIZON) if steps % EPISODE_HORIZON == 0 else None


def discover_tasks(roots):
    """One task per stored eval, each already carrying its row's identity fields."""
    found = {}  # run_dir -> {eval_dir: [(path, checkpoint_id, step)]}
    for root in map(Path, roots):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in PRUNED_DIRS)
            evals = [
                (Path(dirpath) / name, match.group(1), int(match.group(1)))
                for name in sorted(filenames)
                if (match := EVAL_RETURNS_RE.match(name))
            ]
            if not evals:
                continue
            run_dir = find_run_dir(Path(dirpath), root)
            if run_dir is None:
                log.warning(
                    f"{dirpath} holds eval returns but no config.yaml at or above "
                    f"it (up to {root}); skipped, since map and algorithm are "
                    "unknowable"
                )
                continue
            found.setdefault(run_dir, {})[Path(dirpath)] = evals

    tasks = []
    for run_dir in sorted(found):
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        if not config.get("STORE_EVAL_RETURNS", False):
            log.warning(
                f"{run_dir} has eval returns but STORE_EVAL_RETURNS is unset in "
                "its config; using them anyway"
            )
        if not config.get("WANDB_RUN_ID"):
            # plotting/figure_3.ipynb groups seeds by WANDB_RUN_ID; if it is blank
            # every such run collapses into one group and the figure silently shows
            # a single point per algorithm instead of one per seed.
            log.warning(
                f"{run_dir} has no WANDB_RUN_ID — downstream code that groups runs "
                "by it will merge this run with any other that is also missing it"
            )
        # A sidecar means a post-hoc evaluation, which knows its own identity.
        # Without one the returns came from training, and identity is derived from
        # the run config — including VMAPPED_SEED by position, which is only
        # meaningful among the training directories, not across post-hoc ones.
        eval_dirs = sorted(found[run_dir])
        metas = {d: read_eval_meta(d) for d in eval_dirs}
        training_dirs = [d for d in eval_dirs if metas[d] is None]
        seed_index = {d: i for i, d in enumerate(training_dirs)}

        if int(config.get("NUM_SEEDS", 1) or 1) > 1 and len(training_dirs) > 1:
            # With NUM_SEEDS=1 position gives 0 and is exact.
            log.warning(
                f"{run_dir} has NUM_SEEDS>1: VMAPPED_SEED is the position in the "
                "sorted eval-directory listing, which may not be the vmap order"
            )

        for eval_dir in eval_dirs:
            meta = metas[eval_dir]
            for path, checkpoint_id, step in found[run_dir][eval_dir]:
                tasks.append({
                    "map_name": (config.get("ENV_KWARGS") or {}).get("map_name", "unknown"),
                    "algorithm": meta["algorithm"] if meta else config.get("ALG_NAME", "unknown"),
                    "checkpoint_dir_name": str(run_dir),
                    "WANDB_RUN_ID": config.get("WANDB_RUN_ID", ""),
                    "checkpoint_id": checkpoint_id,
                    "VMAPPED_SEED": meta["vmapped_seed"] if meta else seed_index[eval_dir],
                    # Training's eval draws come from the training RNG, so no fixed
                    # eval seed exists and the column stays empty for those.
                    "eval_seed": meta["eval_seed"] if meta else "",
                    "checkpoint_step": step,
                    "vmapped_seed_key": meta["vmapped_seed_key"] if meta else eval_dir.name,
                    "sampled_seed": config.get("SEED", ""),
                    "source_file": str(path),
                    "wanted_episodes": meta["num_episodes"] if meta
                                       else expected_episodes(config),
                })
    return tasks


def process(task, n_resamples, bootstrap_seed):
    """Statistics for one stored eval. The unit of parallel work.

    Returns `(row_or_None, warnings)`; runs in a worker, so it reports problems as
    strings rather than raising something unpicklable.
    """
    path = Path(task["source_file"])
    try:
        returns = read_eval_returns(path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return None, [f"could not read {path}: {exc}"]
    if len(returns) == 0:
        return None, [f"{path} holds no returns, skipped"]

    warnings = []
    if task["wanted_episodes"] not in (None, len(returns)):
        warnings.append(
            f"{path} holds {len(returns)} returns, config implies "
            f"{task['wanted_episodes']} (truncated file, or the eval was not the "
            "size you think)"
        )

    rng = None
    if bootstrap_seed is not None:
        # Reproducible per checkpoint, and independent of worker scheduling.
        rng = np.random.default_rng(
            [bootstrap_seed, task["VMAPPED_SEED"], task["checkpoint_step"]]
        )
    stats = episode_return_stats(returns, n_resamples, rng)

    if not (np.isfinite(stats["lower_ci"]) and np.isfinite(stats["upper_ci"])):
        # BCa returns NaN bounds on a degenerate sample rather than raising, and a
        # silent NaN in a results table is worse than a loud warning.
        warnings.append(
            f"{path} produced a non-finite {BOOTSTRAP_METHOD} interval "
            f"({stats['lower_ci']}, {stats['upper_ci']}); sd is {np.std(returns):.6g}"
        )

    row = {key: task[key] for key in COLUMNS if key in task}
    row.update(stats, num_episodes=len(returns), n_resamples=n_resamples,
               bootstrap_seed="" if bootstrap_seed is None else bootstrap_seed)
    return row, warnings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add = parser.add_argument
    add("roots", nargs="*", type=Path, default=[DEFAULT_ROOT],
        help=f"run directories, or anything above them (default: {DEFAULT_ROOT}/, "
             "where hydra puts them)")
    add("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"destination for inference_results.csv and the per-map/alg splits "
             f"(default: {DEFAULT_OUTPUT}/)")
    add("--workers", type=int, default=1, help="worker processes (default: 1)")
    add("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
        help=f"bootstrap resamples (default: {DEFAULT_N_RESAMPLES}; scipy's own "
             "default is 9999)")
    add("--bootstrap-seed", type=int, default=0,
        help="seed for reproducible CI bounds; -1 leaves it unseeded (default: 0)")
    add("--resume", action="store_true",
        help="append to an existing inference_results.csv, skipping rows already in it")
    add("--dry-run", action="store_true",
        help="report what was discovered, compute nothing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)-7s %(message)s")
    bootstrap_seed = None if args.bootstrap_seed < 0 else args.bootstrap_seed

    tasks = discover_tasks(args.roots)
    if not tasks:
        log.error(
            f"no eval_returns_*.csv under {', '.join(map(str, args.roots))} — was "
            "the run trained with STORE_EVAL_RETURNS: True?"
        )
        return 1
    cells = {(task["checkpoint_dir_name"], task["VMAPPED_SEED"]) for task in tasks}
    log.info(f"found {len(tasks)} stored evals across {len(cells)} run/seed cells")

    if args.dry_run:
        for cell in sorted(cells):
            in_cell = [t for t in tasks if (t["checkpoint_dir_name"], t["VMAPPED_SEED"]) == cell]
            first = in_cell[0]
            log.info(
                f"{first['map_name']:<14} {first['algorithm']:<10} "
                f"seed[{first['VMAPPED_SEED']}]={first['vmapped_seed_key']}  "
                f"{len(in_cell):>3} evals  {first['checkpoint_dir_name']}"
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "inference_results.csv"

    append = args.resume and combined.is_file() and combined.stat().st_size > 0
    if append:
        with open(combined, newline="") as fh:
            done = {tuple(row[key] for key in KEY_COLUMNS) for row in csv.DictReader(fh)}
        before = len(tasks)
        tasks = [t for t in tasks
                 if tuple(str(t[key]) for key in KEY_COLUMNS) not in done]
        log.info(f"resuming: {before - len(tasks)} already in {combined}")
        if not tasks:
            split_combined(combined, args.output_dir)
            return 0

    started, written, total = time.time(), 0, len(tasks)
    handle = open(combined, "a" if append else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
    if not append:
        writer.writeheader()
    handle.flush()
    log.info(f"bootstrapping {total} checkpoints on {args.workers} workers")

    def consume(row, warnings):
        nonlocal written
        for warning in warnings:
            log.warning(warning)
        if row is None:
            return
        writer.writerow(row)
        handle.flush()  # a killed run keeps every checkpoint already finished
        written += 1
        if written % 100 == 0 or written == total:
            rate = written / max(time.time() - started, 1e-9)
            log.info(f"{written}/{total} rows | {time.time() - started:.0f}s elapsed "
                     f"| ~{(total - written) / rate:.0f}s left")

    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process, t, args.n_resamples, bootstrap_seed)
                           for t in tasks]
                for future in as_completed(futures):
                    consume(*future.result())
        else:
            for task in tasks:
                consume(*process(task, args.n_resamples, bootstrap_seed))
    finally:
        handle.close()

    log.info(f"wrote {written} rows to {combined} in {time.time() - started:.0f}s")
    split_combined(combined, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
