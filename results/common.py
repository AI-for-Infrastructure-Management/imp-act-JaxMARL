"""Shared code of the results: paths, the Zenodo release, the comparison protocol, names and
plot style.

Only the standard library is imported at module level (pandas is imported by the functions
that need it), and importing this module has no side effects, so notebooks can fetch their
data before the heavy imports.
"""

import csv
import functools
import os
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Relative to the working directory
RESULTS_DIR = Path(os.path.relpath(Path(__file__).resolve().parent))
DATA_DIR = RESULTS_DIR / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
# Absolute: the experiment configs are read from experiments/config
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Zenodo release: data files and checkpoints, downloaded when they are missing
# ---------------------------------------------------------------------------
# Anonymous Zenodo record
ZENODO_FILES_URL = "https://zenodo.org/records/24175731/files"


def download(name, path):
    """Download the release file `name` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from {ZENODO_FILES_URL}")
    partial = path.with_name(f"{path.name}.part")  # renamed once complete
    urllib.request.urlretrieve(f"{ZENODO_FILES_URL}/{name}?download=1", partial)
    partial.rename(path)


def data_file(name):
    """Path of a released data file (results.csv, heuristics.csv, ...) in DATA_DIR."""
    path = DATA_DIR / name
    if not path.exists():
        download(name, path)
    return path


def best_checkpoint(map_name, algorithm, rank=1):
    """The results.csv row and the run folder of the rank-`rank` checkpoint.

    The folder (config.yaml and checkpoints/<key>/checkpoint_<step>.safetensors) is
    extracted from its rankNN.tar.gz archive, which is downloaded if needed.
    """
    with open(data_file("results.csv")) as f:
        row = next(
            r
            for r in csv.DictReader(f)
            if r["map_name"] == map_name
            and r["algorithm"] == algorithm
            and r["rank"]
            and int(float(r["rank"])) == rank
        )
    folder = CHECKPOINT_DIR / row["release_folder"]
    if not folder.exists():
        archive = CHECKPOINT_DIR / row["release_archive"]
        if not archive.exists():
            download(row["release_archive"], archive)
        with tarfile.open(archive) as tar:
            members = [
                m
                for m in tar.getmembers()
                if m.name.split("/")[0] == row["release_folder"]
            ]
            tar.extractall(CHECKPOINT_DIR, members=members, filter="data")
    return row, folder


# ---------------------------------------------------------------------------
# Comparison protocol: returns relative to the prioritized segment-based heuristic H_PS
# ---------------------------------------------------------------------------
BASELINE_HEURISTIC = "segment_based_heuristic"  # H_PS


def load_results():
    """Evaluation of every checkpoint of every final run (results.csv), one row per checkpoint."""
    import pandas as pd

    return pd.read_csv(data_file("results.csv"))


@functools.cache
def heuristics():
    """Heuristic results (heuristics.csv), indexed by (map_name, heuristic)."""
    import pandas as pd

    return pd.read_csv(data_file("heuristics.csv")).set_index(["map_name", "heuristic"])


def baseline_values():
    """Expected return of H_PS per map_name."""
    return (
        heuristics().xs(BASELINE_HEURISTIC, level="heuristic")["mean_reward"].to_dict()
    )


def normalize(value, baseline):
    """Return as % over the baseline return: (value - baseline) / |baseline| * 100."""
    return (value - baseline) / abs(baseline) * 100


def denormalize(pct, baseline):
    """Inverse of normalize: the return that is `pct` % over the baseline return."""
    return baseline + pct / 100 * abs(baseline)


def best_per_run(data, map_name):
    """Best checkpoint (highest mean return) of every final run on `map_name`.

    One row per (algorithm, run), in the order of `data`, with mean_pct, lower_ci_pct and
    upper_ci_pct: the checkpoint's mean return and 95% CI as % over H_PS.
    """
    env_data = data[data["map_name"] == map_name]
    best = env_data.loc[
        env_data.groupby(["algorithm", "run"], sort=False)["mean"].idxmax()
    ]
    baseline = baseline_values()[map_name]
    for column in ("mean", "lower_ci", "upper_ci"):
        best[f"{column}_pct"] = normalize(best[column], baseline)
    return best


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
# Paper order of the algorithms
ALGORITHMS = ["vdn_rnn", "vdn_ba_rnn", "qmix_rnn", "pqn_rnn", "mappo_rnn", "ippo_rnn"]
ALGORITHM_NAMES = {
    "vdn_rnn": "VDN",
    "qmix_rnn": "QMIX",
    "pqn_rnn": "PQN-VDN",
    "ippo_rnn": "IPPO",
    "mappo_rnn": "MAPPO",
    "vdn_ba_rnn": "VDN-BA",
}

# Budgets of the constrained scenarios from the imp-act presets (budget_amount)
ENV_BUDGETS = {
    "ToyExample-v2": "6.5M",
    "Cologne-v1": "200M",
    "CologneBonnDusseldorf-v1": "800M",
}


def tt(map_name):
    """Map name in typewriter font, as \\texttt in the paper."""
    return rf"$\mathtt{{{map_name}}}$"


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
PLOT_PARAMS = {
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.usetex": False,
    # Math in Times, and \mathtt in LaTeX's typewriter font (the default DejaVu Sans Mono is
    # wider and heavier than the surrounding Times text)
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.tt": "CMU Typewriter Text",
}

# Y-axis label of the relative performance plots
RELATIVE_PERFORMANCE_LABEL = r"Normalized Expected Return ($\Delta \bar{R}^{\pi}_0$ %)"
ABSOLUTE_RETURN_LABEL = r"Expected Return ($\mathbb{E}\left[R_0^\pi\right]$)"

# Paper palette: viridis tones and orange/purple accents
DARK, TEAL, ORANGE, PURPLE = "#414487", "#22a884", "#d95f02", "#7570b3"
GOLD = "#e6ab02"  # Dark2 mustard: unused elsewhere, clearly not red
EDGE = "#3a3a3a"  # outlines and medians
