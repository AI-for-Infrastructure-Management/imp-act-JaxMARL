"""Files of the Zenodo release, downloaded when they are missing.

Standard library only, so notebooks can fetch their data before the heavy imports.
"""

import csv
import os
import tarfile
import urllib.request
from pathlib import Path

# Anonymous Zenodo record; also set in plotting/create_figures.py
ZENODO_FILES_URL = "https://zenodo.org/records/24175731/files"

# Relative to the working directory, so printed paths do not reveal the local file system
EVALUATION_DIR = Path(os.path.relpath(Path(__file__).resolve().parent))
DATA_DIR = EVALUATION_DIR / "plotting" / "data"
CHECKPOINT_DIR = EVALUATION_DIR / "data" / "zenodo_checkpoints"


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
            r for r in csv.DictReader(f)
            if r["map_name"] == map_name and r["algorithm"] == algorithm
            and r["rank"] and int(float(r["rank"])) == rank
        )
    folder = CHECKPOINT_DIR / row["release_folder"]
    if not folder.exists():
        archive = CHECKPOINT_DIR / row["release_archive"]
        if not archive.exists():
            download(row["release_archive"], archive)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name.split("/")[0] == row["release_folder"]]
            tar.extractall(CHECKPOINT_DIR, members=members, filter="data")
    return row, folder
