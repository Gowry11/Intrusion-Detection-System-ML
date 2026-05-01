"""
Utility functions for the IDS project.
"""

import os
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# NSL-KDD default file names
TRAIN_FILE = "KDDTrain+.csv"
TEST_FILE = "KDDTest+.csv"

# Alternate names for different download formats
TRAIN_ALT_NAMES = ["KDDTrain+.txt", "Train.csv", "train.csv"]
TEST_ALT_NAMES = ["KDDTest+.txt", "Test.csv", "test.csv"]

# Single combined CSV (e.g. nsl-kdd.xlsx.csv)
COMBINED_NAMES = ["nsl-kdd.xlsx.csv", "nsl-kdd.csv", "NSL-KDD.csv"]

RANDOM_STATE = 42


def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def find_dataset_file(names, directory=None, also_project_root=False):
    """Find first existing file from list of possible names."""
    dirs = [directory or DATA_DIR]
    if also_project_root:
        dirs.insert(0, PROJECT_ROOT)
    for d in dirs:
        for name in names:
            path = Path(d) / name
            if path.exists():
                return path
    return None


def save_json(data, filepath):
    """Save dictionary to JSON file."""
    ensure_dirs()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load JSON file to dictionary."""
    with open(Path(filepath)) as f:
        return json.load(f)


def get_paths():
    """Return dict of project paths."""
    return {
        "project_root": PROJECT_ROOT,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "plots": PLOTS_DIR,
        "reports": REPORTS_DIR,
    }
