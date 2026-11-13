"""Paths and frozen design constants for TalentScout v2.

Everything that defines the study design (label rule, split, experiment
budget) lives here so that notebooks and scripts cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

RAW_CSV = DATA_RAW / "male_players.csv"
LABELLED = DATA_PROCESSED / "labelled.parquet"
SPLIT_IDS = DATA_PROCESSED / "split_ids.csv"

HPO_DIR = RESULTS / "hpo"
FS_DIR = RESULTS / "feature_selection"
SWEEP_DIR = RESULTS / "alpha_sweep"
FINAL_DIR = RESULTS / "final"
SANITY_DIR = RESULTS / "sanity_check"

# --------------------------------------------------------------------------- label
ELITE_THRESHOLD = 80      # "elite" = overall rating of 80 or more
MAX_AGE = 21              # a row is eligible if the player is this age or younger ...
HORIZON = 3               # ... and the label looks this many game versions ahead
SECOND_THRESHOLD = 75     # second label used only for the circularity check (plan section 9)

# --------------------------------------------------------------------------- split
FIRST_VERSION, LAST_VERSION = 15, 24          # FIFA 15 ... EA Sports FC 24
FEATURE_VERSIONS = (15, 21)                   # rows that have a full 3-version label window
TRAIN_VERSIONS = (15, 18)
VAL_VERSIONS = (19, 20)
TEST_VERSIONS = (21, 21)

# --------------------------------------------------------------------------- experiments
SEED = 0
N_JOBS = 4                # threads per LightGBM / random-forest fit (fixed for reproducibility)

HPO_TRIALS = 50           # Optuna trials per study
HPO_SEEDS = (0, 1, 2, 3, 4)

POP_SIZE = 20             # population / swarm / pack size
BUDGET = 600              # fitness evaluations per optimiser run (cache hits included)
RUN_SEEDS = tuple(range(10))
ALPHA = 0.99              # weight of (1 - PR-AUC) in the fitness
SWEEP_ALPHAS = (0.95, 0.90, 0.80)
SWEEP_SEEDS = (0, 1, 2, 3, 4)

# fixed LightGBM settings used inside the fitness function (chosen for speed)
FITNESS_LGBM = dict(n_estimators=200, learning_rate=0.05)

BOOTSTRAP_RESAMPLES = 1000
SHORTLIST_K = 50          # Precision@50 / Recall@50: a scout's shortlist


def ensure_dirs() -> None:
    """Create every output directory used by the project."""
    for d in (DATA_PROCESSED, RESULTS, FIGURES, HPO_DIR, FS_DIR, SWEEP_DIR, FINAL_DIR, SANITY_DIR):
        d.mkdir(parents=True, exist_ok=True)
