"""Temporal, player-grouped train / validation / test split (plan section 5).

* Time: FIFA 15-18 train, FIFA 19-20 validation, FIFA 21 test. A scout
  predicts the future from the past, so the newest version is the test set.
* Players: every player who appears in the test set is removed from train and
  validation, so the model cannot simply recognise him.

The plan removes only test players, so a player can still be in both train
and validation. That slightly flatters validation scores; the
validation-to-test gap in notebook 05 measures it.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

SETS = ("train", "val", "test")


def version_to_set(version: int) -> str:
    if config.TRAIN_VERSIONS[0] <= version <= config.TRAIN_VERSIONS[1]:
        return "train"
    if config.VAL_VERSIONS[0] <= version <= config.VAL_VERSIONS[1]:
        return "val"
    if config.TEST_VERSIONS[0] <= version <= config.TEST_VERSIONS[1]:
        return "test"
    raise ValueError(f"Version {version} is outside the study window")


def temporal_group_split(df: pd.DataFrame) -> pd.Series:
    """Return the set of each row: ``train``, ``val``, ``test`` or ``removed``.

    ``removed`` marks train / validation rows of players who are in the test set.
    """
    by_version = df["fifa_version"].map(version_to_set)
    test_players = set(df.loc[by_version == "test", "player_id"])
    clash = by_version.isin(["train", "val"]) & df["player_id"].isin(test_players)
    return by_version.mask(clash, "removed").rename("set")


def check_split(df: pd.DataFrame, set_col: str = "set") -> None:
    """Fail loudly if the split breaks either rule."""
    test_players = set(df.loc[df[set_col] == "test", "player_id"])
    for name in ("train", "val"):
        overlap = test_players & set(df.loc[df[set_col] == name, "player_id"])
        assert not overlap, f"{len(overlap)} test players also appear in {name}"
    bounds = {"train": config.TRAIN_VERSIONS, "val": config.VAL_VERSIONS, "test": config.TEST_VERSIONS}
    for name, (lo, hi) in bounds.items():
        versions = df.loc[df[set_col] == name, "fifa_version"]
        assert versions.between(lo, hi).all(), f"{name} contains versions outside {lo}-{hi}"
    assert not df.duplicated(["player_id", "fifa_version"]).any(), "duplicate player-version rows"


def split_summary(df: pd.DataFrame, label: str = "y", set_col: str = "set") -> pd.DataFrame:
    """Rows, players, positives and positive rate per set."""
    part = df[df[set_col].isin(SETS)]
    out = part.groupby(set_col).agg(
        rows=(label, "size"),
        players=("player_id", "nunique"),
        positives=(label, "sum"),
        positive_rate=(label, "mean"),
    )
    out["positives"] = out["positives"].astype(int)
    return out.loc[[s for s in SETS if s in out.index]]


def grouped_cv(n_splits: int = 3, seed: int = config.SEED) -> StratifiedGroupKFold:
    """Cross-validation that keeps all rows of a player in the same fold.

    Use as ``grouped_cv().split(X, y, groups=player_id)``.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
