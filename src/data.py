"""Raw CSV -> labelled table.

The steps follow plan section 5 and are applied in this order:

1. keep the first update of each game version (launch ratings);
2. look up every player's overall rating one, two and three versions ahead;
3. remove goalkeepers and the goalkeeping columns;
4. keep eligible rows (age <= 21, overall < 80, FIFA 15 to 21) and build the label;
5. add the two engineered features and encode the two categorical ones.

``build_labelled_table`` runs all of them; notebook 01 walks through each step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import (
    FEATURES, GOALKEEPING_COLS, META_COLS, SET_ASIDE,
    encode_position_group, encode_preferred_foot, first_position,
)

KEYS = ["player_id", "fifa_version"]
FUTURE_COLS = [f"overall_t{k}" for k in range(1, config.HORIZON + 1)]


# --------------------------------------------------------------------------- loading
def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read ``male_players.csv`` (about 180,000 rows x 109 columns)."""
    df = pd.read_csv(path, low_memory=False)
    df["fifa_version"] = df["fifa_version"].astype(int)
    df["fifa_update"] = df["fifa_update"].astype(int)
    return df


# --------------------------------------------------------------------------- step 1
def keep_launch_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first update of each version: one row per player per version."""
    first = df.groupby("fifa_version")["fifa_update"].transform("min")
    out = df[df["fifa_update"] == first]
    out = out.drop_duplicates(KEYS, keep="first")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- step 2
def add_future_overall(df: pd.DataFrame, horizon: int = config.HORIZON) -> pd.DataFrame:
    """Add ``overall_t1..overall_t{horizon}``, ``future_max`` and ``n_future``.

    ``overall_tk`` is the player's overall rating k versions later, or NaN if
    he is not in that version (or that version does not exist yet).
    """
    out = df.copy()
    lookup = df[KEYS + ["overall"]]
    for k in range(1, horizon + 1):
        ahead = lookup.assign(fifa_version=lookup["fifa_version"] - k)
        ahead = ahead.rename(columns={"overall": f"overall_t{k}"})
        out = out.merge(ahead, on=KEYS, how="left", validate="one_to_one")
    cols = [f"overall_t{k}" for k in range(1, horizon + 1)]
    out["future_max"] = out[cols].max(axis=1, skipna=True)
    out["n_future"] = out[cols].notna().sum(axis=1).astype("int8")
    return out


def add_overall_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``overall_change`` = overall(t) - overall(t-1); NaN if absent at t-1."""
    previous = df[KEYS + ["overall"]].assign(fifa_version=df["fifa_version"] + 1)
    previous = previous.rename(columns={"overall": "overall_prev"})
    out = df.merge(previous, on=KEYS, how="left", validate="one_to_one")
    out["overall_change"] = out["overall"] - out["overall_prev"]
    return out.drop(columns="overall_prev")


# --------------------------------------------------------------------------- step 3
def is_goalkeeper(df: pd.DataFrame) -> pd.Series:
    return first_position(df["player_positions"]).eq("GK")


def drop_goalkeepers(df: pd.DataFrame) -> pd.DataFrame:
    """Outfield players only; the goalkeeping_* columns go as well."""
    out = df[~is_goalkeeper(df)]
    return out.drop(columns=[c for c in GOALKEEPING_COLS if c in out.columns]).reset_index(drop=True)


# --------------------------------------------------------------------------- step 4
def eligible_mask(df: pd.DataFrame, threshold: int = config.ELITE_THRESHOLD,
                  max_age: int = config.MAX_AGE) -> pd.Series:
    """Young, not yet elite, and with a complete three-version label window."""
    lo, hi = config.FEATURE_VERSIONS
    return (df["age"] <= max_age) & (df["overall"] < threshold) & df["fifa_version"].between(lo, hi)


def add_label(df: pd.DataFrame, threshold: int = config.ELITE_THRESHOLD, name: str = "y") -> pd.DataFrame:
    """``name`` = 1 if the best future overall reaches ``threshold``, 0 if not, NaN if unknown.

    The label is unknown when the player is missing from all future versions
    in the window (``missing_future``).
    """
    out = df.copy()
    out["missing_future"] = out["n_future"].eq(0)
    label = (out["future_max"] >= threshold).astype("float")
    out[name] = label.where(~out["missing_future"])
    return out


# --------------------------------------------------------------------------- step 5
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["position_group"] = encode_position_group(out["player_positions"])
    out["preferred_foot"] = encode_preferred_foot(out["preferred_foot"])
    return out


# --------------------------------------------------------------------------- full pipeline
def build_labelled_table(raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Run steps 1-5 and return (labelled table, dictionary of row counts).

    The table holds every eligible row, including those whose label is unknown
    (``missing_future``); the main analysis drops them, the sensitivity check
    in notebook 05 counts them as negatives.
    """
    counts: dict[str, int] = {}
    raw = load_raw() if raw is None else raw
    counts["rows_raw"] = len(raw)

    snap = keep_launch_snapshot(raw)
    counts["rows_snapshot"] = len(snap)

    snap = add_future_overall(snap)
    snap = add_overall_change(snap)

    counts["rows_goalkeeper"] = int(is_goalkeeper(snap).sum())
    outfield = drop_goalkeepers(snap)
    counts["rows_outfield"] = len(outfield)

    eligible = outfield[eligible_mask(outfield)].copy()
    counts["rows_eligible"] = len(eligible)

    eligible = add_label(eligible, config.ELITE_THRESHOLD, "y")
    second = (eligible["future_max"] >= config.SECOND_THRESHOLD).astype("float")
    eligible[f"y{config.SECOND_THRESHOLD}"] = second.where(~eligible["missing_future"])
    counts["rows_missing_future"] = int(eligible["missing_future"].sum())
    counts["rows_labelled"] = int((~eligible["missing_future"]).sum())
    counts["positives"] = int(eligible["y"].sum())

    eligible = encode_features(eligible)

    keep = META_COLS + FEATURES + SET_ASIDE + FUTURE_COLS + [
        "future_max", "n_future", "missing_future", "y", f"y{config.SECOND_THRESHOLD}",
    ]
    table = eligible[keep].sort_values(KEYS).reset_index(drop=True)
    return table, counts


# --------------------------------------------------------------------------- access for later notebooks
def load_labelled(path=config.LABELLED) -> pd.DataFrame:
    """The frozen processed table (all eligible rows, with the ``set`` column)."""
    return pd.read_parquet(path)


def load_main(path=config.LABELLED) -> pd.DataFrame:
    """Rows used in the main analysis: known label and assigned to train / val / test."""
    df = load_labelled(path)
    return df[df["set"].isin(["train", "val", "test"])].reset_index(drop=True)


def get_xy(df: pd.DataFrame, set_name: str, features=None, label: str = "y"):
    """Feature matrix (DataFrame) and 0/1 label vector for one of train / val / test."""
    features = FEATURES if features is None else list(features)
    part = df[df["set"] == set_name]
    X = part[features].astype("float64")
    y = part[label].to_numpy().astype(int)
    return X, y
