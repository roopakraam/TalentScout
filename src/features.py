"""Candidate feature pool for the feature-selection study.

``FEATURES`` is the frozen, ordered list of attributes the optimisers choose
from. Bit ``j`` of a solution vector refers to ``FEATURES[j]``, so the order
must never change once experiments have started. ``D = len(FEATURES)`` sets
the size of the search space (2**D subsets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- the pool
RATING = ["overall"]                                   # current rating, known at version t

BASIC = [
    "age", "height_cm", "weight_kg",
    "preferred_foot",                                  # Left = 0, Right = 1
    "weak_foot", "skill_moves", "league_level",
]

ENGINEERED = [
    "position_group",                                  # DEF = 0, MID = 1, FWD = 2
    "overall_change",                                  # overall(t) - overall(t-1); NaN on first appearance
]

FACE_STATS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

ATTACKING = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
]
SKILL = [
    "skill_dribbling", "skill_curve", "skill_fk_accuracy",
    "skill_long_passing", "skill_ball_control",
]
MOVEMENT = [
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
]
POWER = [
    "power_shot_power", "power_jumping", "power_stamina",
    "power_strength", "power_long_shots",
]
MENTALITY = [
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
]
DEFENDING = [
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
]
DETAILED = ATTACKING + SKILL + MOVEMENT + POWER + MENTALITY + DEFENDING

FEATURES: list[str] = RATING + BASIC + ENGINEERED + FACE_STATS + DETAILED
D: int = len(FEATURES)

FEATURE_GROUPS: dict[str, list[str]] = {
    "Rating": RATING,
    "Profile": BASIC + ENGINEERED,
    "Face stats": FACE_STATS,
    "Attacking": ATTACKING,
    "Skill": SKILL,
    "Movement": MOVEMENT,
    "Power": POWER,
    "Mentality": MENTALITY,
    "Defending": DEFENDING,
}
GROUP_OF: dict[str, str] = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

# --------------------------------------------------------------------------- left out on purpose
# Known at version t but deliberately kept out of the pool (plan section 5): they
# follow from how good a player already is, or are EA's own forecast of the label.
SET_ASIDE = ["potential", "value_eur", "wage_eur", "release_clause_eur", "international_reputation"]

# Dropped as leakage or as unusable text / identifiers.
POSITION_RATING_COLS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm",
    "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk",
]
GOALKEEPING_COLS = [
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed",
]
IDENTIFIER_COLS = [
    "player_url", "fifa_update", "update_as_of", "dob",
    "club_team_id", "league_id", "club_position", "club_jersey_number", "club_loaned_from",
    "club_joined_date", "club_contract_valid_until_year", "nationality_id", "nation_team_id",
    "nation_position", "nation_jersey_number", "work_rate", "body_type", "real_face",
    "player_tags", "player_traits",
]
DROPPED = IDENTIFIER_COLS + POSITION_RATING_COLS + GOALKEEPING_COLS

# Kept in the processed table for bookkeeping and case studies, never used as model input.
META_COLS = [
    "player_id", "fifa_version", "short_name", "long_name", "player_positions",
    "club_name", "league_name", "nationality_name",
]

# --------------------------------------------------------------------------- encoders
_POSITION_GROUP = {
    "CB": 0, "LB": 0, "RB": 0, "LWB": 0, "RWB": 0,                 # defenders
    "CDM": 1, "CM": 1, "CAM": 1, "LM": 1, "RM": 1,                 # midfielders
    "LW": 2, "RW": 2, "CF": 2, "ST": 2, "LF": 2, "RF": 2,          # forwards
}
POSITION_GROUP_NAMES = {0: "DEF", 1: "MID", 2: "FWD"}


def first_position(player_positions: pd.Series) -> pd.Series:
    """First listed position, e.g. ``"ST, LW"`` -> ``"ST"``."""
    return player_positions.str.split(",").str[0].str.strip()


def encode_position_group(player_positions: pd.Series) -> pd.Series:
    """Ordinal position group from the first listed position (DEF 0, MID 1, FWD 2)."""
    codes = first_position(player_positions).map(_POSITION_GROUP)
    if codes.isna().any():
        unknown = sorted(first_position(player_positions)[codes.isna()].unique())
        raise ValueError(f"Unmapped positions: {unknown}")
    return codes.astype("int8")


def encode_preferred_foot(preferred_foot: pd.Series) -> pd.Series:
    """Left = 0, Right = 1."""
    codes = preferred_foot.map({"Left": 0, "Right": 1})
    if codes.isna().any():
        raise ValueError("Unexpected value in preferred_foot")
    return codes.astype("int8")


def pretty(feature: str) -> str:
    """Readable label for plots and tables, e.g. ``movement_sprint_speed`` -> ``Sprint speed``."""
    special = {
        "overall": "Overall (current)", "age": "Age", "height_cm": "Height", "weight_kg": "Weight",
        "preferred_foot": "Preferred foot", "weak_foot": "Weak foot", "skill_moves": "Skill moves",
        "league_level": "League level", "position_group": "Position group",
        "overall_change": "Overall change (1 yr)", "physic": "Physic (face)",
        "pace": "Pace (face)", "shooting": "Shooting (face)", "passing": "Passing (face)",
        "dribbling": "Dribbling (face)", "defending": "Defending (face)",
        "skill_fk_accuracy": "FK accuracy", "defending_marking_awareness": "Marking awareness",
    }
    if feature in special:
        return special[feature]
    for prefix in ("attacking_", "skill_", "movement_", "power_", "mentality_", "defending_"):
        if feature.startswith(prefix):
            return feature[len(prefix):].replace("_", " ").capitalize()
    return feature.replace("_", " ").capitalize()


def mask_to_features(mask) -> list[str]:
    """Names of the features switched on in a 0/1 solution vector."""
    mask = np.asarray(mask).astype(bool)
    if mask.shape != (D,):
        raise ValueError(f"Expected a mask of length {D}, got shape {mask.shape}")
    return [f for f, keep in zip(FEATURES, mask) if keep]


def features_to_mask(features) -> np.ndarray:
    """0/1 vector of length D for a list of feature names."""
    wanted = set(features)
    unknown = wanted - set(FEATURES)
    if unknown:
        raise ValueError(f"Not in the candidate pool: {sorted(unknown)}")
    return np.array([int(f in wanted) for f in FEATURES], dtype=np.uint8)
