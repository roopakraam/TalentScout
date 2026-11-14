"""The objective of the feature-selection problem (plan section 6.3).

Decision variables   x in {0,1}^D, x_j = 1 keeps feature j
Constraint           at least one feature selected
Objective (minimise) fitness(x) = alpha * (1 - PR-AUC_val(x)) + (1 - alpha) * |x| / D

PR-AUC_val(x) is the validation PR-AUC of a LightGBM model with fixed,
moderate settings, trained on the training rows using only the selected
columns. With alpha = 0.99 accuracy dominates and the number of features
mostly breaks ties.

Every evaluated subset is cached. The cache stores PR-AUC rather than fitness,
so runs with another alpha can reuse it. Every call counts towards an
optimiser's budget, cache hit or not.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .metrics import pr_auc
from .models import class_ratio, make_lgbm


def mask_to_bits(mask) -> str:
    """0/1 vector -> string such as ``"0110..."`` (used in result files)."""
    return "".join("1" if b else "0" for b in np.asarray(mask).astype(bool))


def bits_to_mask(bits: str) -> np.ndarray:
    return np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")


class FitnessFunction:
    """Callable fitness with a PR-AUC cache and a per-call log."""

    def __init__(self, X_train, y_train, X_val, y_val, alpha: float = config.ALPHA,
                 lgbm_params: dict | None = None, n_jobs: int = config.N_JOBS,
                 seed: int = config.SEED, cache: dict | None = None):
        self.X_train = np.ascontiguousarray(np.asarray(X_train, dtype="float64"))
        self.X_val = np.ascontiguousarray(np.asarray(X_val, dtype="float64"))
        self.y_train = np.asarray(y_train).astype(int)
        self.y_val = np.asarray(y_val).astype(int)
        self.D = self.X_train.shape[1]
        self.alpha = float(alpha)
        self.lgbm_params = dict(config.FITNESS_LGBM if lgbm_params is None else lgbm_params)
        self.n_jobs = n_jobs
        self.seed = seed
        self.scale_pos_weight = class_ratio(self.y_train)
        self.cache: dict[bytes, float] = {} if cache is None else cache
        self._repair_rng = np.random.default_rng(seed)
        self.reset_log()

    # ----------------------------------------------------------------- bookkeeping
    def reset_log(self) -> None:
        self.n_calls = 0
        self.n_trainings = 0
        self.train_seconds = 0.0
        self.log: list[dict] = []

    # ----------------------------------------------------------------- the objective
    def value(self, pr_auc_val: float, n_selected: int) -> float:
        """Weighted sum of the two goals for a given alpha."""
        return self.alpha * (1.0 - pr_auc_val) + (1.0 - self.alpha) * n_selected / self.D

    def train_and_score(self, mask: np.ndarray) -> float:
        """Train LightGBM on the selected columns, return validation PR-AUC (no cache)."""
        cols = np.flatnonzero(mask)
        model = make_lgbm(self.scale_pos_weight, seed=self.seed, n_jobs=self.n_jobs, **self.lgbm_params)
        model.fit(self.X_train[:, cols], self.y_train)
        return pr_auc(self.y_val, model.predict_proba(self.X_val[:, cols])[:, 1])

    def pr_auc(self, mask: np.ndarray) -> tuple[float, bool]:
        """Validation PR-AUC of a subset and whether it came from the cache."""
        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key], True
        start = time.perf_counter()
        score = self.train_and_score(mask)
        self.train_seconds += time.perf_counter() - start
        self.n_trainings += 1
        self.cache[key] = score
        return score, False

    def __call__(self, mask) -> float:
        mask = np.asarray(mask).astype(np.uint8)
        if mask.shape != (self.D,):
            raise ValueError(f"Expected a 0/1 vector of length {self.D}")
        if not mask.any():                         # constraint: at least one feature
            mask = mask.copy()
            mask[self._repair_rng.integers(self.D)] = 1
        score, hit = self.pr_auc(mask)
        n_selected = int(mask.sum())
        fitness = self.value(score, n_selected)
        self.n_calls += 1
        self.log.append({"call": self.n_calls, "mask": mask_to_bits(mask), "n_features": n_selected,
                         "pr_auc": score, "fitness": fitness, "cache_hit": hit})
        return fitness

    # ----------------------------------------------------------------- persistence
    def log_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.log)
        frame["best_so_far"] = np.minimum.accumulate(frame["fitness"].to_numpy())
        return frame

    def save_cache(self, path: Path | str) -> None:
        rows = [{"mask": mask_to_bits(np.frombuffer(k, dtype=np.uint8)), "pr_auc": v} for k, v in self.cache.items()]
        pd.DataFrame(rows).to_csv(path, index=False)


def load_cache(paths) -> dict[bytes, float]:
    """Build a PR-AUC cache from the per-call logs (or cache files) of earlier runs.

    Only valid for runs that used the same data and the same fixed LightGBM
    settings, which is the case for everything under ``results/``.
    """
    cache: dict[bytes, float] = {}
    for path in paths:
        frame = pd.read_csv(path, usecols=["mask", "pr_auc"], dtype={"mask": str})
        for bits, score in zip(frame["mask"], frame["pr_auc"]):
            cache[bits_to_mask(bits).tobytes()] = float(score)
    return cache
