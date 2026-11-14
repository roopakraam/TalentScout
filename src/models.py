"""Prediction models for RQ1 (plan section 6.1).

Every builder returns an unfitted estimator with a scikit-learn interface.
Class imbalance is handled with class weights everywhere. Thread counts and
seeds are fixed so that a rerun reproduces every number.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

MODEL_LABEL = {
    "potential": "EA potential (baseline)",
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
}


def class_ratio(y) -> float:
    """Negatives divided by positives: the weight that balances the two classes."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def make_logreg(C: float = 1.0, seed: int = config.SEED) -> Pipeline:
    """L2 logistic regression. Imputation and scaling are learned on the training data only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def make_rf(n_estimators: int = 500, max_depth=None, min_samples_leaf: int = 1,
            max_features="sqrt", seed: int = config.SEED, n_jobs: int = config.N_JOBS) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=max_features, class_weight="balanced", n_jobs=n_jobs, random_state=seed)),
    ])


def make_lgbm(scale_pos_weight: float, seed: int = config.SEED, n_jobs: int = config.N_JOBS,
              **params) -> LGBMClassifier:
    """LightGBM classifier; missing values are handled by the trees themselves.

    ``deterministic`` and ``force_row_wise`` make the result independent of
    LightGBM's automatic choice of histogram algorithm.
    """
    return LGBMClassifier(
        scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=n_jobs,
        deterministic=True, force_row_wise=True, verbose=-1, **params)


def build(name: str, params: dict | None = None, scale_pos_weight: float | None = None,
          seed: int = config.SEED):
    """Unfitted model by name with the given hyperparameters."""
    params = dict(params or {})
    if name == "logreg":
        return make_logreg(seed=seed, **params)
    if name == "rf":
        return make_rf(seed=seed, **params)
    if name == "lgbm":
        if scale_pos_weight is None:
            raise ValueError("LightGBM needs scale_pos_weight")
        if "subsample" in params:
            params.setdefault("subsample_freq", 1)      # bagging is off unless a frequency is set
        return make_lgbm(scale_pos_weight, seed=seed, **params)
    if name == "mlp":
        from .mlp import KerasMLP                        # imported lazily: TensorFlow is slow to load
        return KerasMLP(seed=seed, **params)
    raise ValueError(f"Unknown model: {name}")


def fit(name: str, params: dict | None, X_train, y_train, X_val=None, y_val=None,
        seed: int = config.SEED):
    """Build and fit a model. The MLP also needs validation data for early stopping."""
    model = build(name, params, scale_pos_weight=class_ratio(y_train), seed=seed)
    if name == "mlp":
        return model.fit(X_train, y_train, X_val, y_val)
    return model.fit(X_train, y_train)


def scores(model, X) -> np.ndarray:
    """Predicted probability of the positive class."""
    return np.asarray(model.predict_proba(X))[:, 1]
