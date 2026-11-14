"""Hyperparameter optimisation with Optuna (plan section 6.2).

Bayesian optimisation is Optuna's TPE sampler; the control is random search
with the same number of trials. Every study maximises validation PR-AUC and
writes one CSV with one row per trial, so a finished study is never rerun.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from . import config, models
from .metrics import pr_auc

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}


# --------------------------------------------------------------------------- search spaces
def suggest_lgbm(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def suggest_logreg(trial: optuna.Trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}


def suggest_rf(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }


def suggest_mlp(trial: optuna.Trial) -> dict:
    return {
        "hidden": trial.suggest_int("hidden", 32, 256),
        "n_layers": trial.suggest_int("n_layers", 2, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


SPACES = {"lgbm": suggest_lgbm, "logreg": suggest_logreg, "rf": suggest_rf, "mlp": suggest_mlp}


# --------------------------------------------------------------------------- studies
def make_objective(model_name: str, X_train, y_train, X_val, y_val, seed: int = config.SEED):
    """Objective: train on train, return validation PR-AUC."""
    suggest = SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        model = models.fit(model_name, params, X_train, y_train, X_val, y_val, seed=seed)
        if model_name == "mlp":
            trial.set_user_attr("best_epoch", int(model.best_epoch_))
        return pr_auc(y_val, models.scores(model, X_val))

    return objective


def run_study(model_name: str, sampler: str, seed: int, X_train, y_train, X_val, y_val,
              out_csv: Path | str, n_trials: int = config.HPO_TRIALS, model_seed: int = config.SEED,
              force: bool = False) -> pd.DataFrame:
    """Run one study and save its trials; load the CSV instead if it already exists.

    ``seed`` seeds the sampler. The model seed stays fixed, so two studies
    differ only in how hyperparameters were proposed.
    """
    out_csv = Path(out_csv)
    if out_csv.exists() and not force:
        return pd.read_csv(out_csv)

    study = optuna.create_study(direction="maximize", sampler=SAMPLERS[sampler](seed))
    start = time.perf_counter()
    study.optimize(make_objective(model_name, X_train, y_train, X_val, y_val, model_seed), n_trials=n_trials)
    elapsed = time.perf_counter() - start

    rows = []
    for t in study.trials:
        row = {"trial": t.number, "value": t.value, "seconds": (t.datetime_complete - t.datetime_start).total_seconds()}
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update({f"attr_{k}": v for k, v in t.user_attrs.items()})
        rows.append(row)
    trials = pd.DataFrame(rows)
    trials["best_so_far"] = np.maximum.accumulate(trials["value"].to_numpy())
    trials.attrs["seconds"] = elapsed

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp")
    trials.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    return trials


def best_params(trials: pd.DataFrame) -> dict:
    """Hyperparameters of the best trial of a study (first one on ties)."""
    best = trials["value"].idxmax()
    params = {}
    for col in trials.columns:
        if col.startswith("param_"):
            value = trials.at[best, col]
            is_int = pd.api.types.is_integer_dtype(trials[col])
            params[col[len("param_"):]] = int(value) if is_int else float(value)
    return params


def save_params(params: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(params, indent=2))


def load_params(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
