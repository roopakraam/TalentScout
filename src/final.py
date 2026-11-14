"""The five final configurations, fixed from validation results only (T6.1, T6.2).

Rules, written down before the test set was used (decision log D30):

* per optimizer (GA, Binary PSO, Binary GWO), the subset of the run with the best
  validation fitness at alpha = 0.99 (ties go to the lower seed);
* SHAP top-k: the k features with the largest mean |SHAP| of the tuned all-feature
  LightGBM on the **validation** set, with k = the median size of the three subsets above;
* all 45 features.

``python scripts/run_hpo.py final`` and notebook 05 both call ``final_feature_sets``,
so they cannot disagree about which features a configuration uses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, models, tuning
from .features import FEATURES

OPTIMIZERS = ("ga", "bpso", "bgwo")
CONFIGS = ("all", "ga", "bpso", "bgwo", "shap_top_k")
CONFIG_LABEL = {
    "all": "All 45 features",
    "ga": "GA subset",
    "bpso": "Binary PSO subset",
    "bgwo": "Binary GWO subset",
    "shap_top_k": "SHAP top-k",
}
FEATURE_SETS_FILE = config.FINAL_DIR / "feature_sets.json"
SHAP_RANKING_FILE = config.FINAL_DIR / "shap_ranking_val.csv"


def best_runs(fs_dir: Path = config.FS_DIR) -> pd.DataFrame:
    """The run with the lowest validation fitness for each optimizer."""
    rows = []
    for alg in OPTIMIZERS:
        summaries = [json.loads(p.read_text()) for p in sorted(fs_dir.glob(f"{alg}_seed*.json"))]
        summaries.sort(key=lambda s: (s["best_fitness"], s["seed"]))
        rows.append(summaries[0])
    return pd.DataFrame(rows).set_index("algorithm")


def tuned_all_feature_model(df: pd.DataFrame | None = None):
    """The RQ1 LightGBM: all features, best trial of the seed-0 TPE study, fitted on train."""
    df = data.load_main() if df is None else df
    X_train, y_train = data.get_xy(df, "train")
    params = tuning.best_params(pd.read_csv(config.HPO_DIR / "lgbm_tpe_seed0.csv"))
    return models.fit("lgbm", params, X_train, y_train)


def shap_ranking(model=None, df: pd.DataFrame | None = None) -> pd.Series:
    """Mean |SHAP value| per feature on the validation set, largest first."""
    import shap

    df = data.load_main() if df is None else df
    model = tuned_all_feature_model(df) if model is None else model
    X_val, _ = data.get_xy(df, "val")
    values = shap.TreeExplainer(model).shap_values(X_val)
    if isinstance(values, list):                 # older shap versions: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    ranking = pd.Series(np.abs(values).mean(axis=0), index=FEATURES, name="mean_abs_shap")
    return ranking.sort_values(ascending=False)


def final_feature_sets(recompute: bool = False) -> dict:
    """Feature lists of the five configurations, cached in results/final/feature_sets.json."""
    if FEATURE_SETS_FILE.exists() and not recompute:
        return json.loads(FEATURE_SETS_FILE.read_text())

    config.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    chosen = best_runs()
    ranking = shap_ranking()
    ranking.to_csv(SHAP_RANKING_FILE)

    sizes = [len(chosen.loc[alg, "best_features"]) for alg in OPTIMIZERS]
    k = int(np.median(sizes))
    sets = {
        "all": list(FEATURES),
        **{alg: list(chosen.loc[alg, "best_features"]) for alg in OPTIMIZERS},
        "shap_top_k": list(ranking.index[:k]),
        "_meta": {
            "rule": "per optimizer: run with the best validation fitness (alpha 0.99); "
                    "SHAP top-k: k = median size of the three subsets, ranking from validation SHAP",
            "chosen_seeds": {alg: int(chosen.loc[alg, "seed"]) for alg in OPTIMIZERS},
            "chosen_fitness": {alg: float(chosen.loc[alg, "best_fitness"]) for alg in OPTIMIZERS},
            "k": k,
        },
    }
    FEATURE_SETS_FILE.write_text(json.dumps(sets, indent=2))
    return sets
