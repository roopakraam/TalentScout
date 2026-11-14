"""Write report/chapters/12_appendix.md from the result files, so no number is typed by hand.

    python report/make_appendix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config, tuning  # noqa: E402
from src.features import FEATURES, GROUP_OF, pretty  # noqa: E402

OUT = ROOT / "report" / "chapters" / "12_appendix.md"
SETS = json.loads((config.FINAL_DIR / "feature_sets.json").read_text())
SET_NAMES = {"ga": "GA", "bpso": "Binary PSO", "bgwo": "Binary GWO", "shap_top_k": f"SHAP top-{SETS['_meta']['k']}"}


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    lines = ["# Appendix", "",
             "## A. The final feature subsets", "",
             "Chosen on validation by the rules in @sec:final before the test set was used. "
             f"Optimizer subsets come from the run with the best validation fitness (GA seed {SETS['_meta']['chosen_seeds']['ga']}, "
             f"Binary PSO seed {SETS['_meta']['chosen_seeds']['bpso']}, Binary GWO seed {SETS['_meta']['chosen_seeds']['bgwo']}).", ""]
    lines += ["| Configuration | Features | Attributes |", "| --- | ---: | --- |"]
    for key, name in SET_NAMES.items():
        lines.append(f"| {name} | {len(SETS[key])} | {', '.join(pretty(f) for f in SETS[key])} |")
    lines.append("")

    lines += ["## B. Tuned hyperparameters", "",
              "Best trial of each 50-trial TPE study (seed 0), maximising validation PR-AUC.", "",
              "| Model | Hyperparameters |", "| --- | --- |"]
    for name, label in (("logreg", "Logistic regression"), ("rf", "Random forest (500 trees)"), ("mlp", "MLP")):
        params = tuning.load_params(config.HPO_DIR / f"best_params_{name}.json")
        lines.append(f"| {label} | {'; '.join(f'{k} = {fmt(v)}' for k, v in params.items())} |")
    for key, label in (("all", "LightGBM, all 45 features"), ("ga", "LightGBM, GA subset"), ("bpso", "LightGBM, Binary PSO subset"),
                       ("bgwo", "LightGBM, Binary GWO subset"), ("shap_top_k", f"LightGBM, {SET_NAMES['shap_top_k']}")):
        path = config.HPO_DIR / ("lgbm_tpe_seed0.csv" if key == "all" else f"final_lgbm_{key}_tpe_seed0.csv")
        params = tuning.best_params(pd.read_csv(path))
        lines.append(f"| {label} | {'; '.join(f'{k} = {fmt(v)}' for k, v in params.items())} |")
    lines.append("")

    shap = pd.read_csv(config.RESULTS / "tables" / "shap_ranking_val.csv", index_col=0)
    counts = pd.read_csv(config.RESULTS / "tables" / "selection_frequency_vs_shap.csv", index_col=0)
    lines += ["## C. Every candidate feature", "",
              "SHAP rank of the tuned all-feature LightGBM on validation, how many of the 30 main runs kept the feature, "
              "and whether it is in each final subset.", "",
              "| Feature | Group | SHAP rank | Kept (of 30) | GA | BPSO | BGWO | SHAP top-k |",
              "| --- | --- | ---: | ---: | :---: | :---: | :---: | :---: |"]
    for f in shap.index:
        name = pretty(f)
        marks = ["✓" if f in SETS[k] else "" for k in ("ga", "bpso", "bgwo", "shap_top_k")]
        lines.append(f"| {name} | {GROUP_OF[f]} | {int(shap.loc[f, 'rank'])} | {int(counts.loc[name, 'total (of 30)'])} | "
                     + " | ".join(marks) + " |")
    lines.append("")
    assert set(shap.index) == set(FEATURES)

    lines += ["## D. Reproducing the results", "",
              "The repository README gives the full instructions. In short, with the `tf-gpu-env` environment active:", "",
              "1. `python scripts/download_data.py` downloads the Kaggle data (no account needed).",
              "2. `bash scripts/reproduce_all.sh --clean` reruns every step in order: notebook 01, the 13 tuning studies, "
              "notebooks 02 and 03, the 40 feature-selection runs with the ablation and the α sweep, notebook 04, the final "
              "tuning and the fallback runs, notebooks 05 and 06, and this report. It takes about three hours on the project laptop.",
              "3. Every run and study writes its own file and is skipped if the file exists, so the script can be stopped and restarted.",
              "",
              "Every decision that changed data, settings or results is recorded with its reason in `docs/decision_log.md`.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
