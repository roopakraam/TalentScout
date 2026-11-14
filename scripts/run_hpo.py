"""Run the Optuna studies from a terminal (they take a while, so not from a notebook).

    python scripts/run_hpo.py comparison   # T2.6: LightGBM, TPE vs random search, 50 trials x 5 seeds each
    python scripts/run_hpo.py models       # T3.1-T3.2: logistic regression, random forest and MLP, TPE, seed 0
    python scripts/run_hpo.py all
    python scripts/run_hpo.py final        # T6.3: LightGBM retuned on each final feature subset, TPE, seed 0

Each study writes results/hpo/<model>_<sampler>_seed<seed>.csv (final studies:
results/hpo/final_lgbm_<subset>_tpe_seed0.csv) and is skipped if that file exists,
so the script can be stopped and restarted.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

COMPARISON = [("lgbm", sampler, seed) for seed in config.HPO_SEEDS for sampler in ("tpe", "random")]
MODELS = [("logreg", "tpe", 0), ("rf", "tpe", 0), ("mlp", "tpe", 0)]
FINAL_SUBSETS = ["ga", "bpso", "bgwo", "shap_top_k"]      # "all" reuses lgbm_tpe_seed0


def study_path(model: str, sampler: str, seed: int, subset: str | None = None) -> Path:
    if subset is not None:
        return config.HPO_DIR / f"final_{model}_{subset}_{sampler}_seed{seed}.csv"
    return config.HPO_DIR / f"{model}_{sampler}_seed{seed}.csv"


def run_job(job):
    from src import data, tuning

    model, sampler, seed, subset = (*job, None)[:4]
    df = data.load_main()
    columns = None
    if subset is not None:
        from src.final import final_feature_sets
        columns = final_feature_sets()[subset]
    X_train, y_train = data.get_xy(df, "train", columns)
    X_val, y_val = data.get_xy(df, "val", columns)
    start = time.perf_counter()
    trials = tuning.run_study(model, sampler, seed, X_train, y_train, X_val, y_val, study_path(model, sampler, seed, subset))
    return job, float(trials["value"].max()), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["comparison", "models", "all", "final"])
    parser.add_argument("--workers", type=int, default=3, help="studies run in parallel (each fit uses 4 threads)")
    args = parser.parse_args()

    if args.what == "final":
        from src.final import final_feature_sets
        sets = final_feature_sets()            # computes and saves the subsets on first use (validation only)
        print("final subsets:", {k: len(v) for k, v in sets.items() if not k.startswith("_")})
    final_jobs = [("lgbm", "tpe", 0, subset) for subset in FINAL_SUBSETS]
    jobs = {"comparison": COMPARISON, "models": MODELS, "all": COMPARISON + MODELS, "final": final_jobs}[args.what]
    todo = [j for j in jobs if not study_path(*j).exists()]
    print(f"{len(jobs)} studies requested, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    config.ensure_dirs()

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_job, job) for job in todo]
        for future in as_completed(futures):
            job, best, seconds = future.result()
            model, sampler, seed = job[:3]
            subset = f" on {job[3]}" if len(job) > 3 else ""
            print(f"  {model:7s} {sampler:6s} seed {seed}{subset}: best validation PR-AUC {best:.4f}  ({seconds / 60:.1f} min)",
                  flush=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
