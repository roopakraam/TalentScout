"""Run the Optuna studies from a terminal (they take a while, so not from a notebook).

    python scripts/run_hpo.py comparison   # T2.6: LightGBM, TPE vs random search, 50 trials x 5 seeds each
    python scripts/run_hpo.py models       # T3.1-T3.2: logistic regression, random forest and MLP, TPE, seed 0
    python scripts/run_hpo.py all

Each study writes results/hpo/<model>_<sampler>_seed<seed>.csv and is skipped
if that file exists, so the script can be stopped and restarted.
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


def study_path(model: str, sampler: str, seed: int) -> Path:
    return config.HPO_DIR / f"{model}_{sampler}_seed{seed}.csv"


def run_job(job):
    from src import data, tuning

    model, sampler, seed = job
    df = data.load_main()
    X_train, y_train = data.get_xy(df, "train")
    X_val, y_val = data.get_xy(df, "val")
    start = time.perf_counter()
    trials = tuning.run_study(model, sampler, seed, X_train, y_train, X_val, y_val, study_path(*job))
    return job, float(trials["value"].max()), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["comparison", "models", "all"])
    parser.add_argument("--workers", type=int, default=3, help="studies run in parallel (each fit uses 4 threads)")
    args = parser.parse_args()

    jobs = {"comparison": COMPARISON, "models": MODELS, "all": COMPARISON + MODELS}[args.what]
    todo = [j for j in jobs if not study_path(*j).exists()]
    print(f"{len(jobs)} studies requested, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    config.ensure_dirs()

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_job, job) for job in todo]
        for future in as_completed(futures):
            (model, sampler, seed), best, seconds = future.result()
            print(f"  {model:7s} {sampler:6s} seed {seed}: best validation PR-AUC {best:.4f}  ({seconds / 60:.1f} min)",
                  flush=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
