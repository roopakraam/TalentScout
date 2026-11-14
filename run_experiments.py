"""Feature-selection experiments on the football data: one result file per algorithm and seed.

    python run_experiments.py                        # main experiment: GA, BPSO, BGWO, random x seeds 0-9
    python run_experiments.py --algs ga --seeds 0    # one run, to inspect before launching the rest
    python run_experiments.py --algs bpso_w_dec      # ablation: Binary PSO with the falling inertia weight
    python run_experiments.py --algs bgwo --alpha 0.95 0.90 0.80 --seeds 0 1 2 3 4 \
        --out results/alpha_sweep --reuse-cache      # trade-off sweep (T5.7)
    python run_experiments.py --algs ga bpso bgwo --seeds 0 1 2 3 4 --fitness cv \
        --out results/fallback_cv_fitness            # plan section 9 fallback: grouped CV on train

The runner is resumable: a run whose summary file exists is skipped, so a
crash or a restart costs at most the runs that were in progress.

Each run writes two files to the output folder:

  <alg>_seed<s>.csv    one row per fitness call: the subset, its PR-AUC, fitness, best so far
  <alg>_seed<s>.json   best subset, best fitness, validation PR-AUC, number of features,
                       wall-clock time, number of model trainings, diversity per iteration

(runs with an alpha other than the default are named <alg>_alpha<a>_seed<s>).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src import config
from src.features import D, mask_to_features

MAIN_ALGS = ["ga", "bpso", "bgwo", "random"]


def run_name(alg: str, seed: int, alpha: float) -> str:
    if alpha == config.ALPHA:
        return f"{alg}_seed{seed}"
    return f"{alg}_alpha{alpha:.2f}_seed{seed}"


def run_one(alg: str, seed: int, alpha: float, out_dir: str, reuse_cache: bool, fitness_kind: str = "val") -> dict:
    import numpy as np

    from src import data
    from src.fitness import CVFitnessFunction, FitnessFunction, load_cache, mask_to_bits
    from src.optimizers import ALL_RUNNERS, init_population, run_rng

    out = Path(out_dir)
    df = data.load_main()
    X_train, y_train = data.get_xy(df, "train")
    X_val, y_val = data.get_xy(df, "val")

    if fitness_kind == "cv":
        groups = df.loc[df["set"] == "train", "player_id"].to_numpy()
        fitness = CVFitnessFunction(X_train, y_train, groups, alpha=alpha)
        preloaded = 0
    else:
        cache = load_cache(sorted(config.FS_DIR.glob("*.csv"))) if reuse_cache else None
        preloaded = len(cache) if cache else 0
        fitness = FitnessFunction(X_train, y_train, X_val, y_val, alpha=alpha, cache=cache)

    start = time.perf_counter()
    result = ALL_RUNNERS[alg](fitness, D, init_population(seed, config.POP_SIZE, D), config.BUDGET, run_rng(seed))
    wall = time.perf_counter() - start

    log = fitness.log_frame()
    log.insert(1, "iteration", result.iteration_of_call)
    assert len(log) == config.BUDGET and np.allclose(log["best_so_far"], result.history)

    best_row = log.loc[log["fitness"].idxmin()]
    summary = {
        "algorithm": alg, "seed": seed, "alpha": alpha, "budget": config.BUDGET, "fitness": fitness_kind,
        "pop_size": config.POP_SIZE, "D": D,
        "best_mask": mask_to_bits(result.best_mask),
        "best_features": mask_to_features(result.best_mask),
        "n_features": int(result.best_mask.sum()),
        "best_fitness": float(result.best_fitness),
        "best_pr_auc": float(best_row["pr_auc"]),
        "first_hit_call": int(best_row["call"]),
        "wall_seconds": wall,
        "train_seconds": fitness.train_seconds,
        "n_trainings": fitness.n_trainings,
        "n_cache_hits": config.BUDGET - fitness.n_trainings,
        "cache_preloaded": preloaded,
        "n_iterations": int(result.iteration_of_call.max()),
        "diversity": [round(float(d), 4) for d in result.diversity],
    }

    name = run_name(alg, seed, alpha)
    tmp_csv, tmp_json = out / f"{name}.csv.tmp", out / f"{name}.json.tmp"
    log.to_csv(tmp_csv, index=False)
    tmp_json.write_text(json.dumps(summary, indent=2))
    tmp_csv.replace(out / f"{name}.csv")
    tmp_json.replace(out / f"{name}.json")          # written last: its presence marks a finished run
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algs", nargs="+", default=MAIN_ALGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.RUN_SEEDS))
    parser.add_argument("--alpha", nargs="+", type=float, default=[config.ALPHA])
    parser.add_argument("--out", default=str(config.FS_DIR))
    parser.add_argument("--workers", type=int, default=4, help="runs in parallel (each model fit uses 4 threads)")
    parser.add_argument("--fitness", choices=["val", "cv"], default="val",
                        help="val: PR-AUC on the validation set (main study); cv: 3-fold grouped CV on train (fallback)")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="start from the PR-AUC values of every subset already evaluated in results/feature_selection")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # seed-major order: with 4 workers the algorithms of one seed run side by side, under the same machine load
    jobs = [(alg, seed, alpha) for alpha in args.alpha for seed in args.seeds for alg in args.algs]
    todo = [j for j in jobs if not (out / f"{run_name(*j)}.json").exists()]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already finished, {len(todo)} to do", flush=True)

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, alg, seed, alpha, str(out), args.reuse_cache, args.fitness)
                   for alg, seed, alpha in todo]
        for future in as_completed(futures):
            s = future.result()
            print(f"  {run_name(s['algorithm'], s['seed'], s['alpha']):22s} fitness {s['best_fitness']:.5f}  "
                  f"PR-AUC {s['best_pr_auc']:.4f}  features {s['n_features']:2d}  "
                  f"trainings {s['n_trainings']:3d}  {s['wall_seconds'] / 60:5.1f} min", flush=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
