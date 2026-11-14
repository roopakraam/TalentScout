#!/usr/bin/env bash
# Rerun the whole study, from the raw Kaggle CSV to every table, figure and the report.
#
#   source ~/tf-gpu-env/bin/activate
#   bash scripts/reproduce_all.sh            # about 3 hours on an 8-core laptop with an RTX 3050
#
# Every script skips work whose output already exists, so to recompute everything
# delete the result folders first:
#
#   bash scripts/reproduce_all.sh --clean
#
# The order matters: each step reads what the previous steps wrote.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf results/hpo results/feature_selection results/ablation_bpso_inertia results/alpha_sweep \
           results/final results/fallback_cv_fitness results/sanity_check results/tables results/figures \
           data/processed/labelled.parquet data/processed/split_ids.csv
fi

run_notebook() {
    echo "== notebook $1"
    jupyter nbconvert --to notebook --execute --inplace "notebooks/$1" --ExecutePreprocessor.timeout=-1
}

echo "== 1. data (W1)"
python scripts/download_data.py
run_notebook 01_data_and_label.ipynb

echo "== 2. models and tuning (W2-W3): 13 Optuna studies, about 1 hour"
python scripts/run_hpo.py all --workers 4
run_notebook 02_models_and_bayesian_tuning.ipynb

echo "== 3. optimizers (W3-W4): unit tests and the sanity check"
python tests/test_optimizers.py
run_notebook 03_optimizer_sanity_check_breast_cancer.ipynb

echo "== 4. feature-selection runs (W4-W5): about 1 hour"
python run_experiments.py --workers 4
python run_experiments.py --algs bpso_w_dec --out results/ablation_bpso_inertia --workers 4
python run_experiments.py --algs bgwo --alpha 0.95 0.90 0.80 --seeds 0 1 2 3 4 \
    --out results/alpha_sweep --reuse-cache --workers 4
run_notebook 04_feature_selection_experiments.ipynb

echo "== 5. final configurations and the test set (W6)"
python scripts/run_hpo.py final --workers 4
python run_experiments.py --algs ga bpso bgwo --seeds 0 1 2 3 --fitness cv \
    --out results/fallback_cv_fitness --workers 4
run_notebook 05_test_evaluation_and_stats.ipynb
run_notebook 06_shap_and_case_studies.ipynb

echo "== 6. report"
python report/build_report.py
echo "done"
