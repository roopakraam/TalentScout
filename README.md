# TalentScout v2

Nature-inspired feature selection for football talent identification: a Genetic Algorithm,
Binary PSO and Binary Grey Wolf Optimizer compared on FIFA 15 to EA Sports FC 24 player data.

Course project for CSE578 Optimization Techniques for Intelligent Systems
(Lovely Professional University, session 2026–27).

**Question.** Of the players aged 21 or under and rated below 80, who reaches 80+ overall
within the next three game versions? And what is the smallest set of the game's attributes
that predicts it as well as all 45?

## Main results

| | Result |
| --- | --- |
| Data | 19,410 player-versions (FIFA 15–21), 2.7% future elite; train FIFA 15–18, validation 19–20, test FIFA 21 |
| Optimizers (validation, 10 seeds each) | GA, Binary PSO and Binary GWO beat random search on all 10 seeds (Wilcoxon, Holm p = 0.012); no significant difference between them; Binary GWO has the lowest mean fitness and the smallest subsets (18 of 45 features) |
| Trade-off | With α = 0.80, Binary GWO keeps 8–12 features at validation PR-AUC 0.62–0.64 (all 45 features: 0.50 with the same settings) |
| Test season (FIFA 21) | Every model scores PR-AUC 0.28–0.35; subsets of 16–21 features match all 45 within noise; no model beats EA's potential rating (0.33) for the 80+ label, but LightGBM does for a 75+ label (0.60 vs 0.52) |
| Lesson | Validation gains from feature selection did not transfer: the validation set was easier than test and the optimizers fitted its noise |

The full write-up is `report/TalentScout_v2_Report.docx` (and `.pdf`); the paper is `paper/TalentScout_v2_Paper.docx` (and `.pdf`; LaTeX version in `paper/main.tex`).

## Repository layout

```
data/raw/            Kaggle CSV (not committed; scripts/download_data.py fetches it)
data/processed/      labelled rows and frozen split ids (not committed; notebook 01 builds them)
src/                 data pipeline, features, split, metrics, models, tuning, fitness function
src/optimizers/      GA, Binary PSO, Binary GWO, random search (shared interface)
notebooks/           01 data and label · 02 models and Bayesian tuning · 03 optimizer sanity check
                     04 feature-selection experiments · 05 test evaluation · 06 SHAP and case studies
scripts/             data download, Optuna studies, the full rerun
run_experiments.py   resumable feature-selection runner (one CSV + JSON per algorithm and seed)
tests/               unit tests for the optimizers (toy problems)
results/             every run, study, table and figure
report/              report chapters (Markdown) and the builder that makes the Word file
paper/               LaTeX paper draft (LNCS format)
docs/                decision log, literature notes, research-gap check, instructor notes
```

## Rerunning everything

Requirements: Python 3.12 and the packages in `requirements.txt` (exact versions of the
whole environment in `requirements-lock.txt`). The MLP uses TensorFlow on the GPU if one is
visible and falls back to the CPU otherwise; everything else runs on the CPU.

```bash
source ~/tf-gpu-env/bin/activate          # or: python -m venv .venv && pip install -r requirements.txt
python scripts/download_data.py           # Kaggle data, no account needed (about 30 MB)
bash scripts/reproduce_all.sh --clean     # every step in order, about 3 hours on an 8-core laptop
```

`reproduce_all.sh` runs, in order:

| Step | Command | Time | Output |
| --- | --- | --- | --- |
| Data, label, split | notebook 01 | 1 min | `data/processed/`, `results/tables/` |
| 13 Optuna studies | `python scripts/run_hpo.py all` | 1 h | `results/hpo/` |
| Models and tuning | notebook 02 | 2 min | RQ1 validation table |
| Optimizer checks | `python tests/test_optimizers.py`, notebook 03 | 5 min | `results/sanity_check/` |
| 40 feature-selection runs | `python run_experiments.py` | 40 min | `results/feature_selection/` |
| Inertia ablation, α sweep | `run_experiments.py` with `--algs bpso_w_dec`, `--alpha` | 25 min | `results/ablation_bpso_inertia/`, `results/alpha_sweep/` |
| Optimizer comparison | notebook 04 | 1 min | RQ2 tables and figures |
| Final subsets retuned | `python scripts/run_hpo.py final` | 3 min | `results/hpo/final_*`, `results/final/` |
| Cross-validation fallback | `run_experiments.py --fitness cv` | 40 min | `results/fallback_cv_fitness/` |
| Test set, SHAP, case studies | notebooks 05 and 06 | 3 min | test tables and figures |
| Report | `python report/build_report.py` | 10 s | `report/TalentScout_v2_Report.docx` |

Every run and study writes its own file and is skipped if that file exists, so any step can be
stopped and restarted. Seeds, thread counts and LightGBM's deterministic mode are fixed, so a
rerun reproduces every number exactly (see `docs/reproducibility.md`).

Note on this machine: LightGBM needs the OpenMP runtime `libgomp.so.1`. Without root access it was
unpacked into the virtual environment (`tf-gpu-env/lib/openmp`, added to `LD_LIBRARY_PATH` by the
activate script); with root, `sudo apt install libgomp1` does the same.

## How the study protects its results

* The data design (label, split, features) was frozen before any model was fitted (`git tag data-frozen`).
* Every test player is removed from training and validation; a test checks it.
* All optimizers get the same starting population per seed, the same 600 evaluations and the same seeds.
* The final feature sets and the "best subset" were fixed on validation before the test set was used (decision D30), and each configuration was scored on the test set once.
* Every decision is logged with its reason in `docs/decision_log.md`.
