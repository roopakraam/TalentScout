# TalentScout v2

Nature-inspired feature selection for football talent identification: comparing a
Genetic Algorithm, Binary PSO and Binary Grey Wolf Optimizer on FIFA / EA FC player data.

Course project for CSE578 Optimization Techniques for Intelligent Systems
(Lovely Professional University, session 2026-27).

**Question.** EA's games rate every footballer on about 45 attributes. Which young
players (21 or under, below 80 overall) reach 80+ overall within three game versions,
and what is the smallest set of attributes that still predicts it?

## Layout

```
data/raw/            Kaggle CSV (not committed; see data/README.md)
data/processed/      labelled rows and frozen split ids (not committed)
src/                 data pipeline, features, splits, metrics, models, fitness function
src/optimizers/      GA, Binary PSO, Binary GWO, random search
notebooks/           01 data and label ... 06 SHAP and case studies
scripts/             command-line entry points for the long runs
results/             one CSV per run, tables and figures
docs/                decision log, paper notes, research-gap check
```

A full "how to rerun" section is added once the experiments are complete.
