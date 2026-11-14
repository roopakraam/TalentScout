# Introduction {#sec:intro}

Every year clubs sign teenagers on the strength of a few seasons of evidence and a scout's judgement. Most of those players never reach the level the club hoped for, and a few who were passed over become stars elsewhere. The question this project studies is one version of that problem, framed so that it can be answered with public data: given everything a football video game knows about a young player today, will he be an elite player three seasons from now?

EA Sports' FIFA series (renamed EA Sports FC in 2023) rates every licensed professional on an overall scale from 1 to 99 and on about 45 detailed attributes, from sprint speed to interceptions. The ratings are revised every season by a network of scouts and editors, and a public Kaggle dataset collects them for FIFA 15 to EA Sports FC 24. That gives ten consecutive "launch-day" snapshots for more than 30,000 players, enough to follow a player from age 17 to 26 and to see who crossed the line of 80 overall, which in the game roughly separates first-team players at top clubs from everyone else.

## Two questions in one study

The project has a machine-learning part and an optimization part, matching the two halves of its motivation.

The **prediction question** is whether a model trained on past seasons can rank young players by their chance of reaching 80+ within three versions, and whether it does better than the answer EA already publishes: the `potential` rating, EA's own forecast of a player's ceiling. A model that cannot beat that single number adds little for a scout.

The **optimization question** comes from the practical side. A scout watching a youth match cannot measure 45 attributes for every player. If a small set of attributes predicts as well as all of them, that set is a checklist. Finding it is a combinatorial problem: with D = 45 candidate attributes there are 2^45, about 3.5 × 10^13, possible subsets, far too many to try. This is the setting where nature-inspired metaheuristics are used in practice (Xue et al., 2016). The project compares three of them from the CSE578 syllabus, a Genetic Algorithm (Unit II), Binary Particle Swarm Optimization (Unit III) and the Binary Grey Wolf Optimizer (Unit IV), against random search under an equal evaluation budget. Bayesian optimization (Unit V) tunes the models' hyperparameters and is itself compared with random search.

## Research questions

* **RQ1 (prediction).** Which model best predicts future elite players, and does it beat EA's potential rating? *H1a:* LightGBM matches or beats a multilayer perceptron, as tabular-data benchmarks suggest (Grinsztajn, Oyallon and Varoquaux, 2022). *H1b:* the best model beats EA potential used alone as a score.
* **RQ2 (optimization).** Can nature-inspired feature selection shrink the attribute set without losing accuracy, and which optimizer does it best? *H2a:* GA, Binary PSO and Binary GWO each find a subset of at most a third of the features with validation PR-AUC within 0.01 of the all-feature model. *H2b:* all three beat random search under the same evaluation budget. *H2c:* the three differ in convergence speed and subset stability even when their final accuracy is similar.
* **RQ3 (interpretation).** Do the attributes the optimizers keep agree with SHAP importance? *H3:* age and current overall are kept by every optimizer; the optimizers disagree mainly on the skill attributes.

## What the study found, in brief

On the validation seasons, everything worked as hoped: the three metaheuristics beat random search on every one of ten seeds, and the subsets they chose (16 to 25 attributes) scored far higher than all 45 attributes together. On the untouched test season, those gains disappeared. Every model, with or without feature selection, lands between 0.28 and 0.35 PR-AUC, none of them significantly better than EA's potential rating, and the selected subsets are statistically indistinguishable from the full attribute set. The report traces the drop to two causes. The way the planned split builds its validation set makes validation easier than test, and selecting features on any limited data set fits that set's noise: a pre-planned fallback that selects by cross-validation on the training seasons did no better on test. The compact subsets themselves survive. The scouting checklist can be cut to a third or half of the attributes without measurable loss, but no attribute set found here beats EA's own judgement for the 80+ target, although the models do beat it for a 75+ target.

## Contributions

1. A leakage-controlled benchmark for "future elite" prediction on FIFA 15 to FC 24 launch ratings, with a temporal, player-grouped split and an untouched test season, reproducible from one script.
2. A fair comparison of GA, Binary PSO and Binary GWO for wrapper feature selection: shared starting populations, an equal budget of 600 evaluations, ten seeds, Friedman and Holm-corrected Wilcoxon tests, and convergence, diversity and stability measures.
3. A worked example of why validation-driven selection must be checked on held-out data, with a decomposition of the validation-to-test gap and a cross-validation fallback.
4. A practical finding on Binary PSO: the inertia-weight schedule usually recommended for continuous PSO stops the binary version from converging, which a toy problem caught before any real run.

## Structure of the report

@sec:background covers the optimization background, and @sec:related places the study among related work. @sec:data describes the data, the label and the split, and @sec:methods the models, the optimization formulation and the evaluation protocol. @sec:results-prediction, @sec:results-optimization and @sec:results-interpretation report the results for RQ1, RQ2 and RQ3. @sec:discussion discusses what they mean and where they stop, and @sec:conclusion concludes.
