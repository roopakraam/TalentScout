# Methods {#sec:methods}

@fig:design summarises the pipeline. All code is in the project repository: the data pipeline and split in `src/`, the optimizers in `src/optimizers/`, the long runs in `run_experiments.py` and `scripts/`, and the analysis in six notebooks.

![Study design. Feature rows (FIFA 15 to 21) are split by version; each row's label looks three versions ahead. Selection and tuning use train and validation only; the test version is scored once per final configuration.](results/figures/fig00_study_design.png){#fig:design width=6.2}

## Prediction models (RQ1)

Five scorers are compared (@tab:models). Class imbalance is handled with class weights everywhere: `class_weight="balanced"` for logistic regression and the random forest, `scale_pos_weight` = negatives/positives = 34.3 for LightGBM (Ke et al., 2017), and a positive-class weight of 34.3 in the MLP's binary cross-entropy. Imputation and scaling sit inside each model's pipeline and are fitted on the training rows only. Seeds and thread counts are fixed, and LightGBM runs in its deterministic mode, so every number in this report is reproducible to the last digit.

Table: Prediction models. {#tab:models}
| Model | Implementation | Notes |
| --- | --- | --- |
| EA potential | the `potential` column used directly as a score | the baseline to beat (H1b) |
| Logistic regression | scikit-learn, L2 penalty | median imputation and standardisation |
| Random forest | scikit-learn, 500 trees | median imputation |
| LightGBM | lightgbm 4.7 | the main model and the model inside the fitness function |
| MLP | Keras 3 / TensorFlow 2.21 on an RTX 3050 GPU | 2 or 3 hidden ReLU layers with dropout, AdamW, batch 256, early stopping on validation PR-AUC (patience 10) |

## Hyperparameter optimization (Unit V)

Every model is tuned with Optuna's TPE sampler for 50 trials that maximise validation PR-AUC. The search spaces are listed in @tab:spaces. For LightGBM, TPE is compared with Optuna's random sampler: 50 trials each, sampler seeds 0 to 4, model seed fixed, so two studies differ only in the hyperparameters they propose. Each model's final configuration is the best trial of its seed-0 TPE study, so all models receive exactly 50 trials.

Table: Hyperparameter search spaces (TPE, 50 trials). {#tab:spaces}
| Model | Hyperparameter ranges |
| --- | --- |
| LightGBM | leaves 15 to 255; learning rate 0.01 to 0.2 (log); trees 100 to 1000; min. child samples 5 to 100; row subsample 0.5 to 1; column subsample 0.5 to 1; L2 penalty 0.001 to 10 (log) |
| Logistic regression | C from 0.0001 to 100 (log) |
| Random forest | max. depth 3 to 20; min. samples per leaf 1 to 50; features per split 10% to 100% |
| MLP | hidden units 32 to 256; 2 or 3 layers; dropout 0 to 0.5; learning rate 0.0001 to 0.01 (log); weight decay 10^-6 to 10^-3 (log) |

## Feature selection as an optimization problem

The decision variable is **x** ∈ {0, 1}^45. The objective to minimise is the weighted sum of @eq:weighted with PR-AUC as the accuracy measure:

$$f(\mathbf{x}) = \alpha\,\left(1 - \mathrm{PRAUC}_{\mathrm{val}}(\mathbf{x})\right) + (1 - \alpha)\,\frac{|\mathbf{x}|}{45}, \qquad \alpha = 0.99$$ {#eq:fitness}

The only constraint is that at least one feature is selected; an all-zero vector is repaired by switching on one random bit. PRAUC_val(**x**) is the validation PR-AUC of a LightGBM model trained on the training rows with only the selected columns. Because the optimizers call this function thousands of times, LightGBM uses fixed, moderate settings (200 trees, learning rate 0.05, otherwise library defaults) instead of tuned ones. One evaluation takes about 0.23 seconds on the project laptop. Every evaluated subset is cached by its bit pattern; the cache stores PR-AUC rather than fitness, so runs with a different α can reuse it.

## The optimizers

All optimizers share one interface and one fairness protocol. For seed s, the starting population of 20 random bit vectors (each bit 1 with probability 0.5) comes from one random generator and is identical for every algorithm; each algorithm's own random choices come from a second generator. The budget is 600 fitness evaluations, and **every call counts**, including calls on a subset already in the cache, so each algorithm gets exactly 600 looks at the problem. Each algorithm is run with seeds 0 to 9. The settings are in @tab:optimizers.

Table: Optimizer settings. {#tab:optimizers}
| Algorithm | Settings |
| --- | --- |
| Genetic Algorithm | tournament selection (k = 3); uniform crossover, rate 0.9; bit-flip mutation 1/45; 2 elites carried over without re-evaluation (18 new evaluations per generation) |
| Binary PSO | sigmoid transfer (@eq:bpso); c1 = c2 = 2; velocity clamp ±6; inertia w = 1 (see below) |
| Binary GWO (bGWO2) | a falls linearly from 2 to 0; sigmoid transfer of @eq:bgwo; alpha, beta and delta are the three best distinct solutions found so far |
| Random search | the shared starting population, then batches of 20 random vectors with the same bit distribution |

**The inertia weight of Binary PSO.** The plan specified the standard continuous-PSO schedule, w falling from 0.9 to 0.4. On a toy problem with a known optimum (minimise the number of selected features, D = 45, 600 evaluations), that setting never converged: the best subsets kept about 7 features, and the swarm stayed almost as spread out as a random population. The reason lies in @eq:bpso. When a bit already agrees with both the particle's best and the swarm's best, the attraction terms of @eq:pso vanish and the update is just v ← w v. With w < 1 the velocity decays toward zero, and a velocity of zero means the bit is 1 with probability one half. A falling inertia weight therefore makes the binary swarm more random over time, the opposite of its purpose in continuous PSO. With w = 1, as in the original binary PSO of Kennedy and Eberhart (1997), the swarm reaches the optimum on 10 of 10 seeds. The main study uses w = 1, a decision taken on the toy problem before any football run, and the planned schedule is reported as an ablation.

**Verification before the real runs.** Each optimizer was tested on two problems before touching football data (@fig:toy and @fig:bc). On the toy problem, GA, Binary PSO with w = 1 and Binary GWO reach the optimum of one feature on all 10 seeds, while random search stops at 11 to 14 features. On the Breast Cancer Wisconsin data set (569 tumours, 30 features; the setting of the course's Practical 4), with a fitness based on 5-fold cross-validated logistic regression accuracy, all three match or beat random search on every one of 5 seeds. Twelve unit tests check the budget, the shared start, the monotone best-so-far history and run-to-run reproducibility.

![Verification on the toy problem (minimise the number of selected features; optimum = 1), mean of 10 seeds with ±1 sd bands. The planned Binary PSO schedule (w from 0.9 to 0.4) stalls near 7 features.](results/figures/fig03a_toy_convergence.png){#fig:toy width=5.4}

![Verification on Breast Cancer Wisconsin (30 features, fitness from 5-fold cross-validated logistic regression accuracy), mean of 5 seeds with ±1 sd bands.](results/figures/fig03b_breast_cancer_convergence.png){#fig:bc width=5.4}

## Evaluation of the optimizers (Units IV and V)

For each of the 40 main runs the experiment runner records every fitness call. The comparison uses:

* solution quality: best fitness, its validation PR-AUC and subset size (mean ± sd over 10 runs);
* convergence: best fitness so far against the number of evaluations;
* exploration: population diversity, the mean pairwise Hamming distance after each iteration;
* stability: mean pairwise Jaccard similarity (shared features divided by features in either) of the 10 final subsets of an algorithm (Kalousis, Prados and Hilario, 2007);
* cost: wall-clock time and the number of distinct subsets trained;
* significance: a Friedman test across the four methods, then two-sided Wilcoxon signed-rank tests on the six pairs, paired by seed, with Holm's correction (Derrac et al., 2011).

H2a is checked like-for-like: a subset qualifies if it has at most 15 features and its validation PR-AUC is at least 0.01 below that of all 45 features **under the same fixed settings** or higher. To trace the trade-off between accuracy and size (Unit V, multi-objective view), the best optimizer was rerun with α = 0.95, 0.90 and 0.80 (seeds 0 to 4). Every subset evaluated in any run also gives a point (size, PR-AUC), and the best PR-AUC seen at each size forms an empirical Pareto front.

## Final configurations and the test set {#sec:final}

The test set was used only after five final configurations had been fixed on validation data, by rules written into the decision log beforehand:

1. **All 45 features.**
2. to 4. **GA, Binary PSO and Binary GWO subsets:** for each optimizer, the subset of its run with the best validation fitness.
5. **SHAP top-k:** the k features with the largest mean |SHAP| of the tuned all-feature LightGBM on the validation set, where k is the median size of the three optimizer subsets (k = 20).

Each feature set then received its own LightGBM, tuned with 50 TPE trials on validation, because the fixed fitness settings were chosen for speed. The headline "best subset" was fixed as the subset configuration with the highest tuned validation PR-AUC. All final models are fitted on the training rows only, so validation and test scores come from the same fitted model. Each configuration, and each RQ1 model, was scored on the test set once.

Test metrics are PR-AUC with a 95% percentile bootstrap interval (1,000 resamples of test rows), ROC-AUC, precision@50, recall@50 and the Brier score, raw and after Platt scaling fitted on validation predictions. Differences between models use a paired bootstrap: both models are scored on the same resampled rows, and the interval of the difference is reported. Two sensitivity checks reuse the tuned settings: (a) players missing from all three future versions counted as negatives, and (b) a second label, 75+ within three versions, which tests whether the conclusions depend on the exact threshold or on the circularity of predicting EA ratings from EA ratings.

**The fallback for validation overfitting.** The plan named one risk explicitly: if the selected subsets lost much of their validation advantage on the test set, the selection should be repeated with a fitness that never uses validation. This fallback fitness is the mean PR-AUC over a 3-fold cross-validation of the training rows, stratified by label and grouped by player, with the same fixed LightGBM settings, α, budget and starting populations. GA, Binary PSO and Binary GWO were rerun with it for seeds 0 to 3.
