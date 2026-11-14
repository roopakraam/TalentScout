# Results II: comparing the optimizers (RQ2) {#sec:results-optimization}

This chapter is the core of the CSE578 part of the project. All 40 main runs (four methods, ten seeds) completed with exactly 600 evaluations each, no missing values and a best-so-far curve that never rises, and for every seed the four methods started from the same 20 subsets. The runs took about 40 minutes on four parallel workers.

## Solution quality and the H2a check

@tab:rq2-summary summarises the best solution of each run. All three metaheuristics end far below random search, and Binary GWO has the lowest mean fitness, the smallest spread across seeds and the smallest subsets.

Table: Best solution per run, mean ± sd over 10 seeds (validation set, fixed fitness settings). "Distinct subsets" is the number of real model trainings out of 600 calls. {#tab:rq2-summary}
| Method | Best fitness | Validation PR-AUC | Features | Distinct subsets | Minutes per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| GA | 0.389 ± 0.019 | 0.612 ± 0.019 | 22.1 (20 to 25) | 455 | 3.1 |
| Binary PSO | 0.380 ± 0.011 | 0.621 ± 0.011 | 21.6 (17 to 24) | 595 | 4.1 |
| Binary GWO | **0.373 ± 0.007** | **0.627 ± 0.007** | **17.9 (16 to 22)** | 381 | 2.5 |
| Random search | 0.426 ± 0.007 | 0.575 ± 0.008 | 23.7 (19 to 27) | 600 | 4.0 |
| All 45 features | 0.506 | 0.499 | 45 | | |

The last row is the reference for H2a: LightGBM on all 45 features under the same fixed settings reaches a validation PR-AUC of only 0.499. Every one of the 30 metaheuristic runs beats it by 0.11 to 0.15, so the accuracy half of H2a holds many times over. The size half does not: at α = 0.99 no run reaches 15 features or fewer. A feature costs only 0.01/45 ≈ 0.0002 of fitness, so any subset with a slightly higher PR-AUC wins regardless of its size.

## Convergence

@fig:convergence shows the mean best fitness against the number of evaluations. The three metaheuristics are indistinguishable for about 100 evaluations, and all three pass random search's final value after 80 to 85 evaluations; most of the remaining budget buys improvements that random sampling never finds. After about 300 evaluations the GA flattens (its mean best fitness improves by only 0.006 between call 300 and call 600), while Binary GWO keeps improving until about call 400 and Binary PSO until about call 550.

![Convergence on the football data: best fitness so far against fitness evaluations, mean of 10 seeds with ±1 sd bands. The grey line is the fitness of all 45 features.](results/figures/fig04a_convergence.png){#fig:convergence width=6.0}

## Exploration and exploitation

Population diversity (@fig:diversity) separates the algorithms more clearly than fitness does. All start from the same random population (mean Hamming distance 22.7, close to the 22.5 expected for random vectors). Binary GWO's pack contracts fastest and ends with nearly identical wolves (1.6 differing bits on average), the GA keeps about 4 differing bits, and Binary PSO with w = 1 still differs in 7.7 bits after 600 evaluations. This is visible in the cost column of @tab:rq2-summary: Binary PSO trains a new model for 595 of its 600 calls, while Binary GWO revisits subsets and trains only 381. Binary GWO exploits early and hard, Binary PSO keeps exploring, and the GA sits in between.

![Population diversity: mean pairwise Hamming distance of the population against evaluations used, mean of 10 seeds with ±1 sd bands.](results/figures/fig04b_diversity.png){#fig:diversity width=6.0}

## Subset stability

The subsets differ a lot from run to run (@tab:stability). The mean Jaccard similarity between two runs of the same algorithm is 0.39 to 0.44, barely above that of random search's best subsets (0.385), which are essentially unrelated. Only four attributes are in at least 8 of 10 subsets for all three algorithms: age, height, volleys and balance. Many quite different subsets reach almost the same validation PR-AUC, so a single run's subset is not a reliable checklist. This is the stability problem Xue et al. (2016) describe.

Table: Stability of the 10 final subsets of each method (45 pairs per method). {#tab:stability}
| Method | Mean Jaccard | sd | Features in all 10 subsets | Features in at least 8 |
| --- | ---: | ---: | ---: | ---: |
| GA | 0.410 | 0.078 | 2 | 8 |
| Binary PSO | 0.436 | 0.090 | 4 | 8 |
| Binary GWO | 0.387 | 0.092 | 1 | 7 |
| Random search | 0.385 | 0.077 | 0 | 6 |

## Statistical tests

The Friedman test rejects equal performance of the four methods (χ² = 19.9, p = 1.8 × 10⁻⁴); the mean ranks are 1.6 for Binary GWO, 2.0 for Binary PSO, 2.4 for the GA and 4.0 for random search. @tab:wilcoxon gives the pairwise Wilcoxon tests. Each metaheuristic beats random search on all ten seeds, the strongest result possible with ten pairs (p = 0.002, Holm-adjusted 0.012). No difference among the three metaheuristics survives the correction. A secondary family of tests on subset size shows that Binary GWO selects significantly fewer features than the GA (4.2 fewer on average, Holm p = 0.023) and Binary PSO (3.7 fewer, p = 0.039).

Table: Wilcoxon signed-rank tests on the best fitness of each run, paired by seed, with Holm's correction for six comparisons. A negative difference favours the first method. {#tab:wilcoxon}
| Comparison | Mean difference | Seeds won by first | p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| GA vs random search | −0.037 | 10 of 10 | 0.002 | **0.012** |
| Binary PSO vs random search | −0.047 | 10 of 10 | 0.002 | **0.012** |
| Binary GWO vs random search | −0.053 | 10 of 10 | 0.002 | **0.012** |
| GA vs Binary PSO | +0.009 | 3 of 10 | 0.232 | 0.465 |
| GA vs Binary GWO | +0.016 | 3 of 10 | 0.131 | 0.393 |
| Binary PSO vs Binary GWO | +0.006 | 3 of 10 | 0.275 | 0.465 |

## Ablation: the planned inertia schedule

Binary PSO with the planned inertia schedule (w from 0.9 to 0.4) was run on the same ten seeds. It lost to the w = 1 version on all ten (mean fitness 0.408 against 0.380, Wilcoxon p = 0.002), and its swarm never contracted: the final diversity of 20.6 is close to that of a random population. It still beat random search on every seed, so the binary update itself works; the schedule only disables convergence. The decision taken on the toy problem holds on the real problem.

## The accuracy and size trade-off

Binary GWO was rerun with larger prices on a feature (@tab:sweep). From α = 0.95 downward the subsets shrink to 15 features or fewer while validation PR-AUC stays as high as at α = 0.99: at α = 0.80 the subsets have 8 to 12 features and a PR-AUC of 0.620 to 0.640. Fourteen of the fifteen sweep runs meet H2a.

Table: α sweep with Binary GWO (α = 0.99: the ten main runs; other values: seeds 0 to 4). {#tab:sweep}
| α | Runs | Validation PR-AUC | Features, mean (range) | Runs meeting H2a |
| ---: | ---: | ---: | ---: | ---: |
| 0.99 | 10 | 0.627 ± 0.007 | 17.9 (16 to 22) | 0 |
| 0.95 | 5 | 0.628 ± 0.013 | 14.6 (14 to 15) | 5 |
| 0.90 | 5 | 0.633 ± 0.009 | 15.0 (14 to 17) | 4 |
| 0.80 | 5 | 0.631 ± 0.008 | 9.6 (8 to 12) | 5 |

@fig:front places these runs among all 30,132 distinct subsets evaluated in the study. The empirical Pareto front rises steeply up to about nine features (0.566 with six features, 0.620 with eight, 0.637 with nine) and is almost flat beyond that (0.647 with fifteen). On validation, a scout gains little from more than about nine to twelve well-chosen attributes, and nothing from all 45.

![Validation PR-AUC against subset size for every subset evaluated (grey), with the empirical Pareto front (step line) and the best subsets of the α sweep. Diamonds mark all 45 features and current overall alone.](results/figures/fig04c_tradeoff_front.png){#fig:front width=6.0}

## The selected subsets on the test set

The validation picture changes on the test season. @tab:subsets-test compares the four subset configurations with all 45 features, each with its own tuned LightGBM. On validation the GA and Binary PSO subsets lead all 45 features by about 0.04 to 0.06 PR-AUC. On test, all five configurations lie between 0.284 and 0.313, and every paired bootstrap interval of the difference from all 45 features includes zero.

Table: Final feature sets with retuned LightGBM. The validation-to-test gap is validation (tuned) minus test PR-AUC; intervals are 95% paired bootstrap intervals of the test difference from all 45 features. {#tab:subsets-test}
| Configuration | Features | Validation PR-AUC (tuned) | Test PR-AUC [95% CI] | Gap | Test difference from all 45 [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 45 features | 45 | 0.593 | 0.301 [0.205, 0.420] | 0.292 | |
| GA subset | 20 | 0.636 | 0.296 [0.209, 0.422] | 0.340 | −0.005 [−0.062, 0.058] |
| Binary PSO subset | 21 | **0.650** | 0.313 [0.222, 0.435] | 0.337 | +0.012 [−0.068, 0.097] |
| Binary GWO subset | 16 | 0.624 | 0.284 [0.207, 0.405] | 0.341 | −0.017 [−0.092, 0.052] |
| SHAP top-20 | 20 | 0.592 | 0.291 [0.204, 0.403] | 0.301 | −0.010 [−0.063, 0.040] |

Two things follow. First, the compact subsets lose nothing measurable on test: with 16 to 21 of 45 attributes, they score the same as all 45 within the precision 72 test positives allow. Second, the validation advantage of the optimizer subsets was not real: their validation-to-test gap (0.34) is about 0.05 larger than that of all 45 features (0.29). That extra gap is the classic selection bias of wrapper methods: an optimizer that looks at validation PR-AUC thousands of times finds subsets that fit the validation set's noise. @sec:results-prediction shows that most of the remaining gap, the part shared by every model, comes from the validation set being easier than the test set.

## Choosing subsets without the validation set

The plan's fallback for exactly this situation was to repeat the selection with a fitness that never sees the validation set: the mean PR-AUC of a 3-fold cross-validation on the training seasons, grouped by player. GA, Binary PSO and Binary GWO were rerun with it (seeds 0 to 3), and every subset from both criteria was then scored with the same fixed settings on all three data sets (@fig:fallback).

![Subsets chosen on validation (30 runs) and by grouped cross-validation on train (12 runs), each scored by the cross-validation, on validation and on test, with fixed LightGBM settings. Black lines mark means; grey lines mark all 45 features.](results/figures/fig05c_fallback_cv_selection.png){#fig:fallback width=6.0}

Each criterion's subsets look best on the data that chose them. The cross-validation subsets score 0.365 to 0.372 in that cross-validation, against 0.320 for all 45 features, and the validation subsets score 0.61 to 0.63 on validation, against 0.50. On test, neither group beats all 45 features (0.312 with these settings): the validation-chosen subsets average 0.295 and the cross-validation-chosen ones 0.280 (Mann-Whitney p = 0.25). One of the 12 cross-validation subsets and 10 of the 30 validation subsets beat all 45 features on test. Moving the selection to other data did not cure the problem. With 113 validation and 344 training positives, the PR-AUC differences between good subsets are a few hundredths, about as large as the noise, so whichever data the optimizer looks at, it finds subsets that fit that data's noise.

## Verdicts on H2a, H2b and H2c

**H2a: partly supported.** On validation, every metaheuristic run beats the all-feature model's PR-AUC, but at the planned α = 0.99 none reaches 15 features; with α ≤ 0.95 Binary GWO meets both conditions in 14 of 15 runs. On test, subsets of 16 to 21 features are statistically indistinguishable from all 45. The size condition was shown only for Binary GWO.

**H2b: supported.** GA, Binary PSO and Binary GWO each beat random search on all ten seeds under the same budget (Holm-adjusted p = 0.012 each).

**H2c: partly supported.** The three algorithms differ clearly in how they search (diversity, distinct subsets evaluated, subset size) and modestly in convergence, but not significantly in final fitness, and their subset stability is uniformly low rather than different.
