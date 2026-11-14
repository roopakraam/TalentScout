# Results I: prediction (RQ1) {#sec:results-prediction}

## Validation results and the tuning comparison

@tab:rq1-val lists the validation scores of every model before and after tuning. EA's potential rating is already a strong ranking (PR-AUC 0.423, ROC-AUC 0.961); it is built by EA's scouts with exactly this question in mind. Every model beats it on validation, the tuned ones by 0.11 to 0.17. Tuning helps every model, LightGBM the most (0.526 to 0.593). ROC-AUC barely separates the models (0.95 to 0.97), which is why PR-AUC is the primary metric.

Table: Validation results (FIFA 19 and 20; 113 future elite players among 3,974). Tuned models used 50 TPE trials. {#tab:rq1-val}
| Model | Setting | PR-AUC | ROC-AUC | Precision@50 | Recall@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| EA potential | baseline | 0.423 | 0.961 | 0.56 | 0.25 |
| Logistic regression | default / tuned | 0.483 / 0.532 | 0.963 / 0.966 | 0.58 / 0.58 | 0.26 / 0.26 |
| Random forest | default / tuned | 0.514 / 0.539 | 0.958 / 0.956 | 0.66 / 0.72 | 0.29 / 0.32 |
| MLP | default / tuned | 0.532 / 0.573 | 0.962 / 0.962 | 0.64 / 0.70 | 0.28 / 0.31 |
| LightGBM | default / tuned | 0.526 / **0.593** | 0.951 / 0.952 | 0.66 / **0.82** | 0.29 / **0.36** |

The MLP trained on the GPU and peaked within the first few epochs (epoch 3 for the tuned network), after which validation PR-AUC declined; early stopping with restored weights kept it competitive, and the tuned network chose heavy dropout (0.47).

**Bayesian optimization against random search.** @fig:bo compares the two samplers for LightGBM over five seeds. The two curves coincide for the first ten trials, because TPE starts with ten random trials drawn from the same seeded generator. After that, TPE concentrates its trials in good regions: 42% of its trials 11 to 50 score at least 0.58, against 10% for random search, and its median trial is better by 0.018. The best configuration found improves less, 0.596 for TPE against 0.592 for random search after 50 trials, with TPE ahead on four of five seeds and reaching 0.59 on all five (random search on three). With five seeds the difference cannot reach significance (Wilcoxon p = 0.19, where 0.0625 is the smallest possible value). Bayesian optimization finds good settings more reliably, but in this seven-dimensional space 50 random trials are a strong competitor, as Bergstra and Bengio (2012) observed for problems with few important hyperparameters.

![Bayesian optimization (TPE) against random search for LightGBM. Left: best validation PR-AUC so far, mean of 5 seeds with ±1 sd bands. Right: PR-AUC of every trial after TPE's 10 start-up trials; black lines mark the medians.](results/figures/fig02a_bo_vs_random.png){#fig:bo width=6.0}

## Test results

@tab:rq1-test gives the test-season results, the only unbiased estimates in the study. Each model was fitted on the training seasons and scored on FIFA 21 once.

Table: Test results (FIFA 21; 72 future elite players among 3,288; a random ranking scores 0.022). PR-AUC with 95% bootstrap intervals; Brier scores raw and after Platt scaling fitted on validation (always predicting the base rate scores 0.0214). {#tab:rq1-test}
| Model | PR-AUC [95% CI] | ROC-AUC | Precision@50 | Recall@50 | Brier raw | Brier Platt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EA potential | 0.327 [0.233, 0.446] | 0.913 | **0.56** | **0.39** | | |
| Logistic regression | 0.302 [0.217, 0.412] | **0.925** | 0.32 | 0.22 | 0.113 | 0.0187 |
| Random forest | **0.353** [0.254, 0.482] | 0.923 | 0.44 | 0.31 | 0.046 | **0.0170** |
| MLP | 0.307 [0.222, 0.419] | 0.922 | 0.36 | 0.25 | 0.130 | 0.0184 |
| LightGBM | 0.301 [0.205, 0.420] | 0.895 | 0.44 | 0.31 | 0.021 | 0.0188 |

All five scorers land between 0.30 and 0.35 (@fig:test, which also shows the feature-subset configurations of @sec:results-optimization). The intervals are wide, about ±0.10, because the test season has only 72 positives, and paired bootstrap comparisons find no significant difference. LightGBM minus EA potential is −0.026 [−0.116, 0.059], LightGBM minus the MLP −0.006 [−0.090, 0.084], and the MLP minus EA potential −0.020 [−0.105, 0.062]. In shortlist terms, EA potential's top 50 contains 28 of the 72 future elite players, LightGBM's top 50 contains 22.

![Test PR-AUC of every model and final configuration with 95% bootstrap intervals. The grey line marks a random ranking.](results/figures/fig05a_test_pr_auc.png){#fig:test width=6.0}

**Calibration.** The class weights that help ranking distort probabilities. Logistic regression and the MLP predict 25 to 27% on average for an outcome with a 2.2% base rate, and LightGBM is overconfident at the top of its ranking: its 50 highest-scored players average a predicted probability of 0.80, while 44% of them became elite (calibration slope 0.39). A one-parameter Platt rescaling fitted on validation scores fixes most of this. Every model's Brier score then drops to 0.017 to 0.019, below the 0.0214 of always predicting the base rate, and the slopes rise to 0.69 to 0.82 (@fig:pr-cal). Any probability shown to a scout should therefore be recalibrated; the rankings themselves are unaffected.

![Left: precision-recall curves on the test season. Right: calibration by shortlist rank (ranks 1 to 50, 51 to 100, 101 to 200, 201 to 400, 401 to 800, the rest); points below the diagonal are overconfident.](results/figures/fig05b_pr_curves_calibration.png){#fig:pr-cal width=6.0}

## Why validation scores were so much higher

Every model loses a large part of its validation PR-AUC on test: LightGBM 0.29, the MLP 0.27, logistic regression 0.23, the random forest 0.19. Even EA's potential rating, which involves no fitting at all, falls by 0.10. A drop that also hits the unfitted baseline cannot be overfitting alone, so the validation set itself must differ from the test set.

@tab:diagnostic shows how. It splits the labelled FIFA 19 and 20 rows into those that stayed in validation and those removed because the player also appears in the test set.

Table: FIFA 19 and 20 rows kept in validation and removed because the player is in the test set, compared with the test season. PR-AUC of EA potential (no fitting) and of the tuned LightGBM trained on FIFA 15 to 18. {#tab:diagnostic}
| Rows | n | Positive rate | Mean age | Positives already 75+ | EA potential PR-AUC | LightGBM PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kept in validation | 3,974 | 2.8% | 20.1 | 63% | 0.423 | 0.593 |
| Removed (players in test) | 2,426 | 1.8% | 18.8 | 33% | 0.211 | 0.186 |
| All FIFA 19 and 20 rows | 6,400 | 2.4% | 19.6 | 55% | 0.303 | 0.474 |
| Test (FIFA 21) | 3,288 | 2.2% | 19.6 | 46% | 0.327 | 0.301 |

The removed rows belong to players who were still eligible in FIFA 21: young and still below 80, the slow developers. Their outcomes are hard to call for every method (EA potential 0.211, LightGBM 0.186). The rows that stayed are older, and their positives were mostly already close to 80: 63% were rated 75 or more, against 46% in test. Those players are easy to spot, so every method scores well on validation. Over all FIFA 19 and 20 rows, both scores move most of the way back toward their test values. The plan's rule of removing test players from validation, meant to stop the model recognising test players, made validation an easier and different problem than test. A 3-fold cross-validation within the training seasons, grouped by player, gives a far more honest estimate: all 45 features with the fixed fitness settings score 0.320 there and 0.312 on test, against 0.499 on validation.

## Sensitivity checks

@tab:sensitivity repeats the main comparison with the players who left the game counted as negatives, and with a second label, 75+ within three versions, using the hyperparameters tuned for the main label.

Table: Sensitivity checks on the test season: test PR-AUC [95% CI] and the paired difference between LightGBM (all features) and EA potential. {#tab:sensitivity}
| Setting | Positives | LightGBM, all features | LightGBM, Binary PSO subset | EA potential | LightGBM minus EA potential |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main: 80+, missing players dropped | 72 of 3,288 | 0.301 [0.205, 0.420] | 0.313 [0.222, 0.435] | 0.327 [0.233, 0.446] | −0.026 [−0.116, 0.059] |
| (a) 80+, missing players counted as 0 | 72 of 4,304 | 0.314 [0.215, 0.440] | 0.319 [0.225, 0.443] | 0.326 [0.225, 0.458] | −0.012 [−0.106, 0.075] |
| (b) 75+, missing players dropped | 336 of 3,288 | 0.599 [0.548, 0.648] | 0.608 [0.553, 0.659] | 0.519 [0.467, 0.573] | **+0.080 [+0.036, +0.126]** |

Counting missing players as negatives changes nothing. The 75+ label changes the answer to H1b: for this less extreme target, LightGBM beats EA's potential rating clearly and significantly. The models' attributes carry information about development that EA's potential does not fully use, but for the very top, the 80+ threshold, EA's judgement, which draws on match performance the card does not show, is as good as anything learned from the card.

## Verdicts on H1a and H1b

**H1a: supported in its weak form.** LightGBM matched the MLP: it led on validation (0.593 against 0.573), and on test the two cannot be told apart (difference −0.006 [−0.090, 0.084]). The random forest scored highest on test, but not significantly so.

**H1b: not supported for the 80+ label, supported for 75+.** No model beats EA's potential rating on the test season for the main label (best difference −0.026 [−0.116, 0.059]); for the 75+ label LightGBM does (+0.080 [+0.036, +0.126]).
