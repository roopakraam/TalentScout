# Paper notes

Notes for task T1.11. Each paper gets the same four lines (problem, method,
limitation, how TalentScout differs), followed by details worth citing.
Page or section numbers refer to the version I read.

---

## 1. Feldman (2025): the closest football study

J. Feldman, "Can We Really Predict Which Football Players Will Succeed?",
*Proc. 13th Int. Conf. on Sport Sciences Research and Technology Support (icSPORTS 2025)*,
SciTePress, pp. 38–49. doi:10.5220/0013671000003988. Read in full (open access, CC BY-NC-ND 4.0).

| | |
| --- | --- |
| **Problem** | Will a player fulfil the career potential that the FIFA games assigned to him? |
| **Method** | European Soccer Database (Mathien, 2016; FIFA ratings 2008–2016), 8,770 players. Label: "achiever" if the final log-scaled achievement ratio reaches 95% of projected potential. About 40 engineered features (growth metrics, achievement ratios, skill aggregates). XGBoost regression (R² = 0.48) and a random forest classifier, 5-fold cross-validation. |
| **Limitation** | Success is defined through EA's own potential rating; the folds are random rather than temporal, so training and test players overlap in time; the early-career cohort needs at least three ratings before age 21; no feature selection beyond dropping correlated or "leaky" variables. |
| **How TalentScout differs** | Absolute target (80+ overall within three versions) instead of fulfilling EA's forecast, and EA's `potential` becomes the baseline to beat; newer, larger data (FIFA 15 to FC 24); temporal split with players grouped and an untouched test version; PR-AUC for a 2.7% outcome; metaheuristic feature selection as the core contribution. |

Details:

* Early-career model (data before age 21): accuracy 73.95% ± 1.42, F1 0.625, **ROC-AUC 0.79 ± 0.02**, precision 73%, recall 55% (§3.3, Table 7).
* Achievement ratio at time t: log(1 + rating − B) / log(1 + potential − B) with B = 30 (§2.2).
* Cohorts: 2,212 players with complete careers, 4,683 with three or more ratings before 21 (§2.1.2).
* SHAP: growth-trajectory features (growth rate, total growth, years to peak) rank above static skills (§3.2).
* Error analysis: false positives are players with early promise but no growth, false negatives are late bloomers (§3.4). Our hand-checks found the same pattern (Ødegaard at 16 is a late bloomer, label 0).
* Stated limitations (§5): ratings are subjective and influenced by reputation; no injury, psychological or coaching context; not tested on newer cohorts. Future work: objective performance data, newer cohorts.
* Code: github.com/jonfeld/icsports2025.

---

## 2. Xue, Zhang, Browne and Yao (2016): the survey

B. Xue, M. Zhang, W. N. Browne, X. Yao, "A Survey on Evolutionary Computation Approaches to Feature Selection",
*IEEE Transactions on Evolutionary Computation* 20(4):606–626, 2016. doi:10.1109/TEVC.2015.2504420.
Read: introduction, §II.A, §III.A (GAs), §III.C (PSO), §VI (issues and challenges), §VII. Author copy from the Victoria University of Wellington repository.

| | |
| --- | --- |
| **Problem** | Feature selection has a 2^n search space and greedy methods get stuck ("nesting effect"); which evolutionary methods work, and what is still open? |
| **Method** | Survey of GA, GP, PSO, ACO and other EC methods, organised by search technique, evaluation (wrapper, filter, combined) and number of objectives. |
| **Limitation** | Benchmarks are mostly UCI or gene-expression data; it gives no protocol for fair comparisons between optimizers. |
| **How TalentScout differs** | Applies the wrapper approach to a real, temporally structured sports problem, with equal budgets, shared starting populations, 10 seeds and nonparametric tests, and reports the two things §VI flags as neglected: subset stability and selection bias. |

Points to use in the report:

* GAs have "a natural representation of a binary string" (§III.A); PSO uses a bit-string with one bit per feature, binary or thresholded continuous (§III.C).
* Wrappers are more accurate but expensive; most of the cost is in evaluation, which is why our fitness caches every subset (§VI.B).
* **Stability** (§VI.C): stochastic optimizers can reach the same fitness with different features, so stability must cover "the consistency of the selected features", not only fitness. This is our Jaccard analysis (H2c).
* **Selection bias** (§VI.H): using the same data for selection and evaluation inflates results; a separate test set is needed. This is our validation-to-test gap table.
* **Multi-objective** (§VI.F): error and subset size are "not always conflicting"; removing features can also reduce error. Our runs show this directly: the selected subsets score higher on validation than all 45 features.
* GAs "preserve a small set of features" through their operators; PSO is "computationally cheap because of its simple updating mechanisms" (§VII).

---

## 3. Emary, Zawbaa and Hassanien (2016): Binary Grey Wolf Optimizer

E. Emary, H. M. Zawbaa, A. E. Hassanien, "Binary grey wolf optimization approaches for feature selection",
*Neurocomputing* 172:371–381, 2016. doi:10.1016/j.neucom.2015.06.083.
The full text is behind the Elsevier paywall; these notes use the abstract and the publisher record, plus
how later papers describe the method (e.g. Zhang and Cai, 2024, *Biomimetics* 9(10):648).

| | |
| --- | --- |
| **Problem** | GWO (Mirjalili et al., 2014) works on continuous spaces; feature selection needs a binary version. |
| **Method** | Two binary versions, used as wrappers. bGWO1 binarises each of the three moves towards alpha, beta and delta and combines them by stochastic crossover. **bGWO2** averages the three continuous moves and squashes the result with a sigmoid, then thresholds it stochastically. Compared with PSO and GA on 18 UCI datasets. |
| **Limitation** | Small benchmark datasets and a simple classifier; no temporal structure or class imbalance; the comparison with GA and PSO relies on their own parameter choices. |
| **How TalentScout differs** | bGWO2 inside a LightGBM / PR-AUC fitness on an imbalanced (2.7%) real problem, against GA, BPSO and random search under one evaluation budget, with Wilcoxon tests and a held-out test version. |

Implementation details taken from the method as it is commonly described (to be checked against the full text before submission, task T8.4):

* GWO move per leader L: A = 2a·r1 − a, C = 2·r2, D = |C·X_L − X|, X_L' = X_L − A·D; a falls linearly from 2 to 0.
* bGWO2: x_d = 1 if sigmoid(10·(mean of the three moves − 0.5)) ≥ rand, else 0.
* Fitness: α·(error) + β·(selected / total) with β = 1 − α. α = 0.99 is the value used throughout this line of work and in the plan.

---

## Other papers met while checking the gap (not part of the three)

* **Tang, Wei and Tan (2026)**, "The Role of Machine Learning in Talent Identification for Team Sports: A Systematic Review", *J. Sports Sci. Med.* 25:58–83, doi:10.52082/jssm.2026.58. PRISMA review of 27 studies (13 in soccer). 59.3% rated high risk of bias, mostly in the analysis domain; leakage control often inadequate (feature selection on the full data, the same athletes across folds, temporal contamination); calibration rarely reported (recommends the Brier score); external validation rare. No metaheuristic feature selection among the included studies. Supports our grouped temporal split, the Brier score and the untouched test set.
* **Behravan and Razavi (2021)**, "A novel machine learning method for estimating football players' value in the transfer market", *Soft Computing* 25(3):2499–2511, doi:10.1007/s00500-020-05319-3. FIFA 20 data; PSO-based clustering, then PSO for feature selection and SVR tuning; market value, not talent. The closest use of a swarm optimizer on FIFA data.
* **Al-Asadi and Tasdemir (2021)**, "Empirical Comparisons for Combining Balancing and Feature Selection Strategies for Characterizing Football Players Using FIFA Video Game System", *IEEE Access* 9:149266–149286, doi:10.1109/ACCESS.2021.3124931. Nine feature-selection methods and three balancing techniques for predicting a player's position from FIFA attributes; feature selection alone did not beat the baseline. Position classification, not talent.
