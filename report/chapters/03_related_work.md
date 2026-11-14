# Related work {#sec:related}

## Machine learning for talent identification

Most machine-learning studies of football talent use academy data: physical tests, maturity estimates, coach ratings and match statistics of a few hundred players from one or a few clubs. Tang, Wei and Tan (2026) reviewed 27 such studies across team sports, 13 of them in soccer. They report a wide range of ROC-AUC values (roughly 0.58 to 0.96), frequent leakage (feature selection on the full data set, the same athletes in training and test folds, temporal contamination), rare calibration analysis and almost no external validation. Their recommendations, athlete-level splits, calibration metrics such as the Brier score and a clear separation of tuning and test data, are built into this study.

## FIFA ratings as research data

EA's ratings are attractive because they cover every licensed player, in the same format, every season. They have been used to predict market values (Behravan and Razavi, 2021), to classify players into positions (Al-Asadi and Tasdemir, 2021) and to study development. The closest study to this one is Feldman (2025), who used FIFA ratings from the European Soccer Database (2008 to 2016, 8,770 players) to ask whether players fulfil the potential the game assigned to them. A player counted as an achiever if his final log-scaled achievement ratio reached 95% of his projected potential. With data up to age 21, a random forest reached a ROC-AUC of 0.79 in 5-fold cross-validation, and SHAP analysis ranked growth-trajectory features above static attributes. Feldman's error analysis found the same two failure types this study meets: early promise that does not develop, and late bloomers.

TalentScout differs from Feldman's study in four ways. The target is an absolute level (80+ overall within three versions) rather than fulfilling EA's own forecast, so EA's potential becomes a baseline to beat instead of part of the label. The data are newer and larger (FIFA 15 to FC 24). Evaluation uses a temporal split with players grouped and a test season never used for any decision, instead of random folds. And the core of the study is the comparison of feature-selection optimizers, which Feldman does not attempt.

## Evolutionary and swarm feature selection

Xue et al. (2016) surveyed evolutionary computation for feature selection. GAs were the first EC technique applied to it (Yang and Honavar, 1998) and have a natural bit-string representation; PSO, in continuous and binary form, became the most studied method after 2007. The survey lists open issues that this study measures directly: high computational cost (each evaluation trains a model), the **stability** of the selected features across runs, and **feature selection bias** when the same data choose and evaluate a subset. On the binary GWO, Emary, Zawbaa and Hassanien (2016) proposed two variants and compared them with PSO and GA on 18 UCI data sets, using a weighted fitness of classification error and subset size. Many later variants exist, and comparisons in this literature usually use small benchmark data sets, a k-nearest-neighbour classifier and each author's own parameter choices.

## Comparing optimizers fairly

Derrac et al. (2011) set out how to compare evolutionary and swarm algorithms: many independent runs, nonparametric tests, multiple-comparison corrections. Equal budgets measured in fitness evaluations, rather than iterations, are needed because algorithms spend different numbers of evaluations per iteration. This study counts every call to the fitness function, including repeats of an already-evaluated subset, and gives every algorithm the same starting population for each seed.

## The gap this study fills

Searches of the literature (documented in `docs/literature/research_gap.md`) found PSO used for feature selection on FIFA data once, by Behravan and Razavi (2021) for transfer-value regression, and no study comparing GA, Binary PSO and Binary GWO for talent identification. To our knowledge this is the first such comparison, and the first to test the selected subsets on a later, untouched season under a player-grouped split.
