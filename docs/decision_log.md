# Decision log

Every choice that changes data, settings or results, with the reason and what
to do if it turns out wrong. The Methods chapter is written from this file.
The same entries are mirrored in the project tracker.

Format: **what** was decided, **why**, and the **fallback** if it is wrong.

---

## 2026-11-13

**D01. Following plan v2 (13 Nov).** Label: 80+ overall within 3 versions, age 21 or under.
*Why:* the design is fixed before any data is modelled.
*Fallback:* plan section 9 (threshold 78 or 75, or age 23 and under), to be decided in week 1 before any model result.

**D02. Environment: the existing `tf-gpu-env` virtual environment (Python 3.12.16), not a new conda environment.**
Added lightgbm 4.7.0, optuna 5.0.0, shap 0.54.1, seaborn 0.13.2, statsmodels 0.15.0 and pyarrow 26.0.1; nothing already installed was changed.
*Why:* the environment already has a working TensorFlow-GPU stack for the RTX 3050, and a second CUDA stack would risk breaking it.
*Fallback:* `requirements.txt` pins the versions, so a clean environment can be built from it.

**D03. The MLP is written in TensorFlow/Keras and trained on the GPU; LightGBM, the random forest and logistic regression run on the CPU.**
*Why:* the plan allows "PyTorch or scikit-learn" for the MLP; Keras is the GPU framework this environment is built around, and it supports the same weighted loss and early stopping. LightGBM's GPU learners are not usable here: the pip wheel has no CUDA tree learner, WSL2 exposes no OpenCL device, and there is no compiler to rebuild it. With about 12,000 training rows, CPU histogram training takes well under a second anyway.
*Fallback:* the MLP class has a scikit-learn style interface, so a PyTorch version could replace it without touching the notebooks.

**D04. LightGBM needs the OpenMP runtime `libgomp.so.1`, which is not installed system-wide.**
The Ubuntu package `libgomp1` was unpacked into `tf-gpu-env/lib/openmp` and that folder added to `LD_LIBRARY_PATH` in the environment's `activate` script.
*Why:* no sudo access on this machine.
*Fallback:* `sudo apt install libgomp1` makes the workaround unnecessary.

**D05. Data source: Kaggle "EA Sports FC 24 Complete Player Dataset", `male_players.csv`, downloaded with `scripts/download_data.py`.**
The file has 180,021 rows and 109 columns covering FIFA 15 to FC 24, as the plan assumed.
*Why:* public dataset, no login needed, the download is scripted so it can be repeated.
*Fallback:* none needed.

### Week 1 decisions: data, label, split, features (notebook 01)

**D06. The launch-snapshot rule is kept in the code but removes nothing.**
Every version in the file has exactly one update (`fifa_update = 2`), dated between 21 August and 26 September, which is launch time. 180,021 rows before and after; no duplicate (player_id, fifa_version) pairs.
*Why:* the rule protects the pipeline if a file with several updates per version is ever used.
*Fallback:* none needed.

**D07. Future ratings and the one-year change in overall are looked up on the complete table, before goalkeepers and ineligible rows are removed.**
*Why:* 16 players are listed as goalkeeper in some versions and outfield in others; looking ratings up first means a row's label never depends on which rows were filtered out. 20,025 goalkeeper rows (11.1%) are then removed, leaving 159,996 outfield rows.
*Fallback:* none needed; the label was cross-checked against an independent pivot-table implementation on all 28,817 eligible rows and against seven hand-checked players.

**D08. Label threshold 80 and age limit 21 are final. No fallback from plan section 9 is used.**
*Why:* the train versions (FIFA 15 to 18) hold 344 positives, more than twice the 150 the plan requires. Decided before any model was fitted.
*Fallback:* none; changing it now would mean choosing a label after seeing results.

**D09. Rows whose label is unknown are dropped from the main analysis: 6,502 of 28,817 eligible rows (22.6%).**
These players appear in none of the next three versions. They stay in `labelled.parquet` with `set = "unlabelled"`.
*Why:* a missing player has usually left the licensed leagues, not football (Sebastián Driussi vanishes for FIFA 19 to 21 because the Russian league was not in the game, then returns), so calling him a failure would be wrong.
*Fallback:* the sensitivity check in notebook 05 reruns the analysis counting these rows as negatives.

**D10. Split: FIFA 15-18 train, 19-20 validation, 21 test; every test player is removed from train and validation (2,905 rows removed).**
Result: 12,148 train rows (344 positives), 3,974 validation rows (113), 3,288 test rows (72); positive rate 2.73%.
Two side effects are accepted and will be reported as limitations:
(a) 1,562 of the 3,099 validation players (50%) also appear in train, in older versions, which flatters validation scores;
(b) removing test players changes who is left in validation: the remaining rows are players who by FIFA 21 had turned 22, reached 80 or left the game, so validation is older than test (50% aged 21, against 29% in test) and under-represents slow developers.
*Why:* this is the rule in plan section 5; it guarantees that no test player is ever seen in training or selection.
*Fallback:* the validation-to-test gap table (notebook 05) measures the effect; plan section 9 offers a fitness based on 3-fold grouped cross-validation on train as an extra run.

**D11. Label windows overlap the later sets in time.** Train labels use ratings up to FIFA 21 and validation labels up to FC 23, while test features are from FIFA 21.
*Why:* unavoidable with a three-version label window and ten versions of data. No test player is in train or validation, so there is no player-level leakage, but tuning on validation does see how ratings evolved up to FC 23.
*Fallback:* stated as a limitation in the report.

**D12. Leakage: 53 columns dropped, 5 set aside.**
Dropped: `player_url`, `fifa_update`, `update_as_of`, `dob`, all club and national-team fields (`club_team_id`, `league_id`, `club_position`, `club_jersey_number`, `club_loaned_from`, `club_joined_date`, `club_contract_valid_until_year`, `nationality_id`, `nation_team_id`, `nation_position`, `nation_jersey_number`), `work_rate`, `body_type`, `real_face`, `player_tags`, `player_traits`, the 27 position-rating strings (`ls` ... `gk`) and the 6 `goalkeeping_*` attributes.
Set aside (never candidate features): `potential` (kept as the baseline score), `value_eur`, `wage_eur`, `release_clause_eur`, `international_reputation`.
Kept for bookkeeping only: `player_id`, `fifa_version`, names, positions, club, league and nationality names.
*Why:* plan section 5. Position ratings are computed from overall, money and reputation follow from how good a player already is, and `potential` is EA's own forecast of the label.
*Fallback:* none needed.

**D13. Candidate feature pool: D = 45, so 2^45 (about 3.5 x 10^13) subsets.**
1 rating (`overall`), 7 profile attributes, 2 engineered (`position_group`, `overall_change`), 6 face stats and 29 detailed skills. The exact ordered list is `FEATURES` in `src/features.py`.
- **Current `overall` is included.** The list in plan section 5 omits it, but hypothesis H3 expects every optimizer to keep it and the plan's summary speaks of "all 45" attributes. It is known at version t, so it is not leakage.
- **Position group is one ordinal column** (DEF 0, MID 1, FWD 2) from the first listed position. One-hot encoding would spend three bits on one attribute.
- **`overall_change`** = overall(t) - overall(t-1), empty if the player was not in version t-1.
- `preferred_foot` is coded Left 0, Right 1.
*Why:* freezing the list means every optimizer searches exactly the same space.
*Fallback:* if `overall` turns out to dominate everything, the trade-off sweep and the SHAP analysis will show it and the report will say so.

**D14. Missing values are left as NaN for LightGBM and filled with the train median for logistic regression, the random forest and the MLP.**
Only three features have gaps: `league_level` (under 1.1%), `overall_change` (60% of train, 44% of validation, 50% of test) and `mentality_composure` (52% of train, 0% elsewhere).
Two of these shift between sets for reasons that have nothing to do with football: `overall_change` is always empty in FIFA 15 because there is no FIFA 14 in the data, and composure was only added to the game in FIFA 17.
*Why:* imputing inside the pipeline keeps validation and test information out of training.
*Fallback:* if the optimizers lean on either feature in a way that looks like an artefact, it will be discussed with the selection-frequency results.

## 2026-11-14

**D15. Research-gap statement narrowed (T1.12).**
The searches found one study that used PSO for feature selection on FIFA data: Behravan and Razavi (2021, *Soft Computing* 25:2499–2511), for transfer-value regression on FIFA 20. No study compares GA, BPSO and BGWO for talent identification or evaluates selection under a temporal, player-grouped split. The gap statement in `docs/literature/research_gap.md` now cites that study and claims only the narrower gap.
*Why:* an overclaimed gap is the first thing a reviewer checks.
*Fallback:* if the manual Google Scholar pass finds a closer study, cite it and narrow the claim again.

### Week 2 decisions: models and tuning (notebook 02)

**D16. Tuning protocol (T2.6, T3.1, T3.2).** Optuna 5.0, 50 trials per study, maximising validation PR-AUC. The sampler seed varies (0 to 4 in the LightGBM comparison) while the model seed stays at 0, so two studies differ only in the hyperparameters they proposed. Search spaces as in the tracker; the random forest keeps 500 trees and takes `max_features` as a fraction (0.1 to 1); the MLP uses batch 256, at most 200 epochs, AdamW, and early stopping on validation PR-AUC with patience 10.
The final configuration of every model is the best trial of its **seed-0 TPE study**. For LightGBM that is 0.5932, not the best of all ten LightGBM studies (0.6008).
*Why:* every model then gets exactly 50 trials; choosing the best of 500 trials would overfit the validation set.
*Fallback:* none needed; the other four TPE studies are reported only in the BO-vs-random comparison.

**D17. Fitness function rules (T3.3, T3.4).** LightGBM with fixed settings (200 trees, learning rate 0.05, other parameters at library defaults), `scale_pos_weight` = 34.3, seed 0, 4 threads, `deterministic=True`. The cache stores validation PR-AUC by subset, so runs with another alpha can reuse it. **Every call counts toward the budget of 600, cache hits included.** An all-zero vector is repaired by switching on one random feature.
Timing: 0.23 s per evaluation (20 random subsets; 0.28 s while other jobs were running), far below the 3-second fallback trigger, so the budget of 600 and the population of 20 stay.
*Why:* equal "looks" per algorithm is the fairest budget rule; the alternative (counting only real trainings) would reward algorithms that revisit subsets.
*Fallback:* none needed.

**D18. With fixed settings, all 45 features score a validation PR-AUC of 0.499**, below the tuned model (0.593) and below several random subsets (up to 0.53). Current overall alone scores 0.471.
*Why it matters:* some features hurt when a model trained on FIFA 15–18 is applied to FIFA 19–20, so feature selection can raise accuracy, and the H2a comparison must be like-for-like (subset and all features under the same fixed settings).

### Weeks 3-4 decisions: optimizers (src/optimizers, tests, notebook 03)

**D19. One interface and one source of randomness per role (T3.5).** Every optimizer is `run(fitness, D, init_pop, budget, rng)`. The starting population comes from `default_rng(seed)` (each bit on with probability 0.5, all-zero rows repaired) and is identical for all four methods; each method's own random draws come from a separate stream, `default_rng(10000 + seed)`. Diversity is the mean pairwise Hamming distance of the population after each iteration. An `Evaluator` counts calls, stops at the budget and records the best-so-far fitness after every call.
*Why:* the comparison is fair by construction, and `tests/test_optimizers.py` checks budget, shared start, monotone history and reproducibility for all four.

**D20. GA details (T3.6).** The 2 elites keep their known fitness and are not re-evaluated, so a generation costs 18 calls; tournaments draw 3 distinct individuals; uniform crossover produces two complementary children; bit-flip mutation 1/45. With 600 calls the GA runs 32 full generations plus a partial 33rd of 4 calls.
*Why:* re-evaluating elites would spend budget on known answers.

**D21. Binary PSO uses inertia w = 1 (no damping), not the plan's 0.9 to 0.4 (T4.1).** On the toy problem (minimise the number of selected features, D = 45, 600 calls) the falling weight never converges: best 7.1 features on average against an optimum of 1, and the swarm stays almost as scattered as a random population (final diversity 20.6; a random population has 22.5). In binary PSO a velocity of 0 means "bit is 1 with probability 0.5"; when a bit already agrees with both bests the update is v = w·v, so any w < 1 pulls it back to a coin flip. With w = 1, the original algorithm of Kennedy and Eberhart (1997), the swarm reaches the optimum on 10 of 10 seeds. c1 = c2 = 2, clamp ±6 and the sigmoid transfer are unchanged.
Decided on the toy problem, before any football run.
*Fallback:* the plan's schedule is kept as the ablation `bpso_w_dec` and run on the football data with the same 10 seeds.

**D22. Binary GWO details (T4.2).** The bGWO2 update with transfer 1/(1 + e^(−10(x − 0.5))) and a stochastic threshold; `a` falls linearly from 2 to 0 over the 29 iterations. Alpha, beta and delta are the three best *distinct* solutions found so far (a small archive), so the three leaders can never be the same vector.
*Why:* with duplicates the pack follows one leader three times and loses the averaging that defines GWO.

**D23. Random search control (T4.3)** evaluates the shared starting population, then batches of 20 random vectors (each bit on with probability 0.5, the same distribution as the start) until 600 calls; no population, so no diversity curve.

**D24. Sanity checks (T3.7, T4.4).** Toy problem with 10 seeds; Breast Cancer Wisconsin with 5 seeds (the tracker suggested 3) and a budget of 300. The Breast Cancer fitness uses a standardised logistic regression with 5-fold cross-validated accuracy (all 30 features: 0.979). GA, Binary PSO (w = 1) and Binary GWO match or beat random search on every seed and every problem.

**D25. Experiment runner (T4.5).** `run_experiments.py` writes one CSV (every call: subset, PR-AUC, fitness, best so far) and one JSON (best subset, fitness, PR-AUC, size, wall-clock time, trainings, diversity) per algorithm and seed. Files are written to a temporary name and renamed, JSON last, so a finished run is exactly a run whose JSON exists. Jobs are ordered seed by seed, so the four methods of one seed run side by side under the same machine load; 4 workers × 4 LightGBM threads use the 16 cores.
Verified on 14 Nov: killed by `timeout` after 240 s, the runner had saved GA seed 0 and left nothing of seed 1; restarted, it reported "2 runs requested, 1 already finished, 1 to do", ran seed 1 only, and both results matched the main runs exactly.

**D26. Main experiment (T4.6).** GA seed 0 was run alone first and inspected (600 rows, no NaN, 524 distinct subsets, best PR-AUC 0.620 with 25 features), then moved aside, and all 40 runs were launched together on 13 Nov; they finished in about 40 minutes. The batch rerun of GA seed 0 reproduced the solo run exactly. The ablation `bpso_w_dec` (10 seeds) ran right after, into `results/ablation_bpso_inertia/`.

### Week 5 decisions: comparing the optimizers (notebook 04)

**D27. H2a is checked like-for-like, per run.** Reference: all 45 features under the fixed fitness settings (validation PR-AUC 0.4993); "within 0.01" means at least 0.4893; "at most a third" means 15 features or fewer. The tuned all-feature model (0.593) is reported only as context.
*Why:* T5.2 asks for the all-feature LightGBM with the same fixed settings; comparing a fixed-settings subset with a tuned model would mix two effects.

**D28. Statistics.** Primary family: best fitness of each run; Friedman test across the four methods, then two-sided Wilcoxon signed-rank tests on the six pairs (paired by seed), Holm-corrected (Derrac et al., 2011). Secondary family, corrected separately: the same six tests on subset size. The inertia ablation is a separate, pre-declared comparison outside both families.
*Why:* keeps the confirmatory test (H2b) clean while still reporting size differences.

**D29. α sweep with Binary GWO, α = 0.95, 0.90, 0.80, seeds 0-4, cache reused.** Binary GWO had the lowest mean fitness, best mean rank (1.6) and smallest spread; its edge over GA and Binary PSO is not significant, which the report states. Reusing cached PR-AUC values changes run time, not results, because PR-AUC does not depend on α. The empirical Pareto front uses every subset evaluated in any run (30,132 distinct subsets).
*Result:* 14 of 15 sweep runs meet H2a; at α = 0.80 Binary GWO keeps 8-12 features with validation PR-AUC 0.620-0.640.
