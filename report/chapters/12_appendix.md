# Appendix

## A. The final feature subsets

Chosen on validation by the rules in @sec:final before the test set was used. Optimizer subsets come from the run with the best validation fitness (GA seed 9, Binary PSO seed 6, Binary GWO seed 7).

| Configuration | Features | Attributes |
| --- | ---: | --- |
| GA | 20 | Overall (current), Age, Height, Position group, Shooting (face), Crossing, Heading accuracy, Short passing, Volleys, Curve, Long passing, Ball control, Sprint speed, Balance, Shot power, Jumping, Interceptions, Vision, Standing tackle, Sliding tackle |
| Binary PSO | 21 | Age, Height, Preferred foot, Weak foot, Position group, Overall change (1 yr), Shooting (face), Passing (face), Crossing, Finishing, Short passing, Volleys, Long passing, Ball control, Sprint speed, Balance, Interceptions, Vision, Penalties, Composure, Sliding tackle |
| Binary GWO | 16 | Age, Height, Preferred foot, Overall change (1 yr), Pace (face), Passing (face), Volleys, FK accuracy, Ball control, Agility, Shot power, Jumping, Vision, Penalties, Composure, Sliding tackle |
| SHAP top-20 | 20 | Age, Reactions, FK accuracy, Overall (current), Sprint speed, Short passing, Heading accuracy, Height, Positioning, Marking awareness, Jumping, Interceptions, Pace (face), Volleys, Passing (face), Finishing, Composure, Balance, Aggression, Long shots |

## B. Tuned hyperparameters

Best trial of each 50-trial TPE study (seed 0), maximising validation PR-AUC.

| Model | Hyperparameters |
| --- | --- |
| Logistic regression | C = 0.003186 |
| Random forest (500 trees) | max_depth = 17; min_samples_leaf = 10; max_features = 0.1603 |
| MLP | hidden = 113; n_layers = 2; dropout = 0.4721; learning_rate = 0.001671; weight_decay = 1.392e-06 |
| LightGBM, all 45 features | num_leaves = 248; learning_rate = 0.09906; n_estimators = 962; min_child_samples = 67; subsample = 0.9841; colsample_bytree = 0.5879; reg_lambda = 4.941 |
| LightGBM, GA subset | num_leaves = 206; learning_rate = 0.03511; n_estimators = 277; min_child_samples = 58; subsample = 0.7105; colsample_bytree = 0.7369; reg_lambda = 0.0533 |
| LightGBM, Binary PSO subset | num_leaves = 172; learning_rate = 0.07256; n_estimators = 822; min_child_samples = 10; subsample = 0.8486; colsample_bytree = 0.6769; reg_lambda = 1.871 |
| LightGBM, Binary GWO subset | num_leaves = 195; learning_rate = 0.07932; n_estimators = 188; min_child_samples = 54; subsample = 0.877; colsample_bytree = 0.9953; reg_lambda = 0.004817 |
| LightGBM, SHAP top-20 | num_leaves = 144; learning_rate = 0.1182; n_estimators = 222; min_child_samples = 70; subsample = 0.7219; colsample_bytree = 0.8284; reg_lambda = 0.07977 |

## C. Every candidate feature

SHAP rank of the tuned all-feature LightGBM on validation, how many of the 30 main runs kept the feature, and whether it is in each final subset.

| Feature | Group | SHAP rank | Kept (of 30) | GA | BPSO | BGWO | SHAP top-k |
| --- | --- | ---: | ---: | :---: | :---: | :---: | :---: |
| Age | Profile | 1 | 30 | ✓ | ✓ | ✓ | ✓ |
| Reactions | Movement | 2 | 1 |  |  |  | ✓ |
| FK accuracy | Skill | 3 | 15 |  |  | ✓ | ✓ |
| Overall (current) | Rating | 4 | 15 | ✓ |  |  | ✓ |
| Sprint speed | Movement | 5 | 10 | ✓ | ✓ |  | ✓ |
| Short passing | Attacking | 6 | 18 | ✓ | ✓ |  | ✓ |
| Heading accuracy | Attacking | 7 | 22 | ✓ |  |  | ✓ |
| Height | Profile | 8 | 29 | ✓ | ✓ | ✓ | ✓ |
| Positioning | Mentality | 9 | 11 |  |  |  | ✓ |
| Marking awareness | Defending | 10 | 1 |  |  |  | ✓ |
| Jumping | Power | 11 | 16 | ✓ |  | ✓ | ✓ |
| Interceptions | Mentality | 12 | 23 | ✓ | ✓ |  | ✓ |
| Pace (face) | Face stats | 13 | 22 |  |  | ✓ | ✓ |
| Volleys | Attacking | 14 | 26 | ✓ | ✓ | ✓ | ✓ |
| Passing (face) | Face stats | 15 | 14 |  | ✓ | ✓ | ✓ |
| Finishing | Attacking | 16 | 16 |  | ✓ |  | ✓ |
| Composure | Mentality | 17 | 15 |  | ✓ | ✓ | ✓ |
| Balance | Movement | 18 | 25 | ✓ | ✓ |  | ✓ |
| Aggression | Mentality | 19 | 3 |  |  |  | ✓ |
| Long shots | Power | 20 | 4 |  |  |  | ✓ |
| Ball control | Skill | 21 | 13 | ✓ | ✓ | ✓ |  |
| Vision | Mentality | 22 | 19 | ✓ | ✓ | ✓ |  |
| Sliding tackle | Defending | 23 | 19 | ✓ | ✓ | ✓ |  |
| Agility | Movement | 24 | 9 |  |  | ✓ |  |
| Stamina | Power | 25 | 1 |  |  |  |  |
| Standing tackle | Defending | 26 | 15 | ✓ |  |  |  |
| Strength | Power | 27 | 15 |  |  |  |  |
| Overall change (1 yr) | Profile | 28 | 10 |  | ✓ | ✓ |  |
| Shot power | Power | 29 | 13 | ✓ |  | ✓ |  |
| Crossing | Attacking | 30 | 15 | ✓ | ✓ |  |  |
| Weight | Profile | 31 | 3 |  |  |  |  |
| Acceleration | Movement | 32 | 22 |  |  |  |  |
| Shooting (face) | Face stats | 33 | 16 | ✓ | ✓ |  |  |
| Long passing | Skill | 34 | 12 | ✓ | ✓ |  |  |
| Penalties | Mentality | 35 | 23 |  | ✓ | ✓ |  |
| Curve | Skill | 36 | 11 | ✓ |  |  |  |
| League level | Profile | 37 | 8 |  |  |  |  |
| Skill moves | Profile | 38 | 5 |  |  |  |  |
| Dribbling (face) | Face stats | 39 | 10 |  |  |  |  |
| Physic (face) | Face stats | 40 | 6 |  |  |  |  |
| Preferred foot | Profile | 41 | 15 |  | ✓ | ✓ |  |
| Defending (face) | Face stats | 42 | 7 |  |  |  |  |
| Dribbling | Skill | 43 | 10 |  |  |  |  |
| Weak foot | Profile | 44 | 8 |  | ✓ |  |  |
| Position group | Profile | 45 | 15 | ✓ | ✓ |  |  |

## D. Reproducing the results

The repository README gives the full instructions. In short, with the `tf-gpu-env` environment active:

1. `python scripts/download_data.py` downloads the Kaggle data (no account needed).
2. `bash scripts/reproduce_all.sh --clean` reruns every step in order: notebook 01, the 13 tuning studies, notebooks 02 and 03, the 40 feature-selection runs with the ablation and the α sweep, notebook 04, the final tuning and the fallback runs, notebooks 05 and 06, and this report. It takes about three hours on the project laptop.
3. Every run and study writes its own file and is skipped if the file exists, so the script can be stopped and restarted.

Every decision that changed data, settings or results is recorded with its reason in `docs/decision_log.md`.
