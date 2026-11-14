# Results III: interpretation (RQ3) {#sec:results-interpretation}

## What the full model relies on

@fig:shap shows the SHAP values of the tuned all-feature LightGBM on the validation set. Age has the largest mean effect: being younger raises the predicted chance of becoming elite within three versions. Next come reactions, free-kick accuracy, current overall and sprint speed. Reactions is EA's measure of how quickly a player responds to play, and it rises with a player's level (Spearman correlation 0.73 with current overall on the training rows), so the model has two partly overlapping views of the player's present quality.

![SHAP values of the tuned all-feature LightGBM on the validation set, top 15 of 45 features. Each dot is a player; colour shows the feature value.](results/figures/fig06a_shap_beeswarm.png){#fig:shap width=5.8}

## Selection frequency against SHAP rank

Across the 30 main runs (10 per optimizer), @fig:selection counts how often each feature appears in a run's best subset, with the features in SHAP order. The agreement between the two views of importance is weak. The Spearman correlation between mean |SHAP| and selection count is 0.31 (p = 0.037), and only three of SHAP's top ten are among the ten most selected features (age, heading accuracy and height).

![How many of the 30 runs (10 per optimizer, α = 0.99) kept each feature, with features in SHAP order (rank in brackets).](results/figures/fig06b_selection_vs_shap.png){#fig:selection width=4.9}

The clearest disagreement is reactions: second in SHAP importance, kept in 1 of 30 runs. Marking awareness, tenth by SHAP, was also kept once. The two methods answer different questions. SHAP measures how much the full model *uses* a feature, while a wrapper keeps a feature only if the other kept features do not already carry its information. Current overall is largely predictable from the other detailed attributes and the position group (a linear model explains 82% of its variance on validation), so a subset that keeps enough of those attributes can drop overall, and reactions with it, at little cost.

## H3

**H3: partly supported.** Age is kept in all 30 runs, by every optimizer. Current overall is not: only 15 of 30 subsets contain it (6 of the GA's ten, 3 of Binary PSO's, 6 of Binary GWO's), for the redundancy reason above. The second half of the hypothesis holds. Beyond age, the optimizers agree on few attributes (height in 29 of 30 subsets, volleys in 26, balance in 25) and disagree on most of the skill attributes, as the low Jaccard similarities in @tab:stability already showed.

## Three players from the test season

Three test players were chosen by rule from the tuned all-feature LightGBM's test scores, so the examples are not hand-picked: the highest-scored player who became elite, the lowest-scored player who became elite, and the highest-scored player who did not (@tab:cases).

Table: The three case-study players (FIFA 21 cards; rank among 3,288 test players). {#tab:cases}
| Case | Player (club) | Age | Overall | EA potential | Next three versions | Model rank |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Correct star | S. Tonali (AC Milan) | 20 | 77 | 91 | 77, 84, 86 | 1 |
| Missed star | R. Kolo Muani (Nantes) | 21 | 64 | 73 | 74, 78, 84 | 2,412 |
| False alarm | D. Zagadou (Borussia Dortmund) | 21 | 79 | 86 | 78, 77, 74 | 2 |

**Sandro Tonali**, a 20-year-old defensive midfielder rated 77, was ranked first and went on to 84 and 86. The model's reasons (@fig:case-star) are the ones a scout would give: an overall already at 77, and free-kick accuracy, reactions, interceptions and vision in the mid-70s for his age.

![SHAP explanation for the correct star (log-odds; E[f(X)] is the average model output).](results/figures/fig06c_case_correct_star.png){#fig:case-star width=4.9}

**Randal Kolo Muani**, a 21-year-old striker rated 64, was ranked 2,412th, yet he rose to 74, 78 and finally 84. At 21, rated 64 and unchanged from the previous year, nothing on his card pointed to that rise; his height is the only feature in his favour (@fig:case-missed), and EA's own potential of 73 missed him too. This is the late-bloomer failure Feldman (2025) also found: development that happens after the observed card cannot be read from it.

![SHAP explanation for the missed star.](results/figures/fig06d_case_missed_star.png){#fig:case-missed width=4.9}

**Dan-Axel Zagadou**, a 21-year-old centre-back rated 79 at Borussia Dortmund, was ranked second but declined to 78, 77 and 74. The model's case looked strong (@fig:case-false): one point below the threshold, with high reactions, interceptions, sliding tackle and heading accuracy for his position. Ratings can fall for reasons a game card does not record, such as injuries or lost playing time, and a model built from the card cannot see them.

![SHAP explanation for the false alarm.](results/figures/fig06e_case_false_alarm.png){#fig:case-false width=4.9}

The three cases sum up the limits of the approach. The model is good at recognising players who are already close to elite level and still young; it cannot foresee late development or decline, because nothing in a single launch-day card records why either happens.
