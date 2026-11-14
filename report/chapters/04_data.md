# Data and label {#sec:data}

## Dataset

The data are the Kaggle "EA Sports FC 24 Complete Player Dataset" (Leone, 2023), table `male_players.csv`: 180,021 rows and 109 columns, one row per player per game version from FIFA 15 (2014) to EA Sports FC 24 (2023). A row is the player's rating card at one moment: current overall rating, EA's `potential`, age, height, weight, preferred foot, six summary ("face") statistics, 29 detailed attributes, goalkeeping attributes, club, league, market value and wage, and text fields such as position ratings and tags. The same `player_id` appears once per version, so careers can be followed. Every version in the file has exactly one update, dated at the game's launch in August or September, so each row is a launch-day snapshot.

## Label

A row at version t is **eligible** if the player is 21 or younger and rated below 80 overall. It is **positive** if he reaches 80 or more in any of the next three versions:

$$y_{i,t} = \mathbf{1}\left[\max_{k \in \{1,2,3\}} \mathrm{overall}_{i,t+k} \geq 80\right], \quad \mathrm{age}_{i,t} \leq 21, \quad \mathrm{overall}_{i,t} < 80$$ {#eq:label}

Three future versions are needed, so feature rows come from FIFA 15 to FIFA 21 only (21 + 3 = 24). A player missing from all three future versions has an unknown label; he has usually left the licensed leagues rather than football. These rows are dropped from the main analysis and counted as negatives in a sensitivity check. Goalkeepers are removed, because their outfield attributes mean little.

The label was implemented twice, once with table joins and once with a pivot table of overall by player and version, and the two agree on all 28,817 eligible rows. Seven players were checked by hand (@tab:handchecks); they include a late bloomer, whose label is 0 because his rise came after the window, and a player whose rating fell.

Table: Hand-checked labels. Overall ratings at version t and the three following versions. {#tab:handchecks}
| Player | Version t | Age | Overall t | t+1 | t+2 | t+3 | y |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K. Mbappé | FIFA 17 | 17 | 71 | 83 | 87 | 89 | 1 |
| E. Haaland | FIFA 19 | 17 | 67 | 73 | 84 | 88 | 1 |
| M. Ødegaard | FIFA 18 | 18 | 74 | 74 | 78 | 83 | 1 |
| M. Ødegaard | FIFA 16 | 16 | 69 | 70 | 74 | 74 | 0 |
| A. Januzaj | FIFA 15 | 19 | 75 | 75 | 76 | 77 | 0 |
| Renato Sanches | FIFA 17 | 18 | 78 | 76 | 75 | 75 | 0 |
| J. Bellingham | FIFA 21 | 17 | 69 | 79 | 84 | 86 | 1 |

Of 180,021 rows, 20,025 are goalkeepers and 28,817 outfield rows are eligible. Of those, 6,502 (22.6%) have no future version, which leaves 22,315 labelled rows with 572 positives (2.6%). The plan's fallback (a lower threshold or a wider age range) was needed only if the training seasons held fewer than 150 positives; they hold 344, so the threshold of 80 and the age limit of 21 were kept, a decision taken before any model was fitted.

![Share of eligible players who reach 80+ within three versions, by current overall rating (left) and by age (right). Labelled rows of FIFA 15 to 21.](results/figures/fig01_positive_rate.png){#fig:positive-rate width=6.0}

@fig:positive-rate shows how strongly the outcome depends on the present. Four in ten players rated 75 to 79 reach 80, against one in 600 of those rated 59 or below, and older players in the eligible range are more likely to cross the line within three years than younger ones.

## Temporal, player-grouped split

The split follows the calendar, as a scout's problem does: older versions train the models, the middle versions tune them, the newest version tests them (@tab:split). Every player who appears in the test set is removed from training and validation (2,905 rows), so a model can never recognise a test player from his earlier cards. A test is part of the code and fails if any test player appears elsewhere.

Table: The temporal, player-grouped split. Rows of test players were removed from training and validation. {#tab:split}
| Set | Feature versions | Label window | Rows | Players | Positives | Positive rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Train | FIFA 15 to 18 | up to FIFA 21 | 12,148 | 7,064 | 344 | 2.83% |
| Validation | FIFA 19 to 20 | up to FC 23 | 3,974 | 3,099 | 113 | 2.84% |
| Test | FIFA 21 | FIFA 22 to FC 24 | 3,288 | 3,288 | 72 | 2.19% |

The plan removes only test players, so half of the validation players (1,562 of 3,099) also appear in training through their older cards. A less visible consequence turned out to matter more. A player eligible in FIFA 21 belongs to the test set, so every validation row of such a player is removed. The validation players who remain are those who, by FIFA 21, had turned 22, had reached 80 or had left the game. Validation is therefore older than test (half of its rows are 21-year-olds, against 29% of test rows) and contains few slow developers. @sec:results-prediction shows what this does to validation scores.

## Leakage control and the candidate features

Fifty-three columns were dropped: identifiers, dates, club and national-team fields, tags and traits, the 27 position-rating strings (computed from overall) and the goalkeeping attributes. Five columns are known at version t but kept out of the candidate pool: `potential`, which is EA's own forecast of the label and serves only as the baseline score, and market value, wage, release clause and international reputation, which follow from how good a player already is. The optimizers therefore choose among skills and physical traits, which is the scouting question.

The candidate pool has **D = 45** features (@tab:features), so the search space has 2^45 ≈ 3.5 × 10^13 subsets. Current overall is included: the plan's list omitted it, but H3 expects every optimizer to keep it, and it is known at version t. Position is one ordinal column (defender, midfielder, forward), and one engineered feature records the change in overall since the previous version.

Table: The candidate feature pool (D = 45). {#tab:features}
| Group | n | Features |
| --- | ---: | --- |
| Rating | 1 | current overall |
| Profile | 9 | age, height, weight, preferred foot, weak foot, skill moves, league level, position group, one-year change in overall |
| Face statistics | 6 | pace, shooting, passing, dribbling, defending, physic |
| Attacking | 5 | crossing, finishing, heading accuracy, short passing, volleys |
| Skill | 5 | dribbling, curve, free-kick accuracy, long passing, ball control |
| Movement | 5 | acceleration, sprint speed, agility, reactions, balance |
| Power | 5 | shot power, jumping, stamina, strength, long shots |
| Mentality | 6 | aggression, interceptions, positioning, vision, penalties, composure |
| Defending | 3 | marking awareness, standing tackle, sliding tackle |

Only three features have missing values: league level (under 1.1%), the change in overall (empty for a player's first appearance, and always empty in FIFA 15 because the data start there) and composure, which EA added in FIFA 17 and is therefore empty in the first two training versions. Missing values are left to LightGBM, which handles them natively, and filled with the training median for the other models.
