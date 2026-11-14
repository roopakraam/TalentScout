# Discussion and limitations {#sec:discussion}

## What the optimizers achieved

As optimization algorithms, all three metaheuristics did what they should. Under an equal budget of 600 evaluations they beat random search on every seed, the strongest result ten paired runs can give. They also reached that point quickly, overtaking random search's final value within about 85 evaluations. The differences among them are differences of search behaviour more than of outcome. Binary GWO converges hardest: its pack collapses onto almost identical solutions, it revisits subsets instead of training new models, and it ends with the smallest subsets and the most consistent fitness across seeds. Binary PSO with an undamped velocity keeps exploring to the end. The GA sits between them and flattens earliest. With ten seeds none of these differences in final fitness is significant, which is itself a useful result for the course: on this problem the choice among the three standard metaheuristics matters less than giving any of them a fair budget.

The study also produced a concrete lesson about porting a continuous algorithm to binary search spaces. The inertia schedule recommended for continuous PSO, a weight falling from 0.9 to 0.4, quietly turns binary PSO into something close to random search late in the run, because a decaying velocity means a bit drifting back to a coin flip. A toy problem with a known optimum exposed this in seconds. Without that check, the comparison would have reported Binary PSO as the weakest algorithm for a reason that has nothing to do with the algorithm itself.

## Why the validation gains did not survive

On validation, the selected subsets looked like a clear success: 16 to 25 attributes scoring 0.61 to 0.63 PR-AUC against 0.50 for all 45, and even ahead of the tuned all-feature model. On the test season every configuration scores about 0.30 and none differs significantly from the others. Three effects combine.

**The validation set is easier than the test set.** EA's potential rating involves no fitting at all, yet its PR-AUC falls from 0.42 on validation to 0.33 on test. The planned split removes every test player from validation, and in this data that removes exactly the players who were still young and still below 80 in FIFA 21: the slow developers. The rows that remained were older and closer to the threshold (63% of their positives were already rated 75 or more, against 46% in test), and EA potential scores 0.42 on them but only 0.21 on the rows removed (@tab:diagnostic). A model tuned and a subset selected on such a set are optimised for an easier problem than the one they are tested on.

**Selection on validation fits its noise.** With 113 positives, the validation PR-AUC of a subset carries noise of a few hundredths, and an optimizer that compares hundreds of subsets on it will favour those that happen to fit that noise (Ambroise and McLachlan, 2002; Cawley and Talbot, 2010). The selected subsets lose about 0.05 more PR-AUC between validation and test than the all-feature model does, which is the size of this selection bias here. The plan's fallback shows that the cure is not simply to select on other data. Subsets chosen by grouped cross-validation on the training seasons fitted that criterion just as well (0.37 against 0.32 for all features) and did no better on test (0.280, against 0.295 for the validation-chosen subsets and 0.312 for all 45 features). With 113 to 344 positives, the differences an optimizer chases are as large as the noise.

**Tuning on validation does the same, to a smaller degree.** The all-feature LightGBM was itself chosen as the best of 50 configurations on validation, and its gap (0.29) is larger than the random forest's (0.19), whose tuning gained less on validation.

The practical conclusion for the method is simple: in a study like this, the test season must never be used for any choice, and the numbers that matter are the test numbers. Validation scores, however carefully obtained, overstated the benefit of feature selection by an order of magnitude.

## What a scout can take from it

The test results still say something useful. Compact subsets lost nothing measurable: the Binary GWO subset of 16 attributes and the Binary PSO subset of 21 score the same as all 45 within the precision of the test set, and the α sweep shows that on validation the accuracy curve is flat beyond about nine to twelve attributes. A scouting checklist built around age, height, current level and a handful of technical attributes is as informative as the full card. At the same time, no model or subset beat EA's own potential rating for the 80+ target on the test season. EA's raters see more than the card shows: matches, form, injuries, playing time. For the very top of the distribution that judgement is hard to improve on with the card's attributes alone. For the less extreme 75+ target, the models do beat EA potential clearly (test PR-AUC 0.60 against 0.52), so the attributes carry real information about development that the potential rating does not fully use.

## Limitations

* **Game ratings are not match data.** Both features and labels are EA's ratings, which reflect reputation as well as performance. The label measures what EA's raters think, three seasons later, and the 75+ check only partly addresses this circularity.
* **The validation design.** Removing test players from validation made validation easier than test and unrepresentative of it. Grouped cross-validation on the training seasons gave far more honest estimates (0.32 against 0.31 on test for all features), so a future version should tune and estimate with it; the fallback showed, however, that it does not make subset selection itself more reliable.
* **Small numbers of positives.** With 72 positives in the test season, the 95% intervals of PR-AUC are about ±0.10 wide, so differences smaller than about 0.05 cannot be detected.
* **One test season.** All conclusions about generalisation rest on FIFA 21 and the following three versions, a period that includes the disrupted 2020 to 2022 seasons.
* **Outfield players in licensed leagues only.** Goalkeepers were excluded, and players outside EA's licensed leagues disappear from the data, which is why 22.6% of eligible rows had no label.
* **Fixed settings inside the fitness.** The optimizers searched with a fast, untuned LightGBM; a subset that is best under those settings need not be best under tuned ones.
* **Limited sweep and seeds.** The α sweep covers Binary GWO only, and the fallback used four seeds per optimizer.
