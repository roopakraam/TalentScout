# Week 5 results review with the instructor (T5.9)

Bring notebook 04 open, or print these three exhibits:

1. **Summary table:** `results/tables/rq2_summary.csv`
2. **Convergence plot:** `results/figures/fig04a_convergence.png`
3. **Wilcoxon table:** `results/tables/rq2_wilcoxon_holm.csv`

## What to say (three minutes)

* All three metaheuristics beat random search on all 10 seeds (Wilcoxon p = 0.002, Holm 0.012); they do not differ significantly from each other. Binary GWO has the best mean fitness and the smallest spread, and picks significantly fewer features (about 18 against 22).
* At the planned α = 0.99 no run gets down to 15 features. With α = 0.80, Binary GWO keeps 8–12 features and still scores far above the all-feature model on validation.
* The planned Binary PSO inertia schedule (0.9 → 0.4) failed the toy test, because it turns binary PSO into a random search late in the run. The study uses w = 1 and reports the schedule as an ablation.
* Subsets differ a lot between runs (Jaccard about 0.4), so the "checklist" is not unique.
* Feature selection *raised* validation PR-AUC (0.62–0.65 against 0.50). The test set will show how much of that is selection on validation.

## Questions to ask

1. Is replacing the planned inertia schedule with w = 1, plus the ablation, acceptable?
2. Should the α sweep also be run for GA and Binary PSO (about 30 more minutes of compute)?
3. Is the report outline in plan §8 still right for the page limit?

## Record of the meeting

- Date:
- Feedback:
- Changes agreed:
