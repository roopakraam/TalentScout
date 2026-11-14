# Conclusion and future work {#sec:conclusion}

This project asked two questions of ten seasons of FIFA and EA Sports FC ratings: can a model predict which young players reach an 80+ rating within three versions, and can nature-inspired optimizers find a much smaller set of attributes that predicts it as well as all 45?

On the optimization question, the three metaheuristics from the course did their job. Under an equal budget of 600 evaluations, the Genetic Algorithm, Binary PSO and Binary Grey Wolf Optimizer each beat random search on all ten seeds. They differ in how they search more than in what they find: Binary GWO converges fastest and keeps the fewest features, Binary PSO keeps exploring, and the GA flattens earliest. A toy problem with a known answer showed that the inertia schedule borrowed from continuous PSO stops binary PSO from converging, a check worth making before any comparison of binary swarm algorithms.

On the prediction question, the honest answer comes from the untouched test season. Every model, with or without feature selection, reaches a PR-AUC of about 0.30, roughly fourteen times better than chance and no better than EA's own potential rating for the 80+ target, although clearly better for a 75+ target. Subsets of 16 to 21 attributes predict as well as all 45, so a scouting checklist can be cut to half or less of the card. The large gains the subsets showed on validation, however, did not transfer. The validation seasons were made easier than the test season by the way the split removed test players, and selecting subsets on any small data set, validation or cross-validation folds, fits that data's noise.

The lessons reach beyond football. Wrapper feature selection with a powerful optimizer will always find a subset that looks better on the data it is shown; with a few hundred positive cases, that improvement is mostly noise. Only data never used for any choice can tell the difference, and the validation set must resemble the test set, which a grouping rule can quietly prevent.

## Future work

* **True multi-objective search.** NSGA-II (Deb et al., 2002) would return the whole accuracy and size front in one run instead of a weighted sum per α.
* **Stability-aware selection.** Penalising subsets that change between bootstrap samples, or selecting by how often a feature is chosen across runs, could produce a checklist that is both small and reproducible.
* **A hybrid GA-PSO selector**, the course's Unit VI stretch goal, combining GA crossover with the PSO update.
* **Better validation.** Tune and select with grouped cross-validation over several training seasons, and test on more than one later season.
* **Richer data.** Match statistics (for example from FBref), playing time and injury records would add exactly the information the false alarm and the missed star in @sec:results-interpretation lacked.
