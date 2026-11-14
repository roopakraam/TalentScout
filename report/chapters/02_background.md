# Background {#sec:background}

## Talent identification

Talent identification means picking, from a group of young players, those most likely to succeed at a higher level later. Reviews of the field agree on two difficulties (Vaeyens et al., 2008; Sarmento et al., 2018). Early performance is a weak guide because players mature at different ages, and success depends on many factors, physical, technical, psychological and social, that are measured at a single moment. Machine learning has been applied to the problem in many team sports, but a recent systematic review of 27 studies rated 59% of them at high risk of bias, mostly for leakage between training and test data, missing calibration and the absence of external validation (Tang, Wei and Tan, 2026). Those three weaknesses shaped the design of this study.

## Feature selection as an optimization problem

Feature selection chooses a subset of the available input variables. Three families exist (Xue et al., 2016). **Filter** methods score each feature by a statistic such as mutual information and keep the best ones; they are fast but ignore how features work together. **Embedded** methods select inside the learning algorithm, as L1 regularization does. **Wrapper** methods train the actual model on each candidate subset and score it on held-out data; they capture interactions and usually give the best accuracy, at a cost of one model fit per evaluation.

Written as an optimization problem, the decision variable is a binary vector **x** ∈ {0, 1}^D, where x_j = 1 keeps feature j. The search space has 2^D points, and a wrapper objective has no gradient, no convexity and noticeable noise. Exhaustive search is impossible beyond about 30 features, and greedy forward or backward selection suffers from the "nesting effect": a feature added early can never be removed (Xue et al., 2016). Population-based metaheuristics search the space globally and are the standard response.

When accuracy and subset size both matter, the usual single-objective form is a weighted sum,

$$f(\mathbf{x}) = \alpha\,E(\mathbf{x}) + (1-\alpha)\,\frac{|\mathbf{x}|}{D}$$ {#eq:weighted}

where E(**x**) is an error measure and |**x**| the number of selected features. With α close to 1, accuracy dominates and size only breaks ties; lowering α traces an approximation of the Pareto front between the two objectives (Emary, Zawbaa and Hassanien, 2016).

## Genetic Algorithm

A Genetic Algorithm (Holland, 1975; see also Sivanandam and Deepa, 2018, and Luke, 2013) evolves a population of candidate solutions. For feature selection the representation is natural: a chromosome is the bit string **x** (Yang and Honavar, 1998). Each generation, parents are chosen with a preference for low fitness, for example by **tournament selection**, where k individuals are drawn at random and the best one wins. **Crossover** mixes two parents; in uniform crossover each bit of a child comes from either parent with probability one half. **Mutation** flips each bit with a small probability, typically 1/D, which keeps the population from losing bits it can never recover. **Elitism** copies the best individuals unchanged into the next generation, so the best solution found is never lost.

## Particle Swarm Optimization and its binary version

Particle Swarm Optimization (Kennedy and Eberhart, 1995) moves a swarm of particles through a continuous space. Each particle i has a position x_i and a velocity v_i, remembers its own best position p_i and knows the swarm's best position g. The velocity update is

$$v_{i} \leftarrow w\,v_{i} + c_1 r_1 (p_{i} - x_{i}) + c_2 r_2 (g - x_{i})$$ {#eq:pso}

with random numbers r_1, r_2 drawn uniformly from [0, 1] for every dimension, acceleration constants c_1 and c_2, and an inertia weight w (Shi and Eberhart, 1998). In continuous PSO a falling inertia weight, often from 0.9 to 0.4, is the usual way to move from exploration to fine-tuning.

The **binary** version (Kennedy and Eberhart, 1997) keeps the real-valued velocity but reinterprets it: a sigmoid turns each velocity component into the probability that the corresponding bit is 1,

$$x_{ij} = 1 \quad \text{if} \quad r < \frac{1}{1 + e^{-v_{ij}}}, \qquad 0 \text{ otherwise}$$ {#eq:bpso}

so a large positive velocity makes a bit almost surely 1 and a velocity of 0 makes it a fair coin. Velocities are clamped, commonly to ±6, so that no probability becomes exactly 0 or 1 (Mirjalili and Lewis, 2013). @sec:methods shows that this reinterpretation changes the role of the inertia weight.

## Grey Wolf Optimizer and its binary version

The Grey Wolf Optimizer (Mirjalili, Mirjalili and Lewis, 2014) models the social hierarchy and hunting of grey wolves. The three best solutions found so far are called alpha, beta and delta. Every other wolf X moves toward a point determined by all three leaders. For each leader L,

$$A = 2a\,r_1 - a, \qquad C = 2\,r_2, \qquad X_L' = X_L - A\,\left|C\,X_L - X\right|$$ {#eq:gwo}

and the new position is the mean of the three X_L' of @eq:gwo. The coefficient a falls linearly from 2 to 0 over the run. While |A| > 1 the wolves can overshoot the leaders and explore; when |A| < 1 they close in.

Emary et al. (2016) proposed two binary versions for wrapper feature selection. The second, bGWO2, used in this study, squashes the continuous mean position with a steep sigmoid and draws each bit:

$$x_d = 1 \quad \text{if} \quad \frac{1}{1 + e^{-10\,(X_d - 0.5)}} \geq r, \qquad 0 \text{ otherwise}$$ {#eq:bgwo}

## Bayesian optimization of hyperparameters

Hyperparameters such as the number of trees or the learning rate are set before training and strongly affect accuracy. Random search samples configurations independently and is a strong baseline when only a few hyperparameters matter (Bergstra and Bengio, 2012). Bayesian optimization instead builds a model of the objective from the trials so far and proposes the next trial where improvement is most likely. The Tree-structured Parzen Estimator (TPE; Bergstra et al., 2011), the default sampler of the Optuna library (Akiba et al., 2019), models the density of good and bad configurations separately and samples where their ratio is high. It starts with a few random trials, 10 by default, before its model takes over.

## Evaluating rare-event predictions

When only about 3% of players become elite, accuracy is meaningless: predicting "no" for everyone is 97% accurate. The **PR-AUC** (area under the precision-recall curve, computed as average precision) measures how well the rare positives are ranked above everyone else; a random ranking scores about the positive rate (Saito and Rehmsmeier, 2015). ROC-AUC is reported too, but it looks optimistic when negatives vastly outnumber positives. Because a scout works with a shortlist, **precision@50** and **recall@50** describe the 50 highest-ranked players. The **Brier score**, the mean squared error of predicted probabilities, measures calibration (Brier, 1950).

## Comparing stochastic optimizers

Metaheuristics are random, so one run proves nothing. Derrac et al. (2011) recommend running each algorithm many times on the same problem, testing for any difference with the Friedman test, and comparing pairs with the Wilcoxon signed-rank test, which pairs the runs (here by seed) and assumes no normal distribution, followed by a correction for multiple comparisons such as Holm's (Holm, 1979).

## SHAP

SHAP values (Lundberg and Lee, 2017) split a model's prediction for one input into additive contributions from each feature, based on Shapley values from cooperative game theory. Averaging their absolute values over a data set gives a global importance ranking, and for tree ensembles they can be computed exactly and quickly.
