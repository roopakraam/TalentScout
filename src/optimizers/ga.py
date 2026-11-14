"""Genetic Algorithm for feature selection (CSE578 Unit II).

Binary encoding: gene j = 1 keeps feature j.

Each generation
    1. the ``n_elite`` best solutions are copied unchanged (elitism);
    2. parents are chosen by tournament: draw ``tournament_k`` at random, the best wins;
    3. uniform crossover with probability ``crossover_rate``: each bit comes from
       either parent with equal chance, giving two complementary children;
    4. bit-flip mutation: every bit flips with probability ``mutation_rate`` (default 1/D,
       about one bit per child);
    5. all-zero children are repaired, the children are evaluated.

Elites keep their known fitness and are not evaluated again, so a generation
costs ``pop_size - n_elite`` calls. The run stops when the budget is spent,
which can cut the last generation short.
"""
from __future__ import annotations

import numpy as np

from .common import Evaluator, OptResult, diversity, repair

NAME = "ga"


def _tournament(fit: np.ndarray, k: int, rng: np.random.Generator) -> int:
    contenders = rng.choice(len(fit), size=k, replace=False)
    return int(contenders[np.argmin(fit[contenders])])


def run(fitness, D: int, init_pop: np.ndarray, budget: int, rng: np.random.Generator,
        crossover_rate: float = 0.9, mutation_rate: float | None = None,
        tournament_k: int = 3, n_elite: int = 2) -> OptResult:
    mutation_rate = 1.0 / D if mutation_rate is None else mutation_rate
    ev = Evaluator(fitness, budget)

    pop = np.array(init_pop, dtype=np.uint8)
    pop_size = len(pop)
    fit = ev.evaluate_population(pop)
    diversity_log = [diversity(pop)]

    while ev.remaining > 0:
        ev.next_iteration()
        order = np.argsort(fit, kind="stable")
        elites, elite_fit = pop[order[:n_elite]].copy(), fit[order[:n_elite]].copy()

        children = []
        while len(children) < pop_size - n_elite:
            mother = pop[_tournament(fit, tournament_k, rng)]
            father = pop[_tournament(fit, tournament_k, rng)]
            if rng.random() < crossover_rate:
                take = rng.random(D) < 0.5
                pair = [np.where(take, mother, father), np.where(take, father, mother)]
            else:
                pair = [mother.copy(), father.copy()]
            for child in pair:
                flip = rng.random(D) < mutation_rate
                child = np.where(flip, 1 - child, child).astype(np.uint8)
                children.append(repair(child, rng))
        children = np.array(children[:pop_size - n_elite], dtype=np.uint8)

        child_fit = ev.evaluate_population(children)
        pop = np.vstack([elites, children])
        fit = np.concatenate([elite_fit, child_fit])
        diversity_log.append(diversity(pop))

    return ev.result(diversity_log)
