"""Random search: the control every metaheuristic has to beat (H2b).

Evaluates the shared starting population, then fresh random 0/1 vectors
(each bit on with probability 0.5, the same distribution as the starting
population) until the budget is spent, and keeps the best. There is no
population, so no diversity is recorded.
"""
from __future__ import annotations

import numpy as np

from .common import Evaluator, OptResult, repair

NAME = "random"


def run(fitness, D: int, init_pop: np.ndarray, budget: int, rng: np.random.Generator) -> OptResult:
    ev = Evaluator(fitness, budget)
    batch = len(init_pop)
    ev.evaluate_population(np.array(init_pop, dtype=np.uint8))

    while ev.remaining > 0:
        ev.next_iteration()
        candidates = rng.integers(0, 2, size=(batch, D), dtype=np.uint8)
        for row in candidates:
            repair(row, rng)
        ev.evaluate_population(candidates)

    return ev.result([])
