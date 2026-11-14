"""Binary Grey Wolf Optimiser, the bGWO2 variant (CSE578 Unit IV).

Grey Wolf Optimiser (Mirjalili, Mirjalili and Lewis, 2014): the three best
solutions found so far, alpha, beta and delta, lead the pack, and every wolf
moves to a point between them. For each leader L::

    A = 2 * a * r1 - a          C = 2 * r2
    Dist = |C * X_L - X|        X_L' = X_L - A * Dist

and the continuous new position is the mean of the three ``X_L'``. The
coefficient ``a`` falls linearly from 2 to 0, so the wolves roam widely at
first and close in on the leaders at the end.

bGWO2 (Emary, Zawbaa and Hassanien, 2016) turns that position into bits with
a steep sigmoid::

    x_d = 1 if sigmoid(10 * (X_new_d - 0.5)) >= rand else 0

Implementation note: the leaders are the three best *distinct* solutions seen
so far, kept in an archive, so alpha, beta and delta never collapse into the
same vector.
"""
from __future__ import annotations

import math

import numpy as np

from .common import Evaluator, OptResult, diversity, repair, sigmoid

NAME = "bgwo"


def _update_leaders(leaders: list, wolves: np.ndarray, fit: np.ndarray, n_leaders: int = 3) -> list:
    """Three best distinct (fitness, mask) pairs among the old leaders and the current pack."""
    pool = {mask.tobytes(): (value, mask) for value, mask in leaders}
    for mask, value in zip(wolves, fit):
        if np.isfinite(value):
            key = mask.tobytes()
            if key not in pool or value < pool[key][0]:
                pool[key] = (float(value), mask.copy())
    ranked = sorted(pool.values(), key=lambda pair: pair[0])
    return ranked[:n_leaders]


def run(fitness, D: int, init_pop: np.ndarray, budget: int, rng: np.random.Generator,
        a_start: float = 2.0, a_end: float = 0.0, steepness: float = 10.0) -> OptResult:
    ev = Evaluator(fitness, budget)

    wolves = np.array(init_pop, dtype=np.uint8)
    n = len(wolves)
    fit = ev.evaluate_population(wolves)
    diversity_log = [diversity(wolves)]
    leaders = _update_leaders([], wolves, fit)

    n_iter = math.ceil((budget - n) / n)
    for it in range(n_iter):
        if ev.remaining <= 0:
            break
        ev.next_iteration()
        a = a_start - (a_start - a_end) * it / max(n_iter - 1, 1)

        x = wolves.astype(float)
        target = np.zeros((n, D))
        for _, leader in leaders:
            r1, r2 = rng.random((n, D)), rng.random((n, D))
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            dist = np.abs(C * leader - x)
            target += leader - A * dist
        target /= len(leaders)

        wolves = (sigmoid(steepness * (target - 0.5)) >= rng.random((n, D))).astype(np.uint8)
        for row in wolves:
            repair(row, rng)

        fit = ev.evaluate_population(wolves)
        leaders = _update_leaders(leaders, wolves, fit)
        diversity_log.append(diversity(wolves))

    return ev.result(diversity_log)
