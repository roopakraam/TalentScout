"""Binary Particle Swarm Optimisation (CSE578 Unit III).

The discrete version of Kennedy and Eberhart (1997). Each particle keeps a
real-valued velocity per bit; a sigmoid turns the velocity into the
probability that the bit is 1::

    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    v = clip(v, -v_max, v_max)
    x = 1 if rand < sigmoid(v) else 0

* ``pbest`` is the particle's own best position, ``gbest`` the swarm's best;
* velocities are clamped to +-6 so that a probability never sticks at exactly 0 or 1.

Inertia weight
--------------
The default is ``w = 1`` (no damping), which is the original 1997 algorithm.
The plan's starting-point setting was a weight falling from 0.9 to 0.4, the
usual choice for *continuous* PSO. That schedule fails the toy test: in the
binary version a bit agrees with ``pbest`` and ``gbest`` exactly when the
attraction terms are zero, so ``w < 1`` shrinks its velocity towards 0, and a
velocity of 0 means the bit is 1 with probability 0.5. A falling weight
therefore makes the swarm *more* random over time instead of less. The
schedule is kept as the ablation ``bpso_w_dec`` (see ``VARIANTS`` in
``src/optimizers/__init__.py``); set ``w_start`` and ``w_end`` to use any
other linear schedule.
"""
from __future__ import annotations

import math

import numpy as np

from .common import Evaluator, OptResult, diversity, repair, sigmoid

NAME = "bpso"


def run(fitness, D: int, init_pop: np.ndarray, budget: int, rng: np.random.Generator,
        w_start: float = 1.0, w_end: float = 1.0, c1: float = 2.0, c2: float = 2.0,
        v_max: float = 6.0) -> OptResult:
    ev = Evaluator(fitness, budget)

    x = np.array(init_pop, dtype=np.uint8)
    n = len(x)
    fit = ev.evaluate_population(x)
    diversity_log = [diversity(x)]

    v = np.zeros((n, D))
    pbest, pbest_fit = x.copy(), fit.copy()
    g = int(np.argmin(pbest_fit))
    gbest = pbest[g].copy()

    n_iter = math.ceil((budget - n) / n)
    for it in range(n_iter):
        if ev.remaining <= 0:
            break
        ev.next_iteration()
        w = w_start - (w_start - w_end) * it / max(n_iter - 1, 1)

        r1, r2 = rng.random((n, D)), rng.random((n, D))
        xf = x.astype(float)
        v = w * v + c1 * r1 * (pbest - xf) + c2 * r2 * (gbest - xf)
        v = np.clip(v, -v_max, v_max)
        x = (rng.random((n, D)) < sigmoid(v)).astype(np.uint8)
        for row in x:
            repair(row, rng)

        fit = ev.evaluate_population(x)
        better = fit < pbest_fit
        pbest[better], pbest_fit[better] = x[better], fit[better]
        g = int(np.argmin(pbest_fit))
        gbest = pbest[g].copy()
        diversity_log.append(diversity(x))

    return ev.result(diversity_log)
