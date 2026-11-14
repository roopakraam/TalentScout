"""Binary metaheuristics for wrapper feature selection.

All four share one signature, ``run(fitness, D, init_pop, budget, rng)``, and
are looked up by name through ``OPTIMIZERS``. ``VARIANTS`` holds ablations
that are reported separately and are not part of the main comparison.
"""
from functools import partial

from . import bgwo, bpso, ga, random_search
from .common import Evaluator, OptResult, diversity, init_population, repair, run_rng, sigmoid

OPTIMIZERS = {
    "ga": ga.run,
    "bpso": bpso.run,
    "bgwo": bgwo.run,
    "random": random_search.run,
}

VARIANTS = {
    # the plan's starting-point inertia schedule for Binary PSO (fails the toy test)
    "bpso_w_dec": partial(bpso.run, w_start=0.9, w_end=0.4),
}

ALL_RUNNERS = {**OPTIMIZERS, **VARIANTS}

__all__ = [
    "OPTIMIZERS", "VARIANTS", "ALL_RUNNERS", "Evaluator", "OptResult", "diversity",
    "init_population", "repair", "run_rng", "sigmoid",
]
