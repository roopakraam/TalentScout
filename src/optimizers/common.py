"""Shared pieces for all optimisers, so the comparison is fair by construction.

Every optimiser has the same signature::

    run(fitness, D, init_pop, budget, rng, **settings) -> OptResult

* ``fitness`` is any callable that maps a 0/1 vector of length D to a number to minimise;
* ``init_pop`` is the starting population, identical for every algorithm with the same seed;
* ``budget`` is the number of fitness calls allowed. Every call counts, including
  calls on a solution that was seen before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def init_population(seed: int, pop_size: int = 20, D: int = 45) -> np.ndarray:
    """Random 0/1 matrix (each bit on with probability 0.5), the same for a given seed."""
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(pop_size, D), dtype=np.uint8)
    for row in pop:
        repair(row, rng)
    return pop


def run_rng(seed: int) -> np.random.Generator:
    """Random stream for the search itself, separate from the one that built the population."""
    return np.random.default_rng(10_000 + seed)


def repair(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Enforce "at least one feature": switch on one random bit of an all-zero vector (in place)."""
    if not mask.any():
        mask[rng.integers(mask.size)] = 1
    return mask


def diversity(pop: np.ndarray) -> float:
    """Mean pairwise Hamming distance: how many bits differ, on average, between two candidates."""
    pop = np.asarray(pop, dtype=np.int16)
    n = len(pop)
    if n < 2:
        return 0.0
    ones = pop.sum(axis=0)                      # per bit: number of candidates with a 1
    differing_pairs = (ones * (n - ones)).sum()
    return float(differing_pairs / (n * (n - 1) / 2))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OptResult:
    best_mask: np.ndarray
    best_fitness: float
    history: np.ndarray                 # best fitness so far after every call, length = budget
    diversity: list[float] = field(default_factory=list)     # one value per iteration
    iteration_of_call: np.ndarray | None = None
    calls: int = 0

    @property
    def n_selected(self) -> int:
        return int(self.best_mask.sum())


class Evaluator:
    """Counts fitness calls, enforces the budget and records the best-so-far curve."""

    def __init__(self, fitness, budget: int):
        self.fitness = fitness
        self.budget = int(budget)
        self.calls = 0
        self.iteration = 0
        self.best_fitness = np.inf
        self.best_mask: np.ndarray | None = None
        self.history: list[float] = []
        self.iteration_of_call: list[int] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def evaluate(self, mask: np.ndarray) -> float:
        if self.remaining <= 0:
            raise RuntimeError("fitness budget exhausted")
        mask = np.asarray(mask, dtype=np.uint8)
        value = float(self.fitness(mask))
        self.calls += 1
        if value < self.best_fitness:
            self.best_fitness, self.best_mask = value, mask.copy()
        self.history.append(self.best_fitness)
        self.iteration_of_call.append(self.iteration)
        return value

    def evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        """Evaluate the rows in order. Rows left over when the budget runs out get +inf."""
        values = np.full(len(pop), np.inf)
        for i, mask in enumerate(pop):
            if self.remaining <= 0:
                break
            values[i] = self.evaluate(mask)
        return values

    def next_iteration(self) -> None:
        self.iteration += 1

    def result(self, diversity_log: list[float]) -> OptResult:
        return OptResult(
            best_mask=self.best_mask.copy(), best_fitness=self.best_fitness,
            history=np.array(self.history), diversity=list(diversity_log),
            iteration_of_call=np.array(self.iteration_of_call), calls=self.calls)
