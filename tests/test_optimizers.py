"""Toy-problem tests for the optimisers and their shared helpers.

Run with ``python -m pytest tests`` (or ``python tests/test_optimizers.py``).
The toy fitness functions cost nothing, so a bug shows up in seconds instead
of after a night of football runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizers import OPTIMIZERS, diversity, init_population, repair, run_rng  # noqa: E402

D, POP, BUDGET = 45, 20, 600
METAHEURISTICS = ["ga", "bpso", "bgwo"]


def count_ones(mask):
    """Toy fitness 1: minimise the number of selected features. Optimum = a single 1."""
    return float(np.sum(mask))


TARGET = np.random.default_rng(123).integers(0, 2, D)


def distance_to_target(mask):
    """Toy fitness 2: Hamming distance to a hidden target vector. Optimum = 0."""
    return float(np.sum(np.asarray(mask) != TARGET))


def run(name, fitness, seed=0, budget=BUDGET):
    return OPTIMIZERS[name](fitness, D, init_population(seed, POP, D), budget, run_rng(seed))


# --------------------------------------------------------------------------- helpers
def test_init_population_is_shared_and_valid():
    a, b = init_population(3, POP, D), init_population(3, POP, D)
    assert a.shape == (POP, D) and np.array_equal(a, b)
    assert not np.array_equal(a, init_population(4, POP, D))
    assert set(np.unique(a)) <= {0, 1} and a.sum(axis=1).min() >= 1


def test_repair_switches_on_exactly_one_bit():
    rng = np.random.default_rng(0)
    assert repair(np.zeros(D, dtype=np.uint8), rng).sum() == 1
    mask = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert np.array_equal(repair(mask.copy(), rng), mask)


def test_diversity_matches_brute_force():
    pop = np.random.default_rng(1).integers(0, 2, (7, 12))
    brute = np.mean([np.sum(pop[i] != pop[j]) for i in range(7) for j in range(i + 1, 7)])
    assert np.isclose(diversity(pop), brute)
    assert diversity(np.ones((5, 9), dtype=int)) == 0.0


# --------------------------------------------------------------------------- every optimiser
def test_budget_is_respected_exactly():
    for name in OPTIMIZERS:
        calls = []

        def fitness(mask):
            calls.append(1)
            return count_ones(mask)

        result = run(name, fitness)
        assert len(calls) == BUDGET, name
        assert result.calls == BUDGET and len(result.history) == BUDGET, name


def test_history_is_monotone_and_consistent():
    for name in OPTIMIZERS:
        result = run(name, distance_to_target)
        assert np.all(np.diff(result.history) <= 0), name
        assert result.history[-1] == result.best_fitness == distance_to_target(result.best_mask), name


def test_same_seed_same_result():
    for name in OPTIMIZERS:
        a, b = run(name, distance_to_target, seed=5), run(name, distance_to_target, seed=5)
        assert np.array_equal(a.history, b.history) and np.array_equal(a.best_mask, b.best_mask), name


def test_never_evaluates_an_empty_subset():
    for name in OPTIMIZERS:
        def fitness(mask):
            assert np.sum(mask) >= 1
            return count_ones(mask)
        run(name, fitness)


def test_all_start_from_the_same_population():
    for seed in (0, 1):
        firsts = []
        for name in OPTIMIZERS:
            seen = []

            def fitness(mask):
                seen.append(mask.copy())
                return count_ones(mask)

            run(name, fitness, seed=seed)
            firsts.append(np.array(seen[:POP]))
        for other in firsts[1:]:
            assert np.array_equal(firsts[0], other)


# --------------------------------------------------------------------------- the toy tests of the tracker
def test_metaheuristics_reach_a_single_feature():
    """T3.6 / T4.1 / T4.2: minimising the number of 1s must end with exactly one feature."""
    for name in METAHEURISTICS:
        for seed in range(3):
            assert run(name, count_ones, seed).best_fitness == 1.0, (name, seed)


def test_metaheuristics_beat_random_search():
    for name in METAHEURISTICS:
        for fitness in (count_ones, distance_to_target):
            for seed in range(3):
                assert run(name, fitness, seed).best_fitness < run("random", fitness, seed).best_fitness, (name, seed)


def test_decreasing_inertia_keeps_binary_pso_from_converging():
    """The plan's schedule (w from 0.9 to 0.4) is why the default is w = 1; see src/optimizers/bpso.py."""
    from src.optimizers import VARIANTS

    for seed in range(3):
        pop = init_population(seed, POP, D)
        damped = VARIANTS["bpso_w_dec"](count_ones, D, pop, BUDGET, run_rng(seed))
        undamped = run("bpso", count_ones, seed)
        assert undamped.best_fitness == 1.0 < damped.best_fitness
        assert undamped.diversity[-1] < damped.diversity[-1]      # the damped swarm stays scattered


def test_diversity_log_has_one_value_per_iteration():
    assert len(run("bpso", count_ones).diversity) == 30          # start + 29 iterations
    assert len(run("bgwo", count_ones).diversity) == 30
    assert len(run("ga", count_ones).diversity) == 34            # start + 32 full generations + 1 partial
    assert run("random", count_ones).diversity == []


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} tests passed")
