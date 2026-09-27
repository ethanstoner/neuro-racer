import numpy as np
import pytest
from src.config import Config
from src.net import random_population
from src.evolve import next_generation, tournament_select, mutation_sigma

CFG = Config(population=40, elites=4)


def test_elites_are_carried_over_bit_identical():
    rng = np.random.default_rng(0)
    pop = random_population(CFG.population, CFG, rng)
    scores = rng.uniform(0, 100, CFG.population).astype(np.float32)
    order = np.argsort(-scores)
    nxt = next_generation(pop, scores, CFG, rng, sigma=0.3)
    for i in range(CFG.elites):
        np.testing.assert_array_equal(nxt[i], pop[order[i]])


def test_champion_is_always_index_zero():
    """The renderer highlights index 0 as the reigning champion."""
    rng = np.random.default_rng(10)
    pop = random_population(CFG.population, CFG, rng)
    scores = rng.uniform(0, 100, CFG.population).astype(np.float32)
    nxt = next_generation(pop, scores, CFG, rng, 0.3)
    np.testing.assert_array_equal(nxt[0], pop[np.argmax(scores)])


def test_population_size_is_preserved():
    rng = np.random.default_rng(1)
    pop = random_population(CFG.population, CFG, rng)
    scores = rng.uniform(0, 100, CFG.population).astype(np.float32)
    assert next_generation(pop, scores, CFG, rng, 0.3).shape == pop.shape


def test_tournament_favours_higher_scores():
    rng = np.random.default_rng(2)
    scores = np.arange(50, dtype=np.float32)
    picks = tournament_select(scores, 5000, k=3, rng=rng)
    assert picks.mean() > 35, "tournament selection is not applying pressure"


def test_tournament_k_one_is_uniform():
    """Sanity check on the selection pressure knob itself."""
    rng = np.random.default_rng(12)
    scores = np.arange(50, dtype=np.float32)
    picks = tournament_select(scores, 20000, k=1, rng=rng)
    assert 22 < picks.mean() < 27


def test_sigma_decays_but_never_below_the_floor():
    assert mutation_sigma(0, CFG) == pytest.approx(CFG.mutation_sigma_start)
    assert mutation_sigma(500, CFG) == pytest.approx(CFG.mutation_sigma_min)
    assert mutation_sigma(5, CFG) < mutation_sigma(4, CFG)


def test_children_differ_from_parents():
    """Zero effective mutation would make the GA a no-op after the first
    generation and it would look like 'converged' rather than 'broken'."""
    rng = np.random.default_rng(13)
    pop = random_population(CFG.population, CFG, rng)
    scores = rng.uniform(0, 100, CFG.population).astype(np.float32)
    nxt = next_generation(pop, scores, CFG, rng, 0.3)
    children = nxt[CFG.elites:]
    assert not any(np.array_equal(c, p) for c in children for p in pop[:3])


def test_ga_improves_on_a_toy_objective():
    """Maximise the sum of the genome. If this does not climb steadily the GA
    is broken, and no amount of physics tuning will save the real run."""
    cfg = Config(population=60, elites=5)
    rng = np.random.default_rng(7)
    pop = random_population(cfg.population, cfg, rng)
    first = pop.sum(axis=1).max()
    for gen in range(30):
        pop = next_generation(pop, pop.sum(axis=1), cfg, rng, mutation_sigma(gen, cfg))
    assert pop.sum(axis=1).max() > first * 3


def test_best_score_never_decreases_with_elitism():
    """Elitism guarantees monotonic best-of-generation on a static objective."""
    cfg = Config(population=40, elites=4)
    rng = np.random.default_rng(14)
    pop = random_population(cfg.population, cfg, rng)
    best = -np.inf
    for gen in range(20):
        scores = pop.sum(axis=1)
        assert scores.max() >= best - 1e-5
        best = max(best, scores.max())
        pop = next_generation(pop, scores, cfg, rng, mutation_sigma(gen, cfg))


def test_output_is_finite():
    rng = np.random.default_rng(3)
    pop = random_population(CFG.population, CFG, rng)
    scores = np.zeros(CFG.population, dtype=np.float32)
    scores[0] = 1.0
    assert np.isfinite(next_generation(pop, scores, CFG, rng, 0.3)).all()
