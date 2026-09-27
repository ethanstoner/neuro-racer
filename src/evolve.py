"""Genetic algorithm: elitism + tournament selection + uniform crossover +
gaussian mutation. Deliberately plain -- the interesting behaviour should come
from the fitness function and the environment, not from GA cleverness.
"""
import numpy as np
from src.config import Config


def mutation_sigma(generation: int, cfg: Config) -> float:
    """Anneal mutation: explore hard early, refine the racing line later."""
    s = cfg.mutation_sigma_start * (cfg.mutation_decay ** generation)
    return float(max(s, cfg.mutation_sigma_min))


def tournament_select(scores: np.ndarray, n: int, k: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Pick n parents. Each is the best of k randomly drawn contestants."""
    contestants = rng.integers(0, len(scores), size=(n, k))
    best = np.argmax(scores[contestants], axis=1)
    return contestants[np.arange(n), best]


def next_generation(pop: np.ndarray, scores: np.ndarray, cfg: Config,
                    rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Elites are copied unmutated and placed first, so index 0 of every
    generation is the reigning champion. The renderer relies on that."""
    order = np.argsort(-scores)
    elites = pop[order[:cfg.elites]].copy()

    n_children = len(pop) - cfg.elites
    a = pop[tournament_select(scores, n_children, cfg.tournament_k, rng)]
    b = pop[tournament_select(scores, n_children, cfg.tournament_k, rng)]

    take_a = rng.random(a.shape) < 0.5
    children = np.where(take_a, a, b)

    touched = rng.random(children.shape) < cfg.mutation_rate
    children = children + touched * rng.normal(0.0, sigma, children.shape)

    return np.vstack([elites, children]).astype(np.float32)
