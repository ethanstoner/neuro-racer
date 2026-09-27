"""The generation pump must overlap simulation with playback without changing
a single number. If it changed results, the visualisation would stop being a
faithful view of the run.
"""
import time
import numpy as np
import pytest
from src.config import Config
from src.tracks import load
from src.net import random_population
from src.simulation import run_generation
from src.evolve import next_generation, mutation_sigma
from src.pump import GenerationPump

CFG = Config(population=60, max_episode_seconds=20.0)


@pytest.fixture(scope="module")
def track():
    return load("snake", CFG)


def test_pumped_run_is_identical_to_synchronous(track):
    """Same seed, same everything -- threading must not perturb the run."""
    def sync(n):
        rng = np.random.default_rng(7)
        pop = random_population(CFG.population, CFG, rng)
        out = []
        for gen in range(n):
            r = run_generation(pop, track, CFG, record=True)
            out.append(r.scores.copy())
            pop = next_generation(pop, r.scores, CFG, rng, mutation_sigma(gen, CFG))
        return out

    def pumped(n):
        rng = np.random.default_rng(7)
        pop = random_population(CFG.population, CFG, rng)
        pump = GenerationPump(track, CFG)
        pump.submit(pop)
        out = []
        try:
            for gen in range(n):
                r = pump.wait()
                out.append(r.scores.copy())
                pop = next_generation(pop, r.scores, CFG, rng, mutation_sigma(gen, CFG))
                pump.submit(pop)
        finally:
            pump.close()
        return out

    for a, b in zip(sync(4), pumped(4)):
        np.testing.assert_array_equal(a, b)


def test_submit_returns_immediately(track):
    """The whole point: submitting must not block the caller."""
    pop = random_population(CFG.population, CFG, np.random.default_rng(1))
    pump = GenerationPump(track, CFG)
    try:
        t0 = time.perf_counter()
        pump.submit(pop)
        submit_cost = time.perf_counter() - t0

        t0 = time.perf_counter()
        pump.wait()
        sim_cost = time.perf_counter() - t0

        assert submit_cost < 0.1, f"submit blocked for {submit_cost:.3f}s"
        assert sim_cost > submit_cost * 2, "simulation was suspiciously cheap"
    finally:
        pump.close()


def test_wait_calls_the_idle_callback(track):
    """scripts/app.py pumps pygame events through this callback -- without it being
    called, the window still stops responding and nothing is fixed."""
    pop = random_population(CFG.population, CFG, np.random.default_rng(2))
    pump = GenerationPump(track, CFG)
    calls = []
    try:
        pump.submit(pop)
        pump.wait(on_wait=lambda: calls.append(time.perf_counter()))
    finally:
        pump.close()

    assert len(calls) > 5, f"idle callback ran only {len(calls)} times"
    gaps = np.diff(calls)
    assert gaps.max() < 0.2, f"idle callback stalled for {gaps.max():.3f}s"


def test_ready_reports_completion(track):
    pop = random_population(CFG.population, CFG, np.random.default_rng(3))
    pump = GenerationPump(track, CFG)
    try:
        pump.submit(pop)
        assert not pump.ready()
        pump.wait()
        assert not pump.ready(), "wait should clear the pending future"
    finally:
        pump.close()


def test_caller_may_mutate_its_population_after_submitting(track):
    """The worker holds its own copy, so the main thread is free to evolve the
    population immediately without corrupting the in-flight generation."""
    pop = random_population(CFG.population, CFG, np.random.default_rng(4))
    pump = GenerationPump(track, CFG)
    try:
        pump.submit(pop)
        pop[:] = 0.0                      # hostile: scribble all over it
        r = pump.wait()
        assert r.scores.max() > 0, "worker saw the mutated population"
    finally:
        pump.close()


def test_close_is_safe_with_work_in_flight(track):
    pop = random_population(CFG.population, CFG, np.random.default_rng(5))
    pump = GenerationPump(track, CFG)
    pump.submit(pop)
    pump.close()
