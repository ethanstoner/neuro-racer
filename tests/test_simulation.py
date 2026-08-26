import subprocess
import sys
import numpy as np
from config import Config
from src.tracks import load
from src.net import random_population
from src.simulation import run_generation

CFG = Config(population=20, max_episode_seconds=6.0)


def test_simulation_never_imports_pygame():
    """Training must stay headless. If pygame creeps into the simulation path
    the whole fast-forward design is compromised, so this is checked in a fresh
    interpreter rather than relying on import order in this one."""
    code = ("import sys; import src.simulation; "
            "sys.exit(1 if 'pygame' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_run_generation_returns_one_score_per_genome():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(0))
    r = run_generation(g, track, CFG)
    assert r.scores.shape == (CFG.population,)
    assert np.isfinite(r.scores).all()


def test_random_genomes_mostly_crash():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(1))
    r = run_generation(g, track, CFG)
    assert r.alive.sum() < CFG.population, "random genomes should crash"


def test_deterministic_for_the_same_genomes():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(2))
    a = run_generation(g, track, CFG)
    b = run_generation(g, track, CFG)
    np.testing.assert_array_equal(a.scores, b.scores)
    assert a.ticks_run == b.ticks_run


def test_frame_log_records_every_tick_when_requested():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(3))
    r = run_generation(g, track, CFG, record=True)
    assert r.frames is not None
    assert r.frames.pos.shape == (r.ticks_run, CFG.population, 2)
    assert r.frames.activations.shape == (r.ticks_run, CFG.population, CFG.n_hidden)
    assert np.isfinite(r.frames.pos).all()


def test_no_frame_log_by_default():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(4))
    assert run_generation(g, track, CFG).frames is None


def test_recording_does_not_change_the_outcome():
    """The renderer must be a passive observer. If recording perturbs the
    simulation, what you watch is not what the trainer scored."""
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(5))
    plain = run_generation(g, track, CFG)
    recorded = run_generation(g, track, CFG, record=True)
    np.testing.assert_array_equal(plain.scores, recorded.scores)


def test_dead_cars_stop_moving():
    """A dead car must freeze where it died, not coast off across the map."""
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(6))
    r = run_generation(g, track, CFG, record=True)
    f = r.frames
    for t in range(1, r.ticks_run):
        just_dead = ~f.alive[t] & ~f.alive[t - 1]
        if just_dead.any():
            np.testing.assert_array_equal(f.pos[t][just_dead], f.pos[t - 1][just_dead])


def test_all_cars_start_on_the_track():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(7))
    r = run_generation(g, track, CFG, record=True)
    assert r.frames.alive[0].all(), "someone spawned off-track"


def test_every_car_dies_or_survives_to_the_cap():
    track = load("oval", CFG)
    g = random_population(CFG.population, CFG, np.random.default_rng(8))
    r = run_generation(g, track, CFG)
    assert r.ticks_run <= CFG.max_ticks
    if r.ticks_run < CFG.max_ticks:
        assert not r.alive.any(), "stopped early with cars still alive"
