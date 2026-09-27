import numpy as np
import pytest
from src.config import Config
from src.fitness import wrapped_delta, FitnessTracker

CFG = Config()


def test_wrapped_delta_handles_the_seam():
    """Crossing the finish line is forward progress, not a huge negative."""
    assert wrapped_delta(np.float32([0.01]), np.float32([0.99]))[0] == pytest.approx(0.02)
    assert wrapped_delta(np.float32([0.99]), np.float32([0.01]))[0] == pytest.approx(-0.02)
    assert wrapped_delta(np.float32([0.60]), np.float32([0.50]))[0] == pytest.approx(0.10)


def test_a_parked_car_scores_below_a_car_that_progressed_then_crashed():
    """The single most important property in the project. If this inverts,
    generation 1 evolves into parked cars and nothing ever learns to drive."""
    t = FitnessTracker(2, CFG)
    t.update(progress=np.float32([0.0, 0.0]), speed=np.float32([0.0, 300.0]),
             alive=np.array([True, True]), dt=CFG.dt)
    t.update(progress=np.float32([0.0, 0.20]), speed=np.float32([0.0, 300.0]),
             alive=np.array([True, True]), dt=CFG.dt)
    t.update(progress=np.float32([0.0, 0.40]), speed=np.float32([0.0, 300.0]),
             alive=np.array([True, False]), dt=CFG.dt)     # car 1 crashes
    for _ in range(100):
        t.update(progress=np.float32([0.0, 0.40]), speed=np.float32([0.0, 0.0]),
                 alive=np.array([True, False]), dt=CFG.dt)
    s = t.scores()
    assert s[1] > s[0]


def test_crashing_keeps_the_score_already_earned():
    t = FitnessTracker(1, CFG)
    t.update(np.float32([0.0]), np.float32([200.0]), np.array([True]), CFG.dt)
    t.update(np.float32([0.30]), np.float32([200.0]), np.array([True]), CFG.dt)
    earned = t.scores()[0]
    t.update(np.float32([0.30]), np.float32([0.0]), np.array([False]), CFG.dt)
    assert t.scores()[0] == pytest.approx(earned)


def test_idle_car_is_killed():
    t = FitnessTracker(1, CFG)
    alive = np.array([True])
    ticks = int(CFG.idle_seconds / CFG.dt) + 5
    for _ in range(ticks):
        alive = t.update(np.float32([0.10]), np.float32([0.0]), alive, CFG.dt)
    assert not alive[0]


def test_moving_car_is_not_killed():
    t = FitnessTracker(1, CFG)
    alive = np.array([True])
    p = 0.0
    for _ in range(int(CFG.idle_seconds / CFG.dt) + 5):
        p += 0.001
        alive = t.update(np.float32([p % 1.0]), np.float32([200.0]), alive, CFG.dt)
    assert alive[0]


def test_car_going_in_circles_is_killed():
    """Spinning on the spot makes no forward progress, so the idle cull must
    still fire even though the car is technically moving fast."""
    t = FitnessTracker(1, CFG)
    alive = np.array([True])
    # Comfortably past the idle limit: the tracker needs a few ticks to
    # establish its progress peak before the idle counter starts climbing.
    for i in range(int(CFG.idle_seconds / CFG.dt) + 60):
        wobble = 0.10 + 0.002 * np.sin(i * 0.3)     # oscillates, never advances
        alive = t.update(np.float32([wobble]), np.float32([300.0]), alive, CFG.dt)
    assert not alive[0]


def test_completing_a_lap_switches_to_time_optimisation():
    """Two cars both finish a lap; the faster one must score higher."""
    def run(ticks_per_lap):
        t = FitnessTracker(1, CFG)
        alive = np.array([True])
        # Start on the start line, like a real car does. The first update only
        # establishes a baseline -- there is no previous position on tick zero.
        alive = t.update(np.float32([0.0]), np.float32([300.0]), alive, CFG.dt)
        for i in range(ticks_per_lap):
            p = ((i + 1) / ticks_per_lap) % 1.0
            alive = t.update(np.float32([p]), np.float32([300.0]), alive, CFG.dt)
        return t.scores()[0], t.laps[0]

    fast, fast_laps = run(600)
    slow, slow_laps = run(1200)
    assert fast_laps == slow_laps == 1
    assert fast > slow


def test_a_lap_beats_any_amount_of_partial_progress():
    t = FitnessTracker(1, CFG)
    alive = np.array([True])
    alive = t.update(np.float32([0.0]), np.float32([300.0]), alive, CFG.dt)
    for i in range(600):
        alive = t.update(np.float32([((i + 1) / 600) % 1.0]), np.float32([300.0]), alive, CFG.dt)
    lapped = t.scores()[0]

    t2 = FitnessTracker(1, CFG)
    a2 = np.array([True])
    for i in range(590):
        a2 = t2.update(np.float32([(i + 1) / 600]), np.float32([300.0]), a2, CFG.dt)
    assert lapped > t2.scores()[0]


def test_reversing_cannot_erase_earned_progress():
    """Scores use the best progress ever reached, so a car cannot lose points
    by backing up -- and equally cannot farm points by oscillating."""
    t = FitnessTracker(1, CFG)
    alive = np.array([True])
    for p in (0.0, 0.1, 0.2, 0.3, 0.4):
        alive = t.update(np.float32([p]), np.float32([200.0]), alive, CFG.dt)
    peak = t.scores()[0]
    for p in (0.3, 0.2, 0.1):
        alive = t.update(np.float32([p]), np.float32([200.0]), alive, CFG.dt)
    assert t.scores()[0] >= peak
