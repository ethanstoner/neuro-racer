"""Tests of the design premise itself, rather than of any one function.

The project promises cars that learn to "go the correct speed". That is only a
learnable thing if the track actually contains corners the car cannot take at
full throttle. If someone later widens the track, raises the steering rate or
lowers the top speed, the optimal policy silently collapses to "floor it" and
the most interesting part of the project quietly stops existing.

These tests fail loudly when that happens.
"""
import numpy as np
import pytest
from config import Config
from src.physics import CarState, step
from src.tracks import BUILDERS, load

CFG = Config()


def measured_turn_radius(speed: float, cfg: Config = CFG) -> float:
    """Drive at `speed` with full steering lock and measure the circle radius.

    Measured from the simulation rather than derived from the formula, so it
    stays honest if the physics changes.
    """
    car = CarState.spawn(1, 600.0, 400.0, 0.0)
    car.vel[:] = [speed, 0.0]
    lock = np.float32([[0.0, 0.0, 1.0]])

    # Hold speed with light throttle so drag does not bleed it off mid-measure.
    for _ in range(5):
        car = step(car, lock, cfg)
    start_angle = float(car.angle[0])
    turned = 0.0
    ticks = 0
    while turned < np.pi / 2 and ticks < 2000:
        v = float(np.linalg.norm(car.vel))
        throttle = 1.0 if v < speed else 0.0
        car = step(car, np.float32([[throttle, 0.0, 1.0]]), cfg)
        turned = abs(float(car.angle[0]) - start_angle)
        ticks += 1
    arc_time = ticks * cfg.dt
    omega = turned / max(arc_time, 1e-9)
    return speed / max(omega, 1e-9)


def min_corner_radius(name: str, cfg: Config = CFG) -> float:
    """Tightest radius of curvature anywhere on a track's centerline."""
    c = load(name, cfg).centerline
    d1 = np.gradient(c, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    curvature = num / np.maximum(den, 1e-12)
    return float(1.0 / np.maximum(curvature.max(), 1e-12))


def test_turn_radius_grows_with_speed():
    """The premise underneath everything: going faster costs you cornering."""
    assert measured_turn_radius(120.0) < measured_turn_radius(260.0) < measured_turn_radius(420.0)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_track_has_corners_that_require_braking(name):
    """At top speed the car must be unable to hold the tightest corner."""
    tightest = min_corner_radius(name)
    flat_out = measured_turn_radius(CFG.max_speed)
    assert flat_out > tightest, (
        f"{name}: at max speed the car turns in {flat_out:.0f}px but the "
        f"tightest corner is {tightest:.0f}px -- flooring it works everywhere, "
        f"so there is no correct speed to learn")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_corner_is_takeable_at_some_speed(name):
    """And it must be possible at all -- an impossible corner means the cars
    can never complete a lap no matter how well they learn."""
    tightest = min_corner_radius(name)
    slow = measured_turn_radius(90.0)
    assert slow < tightest, (
        f"{name}: even at 90px/s the car needs {slow:.0f}px to turn but the "
        f"tightest corner is {tightest:.0f}px -- the track is undriveable")


def test_the_speed_window_is_wide_enough_to_matter():
    """There must be a meaningful spread between 'can corner' and 'top speed',
    or the optimal policy is a single constant throttle and nothing is learned."""
    tightest = min(min_corner_radius(n) for n in BUILDERS)
    # Highest speed that still fits the tightest corner on any track.
    speeds = np.arange(60.0, CFG.max_speed + 1, 10.0)
    ok = [s for s in speeds if measured_turn_radius(float(s)) < tightest]
    assert ok, "no speed can take the tightest corner"
    corner_speed = max(ok)
    assert corner_speed < CFG.max_speed * 0.85, (
        f"corner speed {corner_speed:.0f} is too close to top speed "
        f"{CFG.max_speed} -- braking barely matters")
