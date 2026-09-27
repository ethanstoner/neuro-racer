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
from src.config import Config
from src.physics import CarState, step
from src.track import min_centerline_radius
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
    return min_centerline_radius(load(name, cfg).centerline)


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


def test_the_track_set_spans_a_range_of_difficulty():
    """Held-out tracks are only a real test if they are not all the same.

    The set must include corners meaningfully tighter than the training tracks,
    or "it generalises" just means "it saw an equally easy track".
    """
    tightest = {n: min_corner_radius(n) for n in BUILDERS}
    assert min(tightest.values()) < 0.5 * max(tightest.values()), (
        f"all tracks are similarly tight: {tightest}")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_no_track_is_secretly_an_oval(name):
    """A track whose curvature never reverses can be solved by a constant
    steering bias, which teaches a memorised trajectory rather than a policy.

    The oval is kept deliberately as the negative control. Everything else must
    change direction somewhere, or it is not adding anything to the test set.
    The first "teardrop" failed this: 0% reverse curvature, an oval in disguise.
    """
    c = load(name, CFG).centerline
    tang = np.diff(np.vstack([c, c[:2]]), axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    cross = tang[:-1, 0] * tang[1:, 1] - tang[:-1, 1] * tang[1:, 0]
    reverse = min((cross < -1e-4).mean(), (cross > 1e-4).mean())

    if name in ("oval", "keyhole"):
        return          # deliberate single-direction controls
    assert reverse > 0.05, (
        f"{name} only turns one way ({reverse:.1%} reverse) -- a constant "
        f"steering bias solves it, so it teaches nothing")


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
