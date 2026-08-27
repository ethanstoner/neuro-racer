import numpy as np
import pytest
from config import Config
from src.track import Track, circle_centerline, self_approach_distance
from src.tracks import BUILDERS, load


@pytest.fixture(scope="module")
def circle_track():
    cfg = Config()
    return Track.from_centerline(circle_centerline(cx=600, cy=400, r=250, n=400), cfg)


def test_drivable_area_matches_annulus(circle_track):
    """A ring of radius 250 and width 90 has a known area."""
    expected = np.pi * (295.0 ** 2 - 205.0 ** 2)
    actual = circle_track.drivable.sum()
    assert abs(actual - expected) / expected < 0.02


def test_border_is_not_drivable(circle_track):
    d = circle_track.drivable
    assert not d[0, :].any() and not d[-1, :].any()
    assert not d[:, 0].any() and not d[:, -1].any()


def test_progress_is_monotonic_around_the_loop(circle_track):
    """Walking the centerline, wrapped progress deltas are all positive.

    Uses fewer query points (100) than the internal resampled centerline has
    samples (~390). Querying at a finer spacing than the resampling would make
    adjacent queries land on the same sample and produce a zero delta, which
    would fail this test without anything actually being wrong.
    """
    pts = circle_centerline(600, 400, 250, 100)
    vals = circle_track.progress[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    d = np.diff(vals)
    d = (d + 0.5) % 1.0 - 0.5
    assert (d > 0).all(), f"{(d <= 0).sum()} non-increasing steps"


def test_progress_spans_the_full_range(circle_track):
    p = circle_track.progress[circle_track.drivable]
    assert p.min() < 0.02 and p.max() > 0.98


def test_start_pose_is_on_track(circle_track):
    x, y, _ = circle_track.start_pose
    assert circle_track.drivable[int(y), int(x)]


def test_checkpoints_are_quantised_progress(circle_track):
    t = circle_track
    m = t.drivable
    assert np.all(t.checkpoint[m] == (t.progress[m] * t.n_checkpoints).astype(np.int16)
                  % t.n_checkpoints)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builtin_track_is_valid(name):
    cfg = Config()
    t = load(name, cfg)
    assert t.drivable.any()
    x, y, _ = t.start_pose
    assert t.drivable[int(y), int(x)], "start pose is off-track"
    # Track must fit inside the world with a margin. Without this, a centerline
    # that runs off the edge gets silently clipped into a wall and the cars
    # train against a track that is not the shape you drew.
    assert t.drivable[2:-2, 2:-2].sum() == t.drivable.sum()


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builtin_track_progress_is_usable(name):
    """Progress must span the full lap on every track, not just the circle."""
    t = load(name, Config())
    p = t.progress[t.drivable]
    assert p.min() < 0.05 and p.max() > 0.95


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_track_does_not_self_intersect(name):
    """Two parts of the track that are far apart along the lap must not come
    within a track width of each other.

    This is not cosmetic. Progress is one value per pixel, so where two parts
    of the track overlap, a pixel cannot say which part of the lap it belongs
    to -- a car driving through the overlap gets a garbage progress delta and
    its fitness becomes meaningless. A figure-eight fails this with 2px of
    separation, which is exactly why there isn't one.
    """
    cfg = Config()
    c = load(name, cfg).centerline
    closest = self_approach_distance(c, cfg.track_width)
    assert closest > cfg.track_width, (
        f"{name} folds back on itself: closest distant approach is "
        f"{closest:.1f}px, needs > {cfg.track_width}px")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_car_cannot_ride_the_wall(name):
    """A car's body must not be allowed to overlap a wall.

    The very first random population discovered that pinning itself against
    the outer wall of a corner was the cheapest way round, because collision
    only tested the car's centre point. body_ok closes that: every legal centre
    is at least one car radius from the tarmac edge.
    """
    cfg = Config()
    t = load(name, cfg)
    assert t.body_ok.sum() < t.drivable.sum(), "body mask is not tighter at all"
    assert not (t.body_ok & ~t.drivable).any(), "body mask escapes the tarmac"

    # Every legal centre must have a full car radius of tarmac around it.
    ys, xs = np.nonzero(t.body_ok)
    take = slice(None, None, max(1, len(xs) // 4000))
    ys, xs = ys[take], xs[take]
    r = int(cfg.car_radius)
    for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
        assert t.drivable[np.clip(ys + dy, 0, cfg.height - 1),
                          np.clip(xs + dx, 0, cfg.width - 1)].all(), (
            f"{name}: a legal centre has wall within {r}px at offset {(dx, dy)}")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_start_pose_leaves_room_for_the_car(name):
    t = load(name, Config())
    x, y, _ = t.start_pose
    assert t.body_ok[int(y), int(x)], "car spawns already clipping a wall"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_track_start_is_not_on_a_blind_corner(name):
    """Every ray from the start pose must return something finite, and the car
    must not be spawned facing a wall it cannot avoid."""
    from src.sensors import cast
    cfg = Config()
    t = load(name, cfg)
    x, y, h = t.start_pose
    rays = cast(np.float32([[x, y]]), np.float32([h]), t.drivable, cfg)
    assert np.isfinite(rays).all()
    assert rays[0, cfg.n_rays // 2] > 0.15, "spawned nose-first into a wall"
