"""The held-out experiment rests entirely on this harness, so it needs its own
tests rather than being trusted because its numbers look plausible.

The failure mode that would quietly invalidate every result: spawning cars
somewhere they cannot survive regardless of skill (inside a wall, or facing
across the track). Then every champion scores 0% and the conclusion would be
"nothing generalises" when the truth is "the harness is broken".
"""
from pathlib import Path
import numpy as np
import pytest
from src.config import Config
from src.tracks import load, BUILDERS
from src.robustness import spawn_grid, assess
from src.artifacts import RunRecorder

CFG = Config()
CHAMPIONS = Path(__file__).resolve().parent.parent / "data" / "champions"


@pytest.fixture(scope="module")
def snake_champion():
    return RunRecorder.load_champion(CHAMPIONS / "snake")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_spawn_pose_is_on_the_track(name):
    """A pose the car cannot even be placed at is not a failed start, it is a
    broken one. assess() filters these out; if the filter ever has nothing to
    do, so much the better."""
    track = load(name, CFG)
    poses = spawn_grid(track, CFG, 24, 3)
    ix = poses[:, 0].astype(int)
    iy = poses[:, 1].astype(int)
    assert track.drivable[iy, ix].all(), f"{name}: spawn poses landed off-track"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_spawn_poses_face_along_the_track(name):
    """Facing across the track turns a robustness test into a demolition derby.

    One step at cruising speed from each pose must stay on the track, which is
    only true if the heading follows the centerline tangent.
    """
    track = load(name, CFG)
    poses = spawn_grid(track, CFG, 24, 3)
    step = 20.0
    x = poses[:, 0] + np.cos(poses[:, 2]) * step
    y = poses[:, 1] + np.sin(poses[:, 2]) * step
    inside = track.drivable[y.astype(int), x.astype(int)]
    assert inside.mean() > 0.95, f"{name}: {(~inside).sum()} poses face off-track"


def test_offsets_straddle_the_centerline():
    """With three offsets the middle one is the centerline itself, and the
    outer two must sit on opposite sides -- otherwise 'lateral robustness' is
    only ever tested in one direction."""
    track = load("oval", CFG)
    poses = spawn_grid(track, CFG, 4, 3)
    c = track.centerline
    for i in range(0, len(poses), 3):
        d = [np.min(np.linalg.norm(c - poses[i + k, :2], axis=1)) for k in range(3)]
        assert d[1] < d[0] and d[1] < d[2], "middle offset is not the centerline"


def test_assess_is_deterministic(snake_champion):
    """The experiment reports a single number per cell. If the harness were
    stochastic those numbers would not be comparable across champions."""
    track = load("snake", CFG)
    a = assess(snake_champion, track, CFG, 12, 3)
    b = assess(snake_champion, track, CFG, 12, 3)
    assert a == b


def test_snake_champion_still_laps_its_own_track(snake_champion):
    """A regression guard on the committed champion: if this drops, the
    published held-out table is measuring something other than what it says."""
    track = load("snake", CFG)
    r = assess(snake_champion, track, CFG, 24, 3)
    assert r["starts"] > 50
    assert r["lapped"] / r["starts"] > 0.9


def test_assess_reports_zero_rather_than_dividing_by_zero():
    """A track where nothing can spawn must report 0%, not raise."""
    track = load("oval", CFG)
    dud = np.zeros(CFG.genome_size, dtype=np.float32)
    r = assess(dud, track, CFG, 24, 3)
    assert r["lapped"] == 0
    assert r["rate"] == 0.0
