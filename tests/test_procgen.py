"""The generated held-out set: reproducible, valid, and not secretly easy."""
import numpy as np
import pytest
from src.config import Config
from src.procgen import generate
from src.track import Track

CFG = Config()


@pytest.fixture(scope="module")
def tracks():
    return generate(40, seed=7, cfg=CFG)


def test_same_seed_same_tracks(tracks):
    again = generate(40, seed=7, cfg=CFG)
    assert [t.name for t in again] == [t.name for t in tracks]
    for a, b in zip(again, tracks):
        np.testing.assert_array_equal(a.centerline, b.centerline)


def test_different_seed_different_tracks(tracks):
    other = generate(5, seed=8, cfg=CFG)
    assert not np.allclose(other[0].centerline, tracks[0].centerline)


def test_every_track_passes_the_editor_rules(tracks):
    for t in tracks:
        assert t.check.ok, (t.name, t.check.problems)
        assert t.check.reverse_fraction >= 0.03, "an oval in disguise got through"


def test_difficulty_spans_both_champion_floors(tracks):
    """The set is only a test of the corner floors if it has tracks either side of them."""
    r = np.array([t.check.tightest_radius for t in tracks])
    assert (r < 51).any() and ((r >= 51) & (r < 131)).any()
    assert r.max() > 110


def test_starts_and_directions_vary(tracks):
    headings = []
    for t in tracks[:10]:
        rasterised = Track.from_centerline(t.centerline, CFG, name=t.name)
        x, y, h = rasterised.start_pose
        assert rasterised.body_ok[int(y), int(x)]
        headings.append(h)
    assert np.ptp(headings) > 1.0
