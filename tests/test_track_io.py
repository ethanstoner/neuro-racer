"""Track files from the virtual-world editor, and the rules both sides enforce.

The fixture was exported from the editor's UI (Export button, template track),
so the parity test below checks the TypeScript port of the measurements against
the numpy originals on a real file, not on a shape made up in Python.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from config import Config
from src.track import circle_centerline
from src.track_io import TrackFileError, check, load_file, read, save_file
from src.tracks import load, slug

CFG = Config()
FIXTURE = Path(__file__).parent / "fixtures" / "editor-template.track.json"


def test_editor_export_loads_as_a_track():
    t = load_file(FIXTURE, CFG)
    assert t.name == "Editor Template"
    x, y, _ = t.start_pose
    assert t.body_ok[int(y), int(x)], "the car must fit at the start"


def test_editor_measurements_match_numpy():
    """The editor's verdict is only worth something if NeuroRacer agrees with it."""
    data = read(FIXTURE)
    r = check(np.asarray(data["centerline"]), CFG)
    for key, value in data["metrics"].items():
        assert getattr(r, key) == pytest.approx(value, abs=0.01), key


def test_load_by_path_through_the_normal_entry_point():
    assert load(str(FIXTURE), CFG).name == "Editor Template"
    assert slug(str(FIXTURE)) == "editor-template"
    assert slug("snake") == "snake"


def _write(tmp_path, **changes):
    data = json.loads(FIXTURE.read_text(encoding="utf-8")) | changes
    p = tmp_path / "t.track.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.mark.parametrize("changes, message", [
    ({"format": "gpx"}, "not a neuroracer-track"),
    ({"version": 2}, "unsupported version"),
    ({"track_width": 120}, "width 120"),
    ({"world": [1600, 900]}, "arena"),
    ({"centerline": [[0, 0], [1, 1]]}, "at least 4"),
])
def test_bad_files_are_refused_with_a_reason(tmp_path, changes, message):
    with pytest.raises(TrackFileError, match=message):
        load_file(_write(tmp_path, **changes), CFG)


def test_overlapping_track_is_refused(tmp_path):
    pinched = [[200, 300], [1000, 300], [1000, 360], [200, 360]]
    with pytest.raises(TrackFileError, match="overlaps itself"):
        load_file(_write(tmp_path, centerline=pinched), CFG)


def test_track_outside_the_arena_is_refused(tmp_path):
    big = circle_centerline(600, 400, 380).tolist()
    with pytest.raises(TrackFileError, match="leaves the"):
        load_file(_write(tmp_path, centerline=big), CFG)


def test_save_then_load_round_trips(tmp_path):
    ring = circle_centerline(600, 400, 250)
    data = save_file(tmp_path / "ring.track.json", "ring", ring, CFG)
    assert data["metrics"]["tightest_radius"] == pytest.approx(247.67, abs=0.01)
    t = load_file(tmp_path / "ring.track.json", CFG)
    assert t.name == "ring"
    assert t.length == pytest.approx(data["metrics"]["length"], abs=0.01)
