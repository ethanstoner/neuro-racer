"""Reversed tracks and two-direction training."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from config import Config
from src.tracks import load

CFG = Config()
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["snake", "oval"])
def test_reversed_track_is_the_same_loop_driven_the_other_way(name):
    f, r = load(name, CFG), load(name, CFG, reverse=True)
    assert r.name == f"{name}-reversed"
    assert r.start_pose[:2] == pytest.approx(f.start_pose[:2])
    assert abs(np.cos(r.start_pose[2] - f.start_pose[2]) + 1) < 1e-3, "heading must flip"
    assert r.length == pytest.approx(f.length, rel=1e-6)
    np.testing.assert_array_equal(r.drivable, f.drivable)


def test_both_directions_training_records_its_direction(tmp_path):
    out = tmp_path / "run"
    subprocess.run([sys.executable, "train.py", "--track", "oval", "--direction", "both",
                    "--generations", "2", "--population", "8", "--out", str(out), "--quiet"],
                   cwd=ROOT, check=True, capture_output=True)
    assert json.loads((out / "config.json").read_text())["direction"] == "both"
    assert len(json.loads((out / "champions.json").read_text())) == 2


def test_training_on_several_tracks_records_them_all(tmp_path):
    out = tmp_path / "run"
    subprocess.run([sys.executable, "train.py", "--track", "oval", "chicane",
                    "--generations", "1", "--population", "8", "--out", str(out), "--quiet"],
                   cwd=ROOT, check=True, capture_output=True)
    info = json.loads((out / "config.json").read_text())
    assert info["tracks"] == ["oval", "chicane"] and info["track"] == "oval"


def test_random_poses_fit_the_car_and_face_along_the_track():
    from src.robustness import random_pose
    track = load("snake", CFG)
    rng = np.random.default_rng(0)
    poses = [random_pose(track, CFG, rng) for _ in range(200)]
    c = track.centerline
    for x, y, h in poses:
        assert track.body_ok[int(y), int(x)]
        i = int(np.argmin(np.linalg.norm(c - [x, y], axis=1)))
        t = c[(i + 3) % len(c)] - c[i]
        assert np.cos(h - np.arctan2(t[1], t[0])) > 0.9, "must face along the lap"
    # spread round the lap, not clustered at the start line
    assert np.ptp([p[0] for p in poses]) > 700
    again = [random_pose(track, CFG, np.random.default_rng(0)) for _ in range(1)]
    np.testing.assert_array_equal(again[0], poses[0])


def test_random_start_training_records_it(tmp_path):
    out = tmp_path / "run"
    subprocess.run([sys.executable, "train.py", "--track", "oval", "--random-starts",
                    "--generations", "2", "--population", "8", "--out", str(out), "--quiet"],
                   cwd=ROOT, check=True, capture_output=True)
    assert json.loads((out / "config.json").read_text())["random_starts"] is True
