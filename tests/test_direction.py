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
