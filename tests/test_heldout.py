"""The published held-out result, pinned.

The README makes three claims about champions that are committed to this repo.
Anyone can re-run `heldout.py` and check them, but nothing stops a later tuning
change to the physics or the fitness from quietly moving the numbers and
leaving the README asserting something that is no longer true.

These run at reduced resolution -- the point is to catch a claim becoming
false, not to reproduce the table to the last percent.
"""
from pathlib import Path
import pytest
from config import Config
from src.tracks import load
from src.robustness import assess
from src.artifacts import RunRecorder

CFG = Config()
POINTS, OFFSETS = 8, 3
CHAMPIONS = Path(__file__).resolve().parent.parent / "champions"


@pytest.fixture(scope="module")
def champions():
    return {n: RunRecorder.load_champion(CHAMPIONS / n)
            for n in ("oval", "chicane", "snake")}


def rate(genome, track_name):
    return assess(genome, load(track_name, CFG), CFG, POINTS, OFFSETS)["rate"]


@pytest.mark.parametrize("track", ["clover", "peanut", "ripple", "keyhole"])
def test_the_oval_champion_generalises_to_nothing(champions, track):
    """It memorised a trajectory. No held-out track is drivable by it -- this
    is the claim the whole project is built around."""
    assert rate(champions["oval"], track) < 0.1


@pytest.mark.parametrize("track", ["clover", "peanut", "ripple", "keyhole"])
def test_the_snake_champion_generalises_to_every_held_out_track(champions, track):
    """Trained on the tightest corner, clears all four unseen tracks including
    two tighter than anything it ever saw."""
    assert rate(champions["snake"], track) > 0.9


def test_the_chicane_champion_sits_between_the_two(champions):
    """A real policy with a floor above the tightest held-out corners: it laps
    peanut (108px) and cannot lap ripple (51px).

    This pair is what falsified the original prediction, so if it ever stops
    holding, devlog 07 needs rewriting rather than patching.
    """
    assert rate(champions["chicane"], "peanut") > 0.9
    assert rate(champions["chicane"], "ripple") < 0.1


def reversed_rate(genome, track_name):
    """The same track, driven the other way: same start point, opposite direction."""
    return assess(genome, load(track_name, CFG, reverse=True), CFG, POINTS, OFFSETS)["rate"]


def test_the_snake_champion_only_drives_one_way_round(champions):
    """Every built-in track runs clockwise. Reversed, the champion that laps all
    seven of them laps none -- not even its own (devlog 09)."""
    assert rate(champions["snake"], "snake") > 0.9
    assert reversed_rate(champions["snake"], "snake") < 0.1


def test_the_chicane_champion_survives_reversal(champions):
    """The contrast that makes the snake result about snake, not the harness."""
    assert reversed_rate(champions["chicane"], "chicane") > 0.8
