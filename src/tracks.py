"""Three tracks of deliberately increasing difficulty.

oval     -- every corner turns the same way. The main training track, and the
            one that produces the most interesting failure: a network can score
            well here by evolving a constant steering bias and never really
            reading its sensors at all.
            Worst corner 161px radius, takeable at 270px/s (64% of top speed).
chicane  -- wider, faster loop with one reverse-curvature section.
            Worst corner 172px radius, takeable at 280px/s (67% of top speed).
snake    -- a long loop with S-curves top and bottom, so roughly a third of the
            lap turns the "wrong" way, and by some margin the hardest of the
            three. This is the generalisation test: an oval-trained champion
            has no answer for it.
            Worst corner 70px radius, takeable at only 145px/s (35% of top
            speed) -- a car that has only ever learned to floor it cannot get
            round this at all.

Numbers from `tools/premise_report.py`, enforced by `tests/test_premise.py`.

A track must NOT self-intersect. Progress is a single value per pixel, so where
two parts of the track overlap the progress map is ambiguous and cars driving
through the overlap get nonsense progress deltas. `test_track_does_not_self_
intersect` enforces this; `tools/probe_tracks.py` scores candidate shapes.
"""
import numpy as np
from src.config import Config
from src.track import Track
from src.track_io import is_track_path, load_file, track_stem

_T = np.linspace(0, 2 * np.pi, 600, endpoint=False)


def _oval() -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    return np.stack([600 + 420 * np.cos(t), 400 + 260 * np.sin(t)], axis=1)


def _snake() -> np.ndarray:
    t = _T
    return np.stack([600 + 430 * np.cos(t),
                     400 + 210 * np.sin(t) + 95 * np.sin(3 * t)], axis=1)


def _chicane() -> np.ndarray:
    t = _T
    return np.stack([600 + 440 * np.cos(t) + 40 * np.cos(3 * t),
                     400 + 240 * np.sin(t) + 80 * np.sin(3 * t)], axis=1)


def _clover() -> np.ndarray:
    """Three lobes with concave joins -- reverse curvature three times a lap."""
    t = _T
    r = 250 + 70 * np.cos(3 * t)
    return np.stack([600 + r * np.cos(t) * 1.55, 400 + r * np.sin(t) * 1.05], axis=1)


def _peanut() -> np.ndarray:
    """Two big lobes pinched in the middle. Fast, with one hard direction change."""
    t = _T
    r = 270 + 90 * np.cos(2 * t)
    return np.stack([600 + r * np.cos(t) * 1.45, 400 + r * np.sin(t) * 0.95], axis=1)


def _ripple() -> np.ndarray:
    """Four shallow waves round a wide loop -- constant small corrections."""
    t = _T
    r = 260 + 60 * np.cos(4 * t)
    return np.stack([600 + r * np.cos(t) * 1.5, 400 + r * np.sin(t) * 1.0], axis=1)


def _keyhole() -> np.ndarray:
    """A wide fast sweep narrowing into one tight end.

    The first attempt here was a "teardrop" (r = 300 - 110cos t) which probed
    out as 186px off-centre with 0% reverse curvature and a 251px worst corner
    -- an oval in disguise, which is the one thing a held-out track must not
    be. Genuine hairpins all pinched below 35px, tighter than the car can turn
    at any speed, so they were undriveable rather than hard. (Devlog 15 later
    found that centerline radius alone doesn't decide this: champions lap 12px
    centerline corners where the road leaves room for a wider line.)
    """
    t = _T
    r_out = 330 - 60 * np.cos(2 * t)
    r_in = 300 - 150 * np.cos(t)
    return np.stack([600 + r_out * np.cos(t) * 1.3,
                     400 + r_in * np.sin(t) * 0.8], axis=1)


BUILDERS = {
    "oval": _oval,
    "chicane": _chicane,
    "snake": _snake,
    "clover": _clover,
    "peanut": _peanut,
    "ripple": _ripple,
    "keyhole": _keyhole,
}

# Tracks used for training runs. The rest are held out as unseen test tracks --
# a champion that has never encountered them is the only honest way to tell a
# learned policy from a memorised trajectory.
TRAINING = ("oval", "chicane", "snake")
HELD_OUT = ("clover", "peanut", "ripple", "keyhole")


def slug(name: str) -> str:
    """Filesystem-safe label for a track name or track-file path, for run and image names."""
    return track_stem(name) if is_track_path(name) else name


def reverse_centerline(points: np.ndarray) -> np.ndarray:
    """Same loop, same start point, driven the other way round."""
    return np.vstack([points[:1], points[:0:-1]])


def load(name: str, cfg: Config, reverse: bool = False) -> Track:
    """A built-in track by name, or a track file by path (anything ending .json).

    Every built-in runs clockwise on screen. `reverse=True` drives it the other
    way, which is how devlog 09 found the snake champion only knows one of them.
    """
    if is_track_path(name):
        track = load_file(name, cfg)
        if not reverse:
            return track
        points = track.centerline
    elif name in BUILDERS:
        points = BUILDERS[name]()
    else:
        raise KeyError(f"unknown track {name!r}; have {sorted(BUILDERS)}")
    if reverse:
        return Track.from_centerline(reverse_centerline(points), cfg, name=f"{slug(name)}-reversed")
    return Track.from_centerline(points, cfg, name=name)
