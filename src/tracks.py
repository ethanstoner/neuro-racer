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
from config import Config
from src.track import Track

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


BUILDERS = {"oval": _oval, "snake": _snake, "chicane": _chicane}


def load(name: str, cfg: Config) -> Track:
    if name not in BUILDERS:
        raise KeyError(f"unknown track {name!r}; have {sorted(BUILDERS)}")
    return Track.from_centerline(BUILDERS[name](), cfg, name=name)
