"""Three tracks of deliberately increasing difficulty.

oval      -- trivial, both corners the same way. Used for the main training run.
figure8   -- requires turning both directions. Tests generalisation.
grand     -- tight hairpin + long straight. Requires braking for corners, which
             is where a network that only ever learned "floor it" falls apart.
"""
import numpy as np
from config import Config
from src.track import Track


def _oval() -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    return np.stack([600 + 420 * np.cos(t), 400 + 260 * np.sin(t)], axis=1)


def _figure8() -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    return np.stack([600 + 400 * np.sin(t), 400 + 260 * np.sin(t) * np.cos(t)], axis=1)


def _grand() -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    r = 300 + 90 * np.sin(3 * t) - 40 * np.cos(2 * t)
    return np.stack([600 + r * np.cos(t) * 1.25, 400 + r * np.sin(t) * 0.82], axis=1)


BUILDERS = {"oval": _oval, "figure8": _figure8, "grand": _grand}


def load(name: str, cfg: Config) -> Track:
    if name not in BUILDERS:
        raise KeyError(f"unknown track {name!r}; have {sorted(BUILDERS)}")
    return Track.from_centerline(BUILDERS[name](), cfg, name=name)
