"""Seeded procedural tracks, for held-out tests nobody hand-picked.

The seven built-in tracks were drawn by me, and the first conclusion drawn from
them did not survive four genuinely held-out tracks (docs/devlog/07). Four is
still a handful chosen by the same person. This generates as many as wanted
from a seed, under rules fixed before any champion is run on them.

Each track is a star-shaped loop: a radius that wobbles with the angle,

    r(t) = 1 + sum_k a_k cos(k t + phi_k),   k = 2..5

stretched to a random aspect ratio and scaled to fill the arena. Keeping r > 0
means the loop can never cross itself, so the overlap rule only ever rejects
near-misses, not tangles. Start point and driving direction are randomised too,
so a champion that only knows how to turn one way cannot hide.

A candidate is kept only if it passes the same checks as an editor export
(src/track_io.check) and turns both ways for at least `min_reverse` of the lap.
"""
from dataclasses import dataclass
import numpy as np
from config import Config
from src.track_io import TrackCheck, check

HARMONICS = (2, 3, 4, 5)
SAMPLES = 600


@dataclass
class Generated:
    name: str
    centerline: np.ndarray
    check: TrackCheck


def candidate(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, SAMPLES, endpoint=False)
    r = np.ones_like(t)
    wobble = rng.uniform(0.15, 1.0)          # overall roughness: gentle sweeps to hairpins
    for k in HARMONICS:
        if rng.random() < 0.65:
            r += wobble * rng.uniform(0.03, 0.3) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    r = np.maximum(r, 0.2)
    aspect = rng.uniform(1.15, 1.8)
    x, y = r * np.cos(t) * aspect, r * np.sin(t)

    pad = cfg.track_width / 2 + rng.uniform(15, 70)
    scale = min((cfg.width - 2 * pad) / np.ptp(x), (cfg.height - 2 * pad) / np.ptp(y))
    x = (x - (x.max() + x.min()) / 2) * scale + cfg.width / 2
    y = (y - (y.max() + y.min()) / 2) * scale + cfg.height / 2
    pts = np.stack([x, y], axis=1)

    pts = np.roll(pts, -int(rng.integers(SAMPLES)), axis=0)
    if rng.random() < 0.5:
        pts = np.vstack([pts[:1], pts[:0:-1]])      # same start, other direction
    return pts


def generate(n: int, seed: int, cfg: Config, min_reverse: float = 0.03,
             max_attempts: int = 50_000) -> list[Generated]:
    """The first `n` valid tracks from `seed`. Same seed, same tracks, same order."""
    rng = np.random.default_rng(seed)
    out: list[Generated] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"only {len(out)} valid tracks in {max_attempts} attempts")
        pts = candidate(rng, cfg)
        ck = check(pts, cfg)
        if ck.ok and ck.reverse_fraction >= min_reverse:
            out.append(Generated(f"gen{seed}-{len(out):03d}", pts, ck))
    return out
