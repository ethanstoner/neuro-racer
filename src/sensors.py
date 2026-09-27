"""Raycasting by sampling the drivable mask.

There is deliberately no ray/segment intersection maths here. We march a fixed
number of samples along each ray and take the first one that lands on a wall.
It is fully vectorised over (cars x rays x samples) as a single gather, and it
has no geometric edge cases to get subtly wrong.
"""
import numpy as np
from src.config import Config


def cast(pos: np.ndarray, angle: np.ndarray,
         drivable: np.ndarray, cfg: Config) -> np.ndarray:
    """Return (N, n_rays) distances normalised to [0, 1].

    1.0 means the ray reached ray_length without hitting anything.
    """
    n = len(pos)
    ang = angle[:, None] + cfg.ray_angles[None, :]          # (N, R)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=2)     # (N, R, 2)

    t = np.linspace(0.0, cfg.ray_length, cfg.ray_samples, dtype=np.float32)
    pts = pos[:, None, None, :] + dirs[:, :, None, :] * t[None, None, :, None]

    h, w = drivable.shape
    # Out-of-range samples clip onto the mask border, which is always non-drivable,
    # so a car outside the world reads walls instead of raising IndexError.
    ix = np.clip(pts[..., 0].astype(np.int32), 0, w - 1)
    iy = np.clip(pts[..., 1].astype(np.int32), 0, h - 1)
    solid = ~drivable[iy, ix]                               # (N, R, S)

    hit_any = solid.any(axis=2)
    first = solid.argmax(axis=2)                            # 0 when nothing hit
    dist = np.where(hit_any, t[first], cfg.ray_length)
    return (dist / cfg.ray_length).astype(np.float32).reshape(n, cfg.n_rays)
