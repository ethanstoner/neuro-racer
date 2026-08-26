"""A track is a centerline polyline plus a width, rasterised into three arrays.

Doing it this way means collision, fitness progress and raycasting are all just
array lookups -- there is no wall geometry anywhere in the hot loop.
"""
from dataclasses import dataclass
import numpy as np
from config import Config


def circle_centerline(cx: float, cy: float, r: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _resample_closed(points: np.ndarray, spacing: float = 4.0) -> np.ndarray:
    """Resample a closed polyline to roughly even spacing.

    Even spacing matters: progress is arc-length based, and unevenly spaced
    control points would make the progress map advance faster in some corners
    than others, which the cars would happily exploit.
    """
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(total / spacing), 16)
    targets = np.linspace(0, total, n, endpoint=False)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


@dataclass
class Track:
    name: str
    centerline: np.ndarray      # (M, 2), evenly spaced, closed loop
    drivable: np.ndarray        # (H, W) bool
    progress: np.ndarray        # (H, W) float32 in [0, 1)
    checkpoint: np.ndarray      # (H, W) int16
    start_pose: tuple           # (x, y, heading_radians)
    n_checkpoints: int
    length: float               # centerline arc length in pixels

    @classmethod
    def from_centerline(cls, points: np.ndarray, cfg: Config,
                        name: str = "unnamed", n_checkpoints: int = 32) -> "Track":
        center = _resample_closed(points, spacing=4.0)
        m = len(center)
        h, w = cfg.height, cfg.width
        half = cfg.track_width / 2.0

        # For every pixel: distance to the nearest centerline sample, and which
        # sample that was. Only pixels within half-width of some sample can ever
        # be on the track, so we stamp a local window per sample instead of
        # comparing all ~1M pixels against all ~400 samples. Same result, and
        # roughly fifty times faster -- this runs at startup on every launch.
        r = int(np.ceil(half)) + 1
        best_d2 = np.full((h, w), np.inf, dtype=np.float32)
        best_i = np.zeros((h, w), dtype=np.int32)

        for i, (cx, cy) in enumerate(center):
            x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, w)
            y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            dx = np.arange(x0, x1, dtype=np.float32) - np.float32(cx)
            dy = np.arange(y0, y1, dtype=np.float32) - np.float32(cy)
            d2 = dy[:, None] ** 2 + dx[None, :] ** 2
            window = best_d2[y0:y1, x0:x1]
            better = d2 < window
            window[better] = d2[better]
            best_i[y0:y1, x0:x1][better] = i

        drivable = best_d2 <= np.float32(half * half)

        # Hard border so out-of-bounds ray samples read as wall.
        drivable[0, :] = drivable[-1, :] = False
        drivable[:, 0] = drivable[:, -1] = False

        progress = (best_i.astype(np.float32) / m)
        progress[~drivable] = 0.0

        checkpoint = ((progress * n_checkpoints).astype(np.int16) % n_checkpoints)
        checkpoint[~drivable] = -1

        # Start at centerline[0], heading towards centerline[1].
        d = center[1] - center[0]
        start = (float(center[0][0]), float(center[0][1]), float(np.arctan2(d[1], d[0])))

        seg = np.linalg.norm(np.diff(np.vstack([center, center[:1]]), axis=0), axis=1)
        return cls(name, center, drivable, progress, checkpoint,
                   start, n_checkpoints, float(seg.sum()))
