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


def min_centerline_radius(centerline: np.ndarray) -> float:
    """Tightest radius of curvature anywhere on a centerline, in pixels.

    The single number that predicts how hard a track is to drive, so it is used
    both to assert the tracks are interesting (tests/test_premise.py) and to
    order them in the held-out experiment.

    Takes a centerline rather than a track name so that candidate shapes can be
    measured before they are ever rasterised.
    """
    d1 = np.gradient(centerline, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    curvature = num / np.maximum(den, 1e-12)
    return float(1.0 / np.maximum(curvature.max(), 1e-12))


def self_approach_distance(centerline: np.ndarray, track_width: float) -> float:
    """How close the track comes to itself, ignoring neighbours along the lap.

    Progress is one value per pixel, so where two distant parts of the lap
    overlap, a pixel cannot say which part it belongs to and cars driving
    through the overlap get garbage progress deltas. Anything at or below
    `track_width` is unusable -- which is why there is no figure-eight.
    """
    seg = np.linalg.norm(np.diff(np.vstack([centerline, centerline[:1]]), axis=0), axis=1)
    total = seg.sum()
    s = np.concatenate([[0.0], np.cumsum(seg)])[:len(centerline)]

    euclid = np.linalg.norm(centerline[:, None, :] - centerline[None, :, :], axis=2)
    arc = np.abs(s[:, None] - s[None, :])
    arc = np.minimum(arc, total - arc)          # circular distance along the lap

    distant = arc > track_width * 1.5
    if not distant.any():
        return float("inf")
    return float(euclid[distant].min())


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
    drivable: np.ndarray        # (H, W) bool -- the tarmac, as drawn and sensed
    body_ok: np.ndarray         # (H, W) bool -- centres where the whole car fits
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

        # Where the car's whole body fits, not just its centre point.
        #
        # Collision has to account for car size or a car can ride with half its
        # body through the wall at zero cost -- and it will, because hugging the
        # outer wall of a smooth corner is the cheapest way to get round it.
        # The very first random population found exactly that exploit.
        #
        # Since the track is by construction "everything within half-width of
        # the centerline", shrinking the half-width by the car radius gives
        # precisely the set of centres where a disc-shaped car fits. No
        # morphological erosion needed, and it costs nothing at runtime.
        inner = half - cfg.car_radius
        if inner <= 1.0:
            raise ValueError(
                f"track_width {cfg.track_width} is too narrow for a car of "
                f"radius {cfg.car_radius}: nothing would be driveable")
        body_ok = best_d2 <= np.float32(inner * inner)

        # Hard border so out-of-bounds ray samples read as wall.
        drivable[0, :] = drivable[-1, :] = False
        drivable[:, 0] = drivable[:, -1] = False
        body_ok[0, :] = body_ok[-1, :] = False
        body_ok[:, 0] = body_ok[:, -1] = False

        progress = (best_i.astype(np.float32) / m)
        progress[~drivable] = 0.0

        checkpoint = ((progress * n_checkpoints).astype(np.int16) % n_checkpoints)
        checkpoint[~drivable] = -1

        # Start at centerline[0], heading towards centerline[1].
        d = center[1] - center[0]
        start = (float(center[0][0]), float(center[0][1]), float(np.arctan2(d[1], d[0])))

        seg = np.linalg.norm(np.diff(np.vstack([center, center[:1]]), axis=0), axis=1)
        return cls(name, center, drivable, body_ok, progress, checkpoint,
                   start, n_checkpoints, float(seg.sum()))
