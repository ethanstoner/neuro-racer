"""Track files: closed centerlines on disk, as written by the virtual-world editor.

A file is JSON with `format: "neuroracer-track"`, the arena size and track width
it was drawn for, and a `centerline` in driving order. The car starts at
centerline[0] heading towards centerline[1], exactly as for the built-in tracks.

Loading refuses anything the trainer could not use honestly -- a track that
overlaps itself, leaves the arena, or has a corner too tight to measure -- with
the same three rules the editor checks live, so a file that exported cleanly
always loads, and a hand-edited one that breaks a rule says which.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from config import Config
from src.track import (Track, _resample_closed, min_centerline_radius,
                       self_approach_distance)

FORMAT = "neuroracer-track"
VERSION = 1

# Below this the centerline radius stops describing the corner a car drives:
# champions lap 12px centerline corners by taking a far wider line (devlog 15).
# Kept at 40 because procgen's held-out sets depend on it, not because tighter
# tracks can't be driven.
MIN_RADIUS = 40.0


def reverse_fraction(centerline: np.ndarray) -> float:
    """Share of the lap turning against the dominant direction.

    Zero means every corner turns the same way: an oval in disguise, which can
    be scored on with a constant steering bias and no sensor reading at all.
    """
    c = centerline
    t = np.diff(np.vstack([c, c[:2]]), axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    r = (cross < -1e-4).sum() / len(cross)
    return float(min(r, 1 - r))


def arena_margin(centerline: np.ndarray, cfg: Config) -> float:
    lo, hi = centerline.min(0), centerline.max(0)
    return float(min(lo[0], lo[1], cfg.width - hi[0], cfg.height - hi[1])
                 - cfg.track_width / 2)


@dataclass
class TrackCheck:
    tightest_radius: float
    self_approach: float
    reverse_fraction: float
    length: float
    arena_margin: float
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check(points: np.ndarray, cfg: Config) -> TrackCheck:
    """Measure a raw centerline the way the trainer will see it (resampled at 4px)."""
    c = _resample_closed(np.asarray(points, dtype=np.float64), spacing=4.0)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    r = TrackCheck(
        tightest_radius=min_centerline_radius(c),
        self_approach=self_approach_distance(c, cfg.track_width),
        reverse_fraction=reverse_fraction(c),
        length=float(seg.sum()),
        arena_margin=arena_margin(c, cfg),
    )
    if r.self_approach <= cfg.track_width:
        r.problems.append(f"overlaps itself: closest approach {r.self_approach:.0f}px, "
                          f"needs > {cfg.track_width:.0f}px")
    if r.arena_margin <= 0:
        r.problems.append(f"leaves the {cfg.width}x{cfg.height} arena by {-r.arena_margin:.0f}px")
    if r.tightest_radius < MIN_RADIUS:
        r.problems.append(f"corner too tight to measure: radius {r.tightest_radius:.0f}px, "
                          f"needs >= {MIN_RADIUS:.0f}px")
    return r


class TrackFileError(ValueError):
    pass


def read(path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise TrackFileError(f"{path}: {e}") from e
    if data.get("format") != FORMAT:
        raise TrackFileError(f"{path}: not a {FORMAT} file")
    if data.get("version") != VERSION:
        raise TrackFileError(f"{path}: unsupported version {data.get('version')!r}")
    return data


def load_file(path, cfg: Config) -> Track:
    path = Path(path)
    data = read(path)
    world = tuple(data.get("world", ()))
    if world != (cfg.width, cfg.height) or data.get("track_width") != cfg.track_width:
        raise TrackFileError(
            f"{path}: drawn for a {world} arena at width {data.get('track_width')}, "
            f"but the config is ({cfg.width}, {cfg.height}) at width {cfg.track_width}")
    pts = np.asarray(data.get("centerline", []), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        raise TrackFileError(f"{path}: centerline must be a list of at least 4 [x, y] points")
    result = check(pts, cfg)
    if not result.ok:
        raise TrackFileError(f"{path}: " + "; ".join(result.problems))
    return Track.from_centerline(pts, cfg, name=data.get("name") or track_stem(path))


def save_file(path, name: str, centerline: np.ndarray, cfg: Config, source: str = "neuro-racer",
              extra: dict | None = None) -> dict:
    """Write a centerline in the shared format, with its measurements alongside.

    Measured after rounding to the 0.01px that goes on disk: the tightest-corner
    estimate moves by about a pixel under that rounding, and the file's numbers
    must be the ones a reader of the file gets.
    """
    pts = np.round(np.asarray(centerline, dtype=np.float64), 2)
    r = check(pts, cfg)
    data = {
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "world": [cfg.width, cfg.height],
        "track_width": cfg.track_width,
        "centerline": [[float(x), float(y)] for x, y in pts],
        "metrics": {
            "tightest_radius": round(r.tightest_radius, 2),
            "self_approach": round(r.self_approach, 2),
            "reverse_fraction": round(r.reverse_fraction, 4),
            "length": round(r.length, 2),
            "arena_margin": round(r.arena_margin, 2),
        },
        "source": source,
    }
    if extra:
        data |= extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def track_stem(path) -> str:
    name = Path(path).name
    for suffix in (".track.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_track_path(name: str) -> bool:
    return name.endswith(".json")
