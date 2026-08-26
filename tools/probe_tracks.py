"""Scratch tool: score candidate track shapes before committing them.

Checks the three things that make a shape usable:
  separation  -- how close two arc-length-distant parts of the track come to
                 each other. Must exceed the track width or the branches
                 overlap and the progress map becomes ambiguous.
  reverse     -- fraction of the lap spent turning the "other" way. A shape
                 with 0% is an oval in disguise and teaches a constant steering
                 bias rather than a policy.
  bbox        -- must fit the world with margin.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.track import _resample_closed


def score(points, width=90.0, world=(1200, 800)):
    c = _resample_closed(points, spacing=4.0)
    m = len(c)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    total = seg.sum()
    s = np.concatenate([[0.0], np.cumsum(seg)])[:m]

    # pairwise euclidean vs arc-length distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    arc = np.abs(s[:, None] - s[None, :])
    arc = np.minimum(arc, total - arc)
    distant = arc > width * 1.5
    separation = d[distant].min() if distant.any() else np.inf

    # signed curvature via cross product of successive tangents
    t = np.diff(np.vstack([c, c[:2]]), axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    reverse = (cross < -1e-4).sum() / len(cross)
    reverse = min(reverse, 1 - reverse)

    lo, hi = c.min(0), c.max(0)
    margin = min(lo[0], lo[1], world[0] - hi[0], world[1] - hi[1]) - width / 2
    return separation, reverse, margin, total


CANDIDATES = {}


def add(name, fn):
    CANDIDATES[name] = fn


t = np.linspace(0, 2 * np.pi, 600, endpoint=False)

add("figure8", lambda: np.stack([600 + 400 * np.sin(t), 400 + 260 * np.sin(t) * np.cos(t)], 1))
add("grand_old", lambda: np.stack([600 + (300 + 90 * np.sin(3 * t) - 40 * np.cos(2 * t)) * np.cos(t) * 1.25,
                                   400 + (300 + 90 * np.sin(3 * t) - 40 * np.cos(2 * t)) * np.sin(t) * 0.82], 1))
add("clover3", lambda: np.stack([600 + (250 + 70 * np.cos(3 * t)) * np.cos(t) * 1.55,
                                 400 + (250 + 70 * np.cos(3 * t)) * np.sin(t) * 1.05], 1))
add("peanut", lambda: np.stack([600 + (270 + 90 * np.cos(2 * t)) * np.cos(t) * 1.45,
                                400 + (270 + 90 * np.cos(2 * t)) * np.sin(t) * 0.95], 1))
add("snake", lambda: np.stack([600 + 430 * np.cos(t),
                               400 + 210 * np.sin(t) + 95 * np.sin(3 * t)], 1))
add("chicane", lambda: np.stack([600 + 440 * np.cos(t) + 40 * np.cos(3 * t),
                                 400 + 240 * np.sin(t) + 80 * np.sin(3 * t)], 1))
add("wave4", lambda: np.stack([600 + (260 + 60 * np.cos(4 * t)) * np.cos(t) * 1.5,
                               400 + (260 + 60 * np.cos(4 * t)) * np.sin(t) * 1.0], 1))

print(f"{'name':12} {'separation':>11} {'reverse%':>9} {'margin':>8} {'length':>8}  verdict")
for name, fn in CANDIDATES.items():
    sep, rev, mar, ln = score(fn())
    ok = sep > 90 and mar > 0 and rev > 0.05
    print(f"{name:12} {sep:11.1f} {rev * 100:8.1f}% {mar:8.1f} {ln:8.0f}  "
          f"{'OK' if ok else 'reject'}")
