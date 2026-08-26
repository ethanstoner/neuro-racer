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

    d1 = np.gradient(c, axis=0)
    d2v = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2v[:, 1] - d1[:, 1] * d2v[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    tightest = float(1.0 / max(float((num / np.maximum(den, 1e-12)).max()), 1e-12))

    lo, hi = c.min(0), c.max(0)
    margin = min(lo[0], lo[1], world[0] - hi[0], world[1] - hi[1]) - width / 2
    centred = abs((lo[0] + hi[0]) / 2 - world[0] / 2) + abs((lo[1] + hi[1]) / 2 - world[1] / 2)
    return separation, reverse, margin, total, tightest, centred


CANDIDATES = {}


def add(name, fn):
    CANDIDATES[name] = fn


t = np.linspace(0, 2 * np.pi, 600, endpoint=False)

add("figure8", lambda: np.stack([600 + 400 * np.sin(t), 400 + 260 * np.sin(t) * np.cos(t)], 1))
add("teardrop_old", lambda: np.stack([600 + (300 - 110 * np.cos(t)) * np.cos(t) * 1.15 - 60,
                                      400 + (300 - 110 * np.cos(t)) * np.sin(t) * 0.95], 1))
# Hairpin candidates: a long fast sweep into one genuinely tight end. The width
# of the loop is modulated so one end pinches without the shape folding over.
add("hairpin_a", lambda: np.stack([600 + 430 * np.cos(t),
                                   400 + (210 - 95 * np.cos(t)) * np.sin(t)], 1))
add("hairpin_b", lambda: np.stack([600 + 440 * np.cos(t),
                                   400 + (230 - 140 * np.cos(t)) * np.sin(t)], 1))
add("hairpin_c", lambda: np.stack([600 + 450 * np.cos(t) - 20,
                                   400 + (250 - 175 * np.cos(t)) * np.sin(t)], 1))
add("keyhole", lambda: np.stack([600 + (330 - 60 * np.cos(2 * t)) * np.cos(t) * 1.3,
                                 400 + (300 - 150 * np.cos(t)) * np.sin(t) * 0.8], 1))

print(f"{'name':14} {'separation':>11} {'reverse%':>9} {'margin':>8} "
      f"{'tightest':>9} {'offcentre':>10} {'length':>8}  verdict")
for name, fn in CANDIDATES.items():
    sep, rev, mar, ln, tight, off = score(fn())
    ok = sep > 90 and mar > 0 and tight > 40 and off < 60
    print(f"{name:14} {sep:11.1f} {rev * 100:8.1f}% {mar:8.1f} {tight:9.0f} "
          f"{off:10.0f} {ln:8.0f}  {'OK' if ok else 'reject'}")
