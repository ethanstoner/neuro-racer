"""Where does each champion actually break?

The held-out run falsified the tidy version of the claim -- champions cleared
tracks tighter than anything they trained on. But they did not clear everything,
so there is still a floor somewhere; the four held-out tracks are just too
coarse a ruler to find it (they cluster at 51, 52, 100 and 108px).

This sweeps a family of tracks that differ only in corner tightness and finds,
for each champion, the tightest corner it can still lap. That number is the
generalisation floor, and it is what the README should be quoting instead of
the training ceiling.

  python tools/corner_sweep.py
  python tools/corner_sweep.py --json docs/results/corner-sweep.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import Config                                        # noqa: E402
from src.track import Track, min_centerline_radius, self_approach_distance  # noqa: E402
from src.tracks import TRAINING                                  # noqa: E402
from src.artifacts import RunRecorder                            # noqa: E402
from src.robustness import assess                                # noqa: E402

_T = np.linspace(0, 2 * np.pi, 600, endpoint=False)


def wave(amplitude: float) -> np.ndarray:
    """A four-lobed loop whose corner tightness is set by `amplitude` alone.

    Everything else -- overall size, lap length, number of direction changes --
    is held as close to constant as the shape allows, so that a champion's
    success or failure across the sweep is attributable to corner radius and
    not to the track becoming a different kind of track.
    """
    r = 260 + amplitude * np.cos(4 * _T)
    return np.stack([600 + r * np.cos(_T) * 1.5, 400 + r * np.sin(_T) * 1.0], axis=1)


p = argparse.ArgumentParser()
p.add_argument("--champions", default="data/champions")
p.add_argument("--points", type=int, default=16)
p.add_argument("--offsets", type=int, default=3)
p.add_argument("--json", default=None)
args = p.parse_args()

cfg = Config()
root = Path(ROOT) / args.champions

# Build the sweep, keeping only shapes that are legitimate tracks. A track that
# folds back on itself would fail for reasons that have nothing to do with the
# champion, and would look exactly like a generalisation failure in the output.
rungs = []
for amp in np.arange(0.0, 96.0, 6.0):
    c = wave(float(amp))
    if self_approach_distance(c, cfg.track_width) <= cfg.track_width:
        continue
    track = Track.from_centerline(c, cfg, name=f"wave{amp:.0f}")
    if track.drivable[2:-2, 2:-2].sum() != track.drivable.sum():
        continue        # runs off the edge of the world
    rungs.append((min_centerline_radius(track.centerline), track))

rungs.sort(key=lambda r: -r[0])     # widest corners first
print(f"{len(rungs)} rungs, corner radius "
      f"{rungs[0][0]:.0f}px down to {rungs[-1][0]:.0f}px\n")

results = {}
for champ in TRAINING:
    genome = RunRecorder.load_champion(root / champ)
    print(f"{champ} champion")
    floor = None
    for radius, track in rungs:
        r = assess(genome, track, cfg, args.points, args.offsets)
        passed = r["rate"] > 0.5
        print(f"   corner {radius:5.0f}px   {r['lapped']:3d}/{r['starts']:3d} "
              f"({100 * r['rate']:3.0f}%)  {'pass' if passed else 'fail'}")
        results.setdefault(champ, []).append(
            {"radius": radius, "rate": r["rate"], "passed": passed})
        if passed:
            floor = radius
        else:
            break       # once it fails, tighter corners only get worse
    print(f"   -> floor {floor:.0f}px\n" if floor else "   -> no rung cleared\n")

if args.json:
    out = Path(ROOT) / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
