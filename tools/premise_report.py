"""Print the cornering numbers that justify the physics tuning.

These are the figures quoted in the devlog: how tight the car can turn at each
speed, versus how tight each track's worst corner actually is.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_premise import measured_turn_radius, min_corner_radius  # noqa: E402
from src.tracks import BUILDERS  # noqa: E402
from config import DEFAULT as CFG  # noqa: E402

print("turn radius achievable at full steering lock")
for v in (90, 150, 210, 270, 330, 420):
    print(f"   {v:4d} px/s  ->  {measured_turn_radius(float(v)):6.0f} px")

print("\ntightest corner on each track")
tight = {}
for n in sorted(BUILDERS):
    tight[n] = min_corner_radius(n)
    print(f"   {n:9} {tight[n]:6.0f} px")

print("\nmax speed each track's worst corner can be taken at")
for n in sorted(BUILDERS):
    ok = [v for v in range(60, int(CFG.max_speed) + 1, 5)
          if measured_turn_radius(float(v)) < tight[n]]
    best = max(ok) if ok else None
    if best is None:
        print(f"   {n:9}  undriveable")
    else:
        print(f"   {n:9} {best:4d} px/s  "
              f"({best / CFG.max_speed * 100:.0f}% of top speed)")
