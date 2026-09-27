"""Did the champion learn a policy, or memorise one trajectory?

Drops a champion at many poses around the lap and counts how many it survives.
The harness itself lives in `src/robustness.py`; this is the CLI over it.

  python -m experiments.robustness --run data/champions/oval
  python -m experiments.robustness --run data/champions/snake --all-tracks
"""
import argparse
import numpy as np
from src.config import Config
from src.tracks import load, BUILDERS
from src.artifacts import RunRecorder
from src.robustness import assess

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--track", default=None)
p.add_argument("--points", type=int, default=24, help="spawn points around the lap")
p.add_argument("--offsets", type=int, default=3, help="lateral offsets per point")
p.add_argument("--all-tracks", action="store_true")
args = p.parse_args()

champs = RunRecorder.load_all(args.run)
entry = champs[-1]
genome = np.array(entry["genome"], dtype=np.float32)
cfg = Config()

trained_on = entry.get("track", "?")
targets = sorted(BUILDERS) if args.all_tracks else [args.track or trained_on]

print(f"champion gen {entry['generation']}, trained on {trained_on}")
print(f"{'track':9} {'starts':>7} {'survived':>9} {'completed a lap':>16} {'best':>8}")
for name in targets:
    r = assess(genome, load(name, cfg), cfg, args.points, args.offsets)
    lap = f"{r['best_lap']:.2f}s" if r["best_lap"] else "--"
    print(f"{r['track']:9} {r['starts']:7d} {r['survived']:9d} "
          f"{r['lapped']:9d} ({100 * r['rate']:3.0f}%) {lap:>8}")
