"""The held-out test: every champion against four tracks none of them trained on.

The three-champion table in the README is measured on the *training* tracks --
each champion run on the two its siblings trained on. That is suggestive, but
those tracks were chosen by me, partly because they made the point.

These four were built as a held-out set and never trained on by anything. Their
corner radii were measured before any champion was run on them, and the
resulting prediction was written down first, in docs/devlog/06-prediction.md:

    a champion laps a track iff that track's tightest corner is at or above
    the tightest corner it saw in training.

  python -m experiments.heldout                 # the four held-out tracks
  python -m experiments.heldout --all-tracks    # all seven, training set included
  python -m experiments.heldout --json docs/results/heldout.json
"""
import argparse
import json
from pathlib import Path
from src.config import Config
from src.track import min_centerline_radius
from src.tracks import load, TRAINING, HELD_OUT, BUILDERS
from src.artifacts import RunRecorder
from src.robustness import assess

p = argparse.ArgumentParser()
p.add_argument("--champions", default="data/champions", help="directory of champion runs")
p.add_argument("--points", type=int, default=24)
p.add_argument("--offsets", type=int, default=3)
p.add_argument("--all-tracks", action="store_true")
p.add_argument("--json", default=None, help="also write results here")
args = p.parse_args()

cfg = Config()
targets = sorted(BUILDERS) if args.all_tracks else list(HELD_OUT)
root = Path(args.champions)

# Rasterise each track once and reuse it across all three champions, so every
# champion is scored against a bit-identical track.
tracks = {name: load(name, cfg) for name in targets}
tight = {n: min_centerline_radius(t.centerline) for n, t in tracks.items()}
ceiling = {t: min_centerline_radius(load(t, cfg).centerline) for t in TRAINING}

rows, results = [], []
for champ in TRAINING:
    genome = RunRecorder.load_champion(root / champ)
    for name in targets:
        r = assess(genome, tracks[name], cfg, args.points, args.offsets)
        # The prediction, evaluated mechanically rather than by eye.
        predicted = tight[name] >= ceiling[champ]
        r |= {"champion": champ, "ceiling": ceiling[champ],
              "tightest": tight[name], "predicted_pass": predicted}
        results.append(r)
        rows.append(r)

print(f"held-out tracks: {', '.join(targets)}")
print(f"{'champion':9} {'ceiling':>8}  {'track':9} {'corner':>7} "
      f"{'starts':>7} {'lapped':>14} {'predicted':>10}  {'':4}")
for r in rows:
    passed = r["rate"] > 0.5
    mark = "ok" if passed == r["predicted_pass"] else "MISS"
    print(f"{r['champion']:9} {r['ceiling']:7.0f}p  {r['track']:9} "
          f"{r['tightest']:6.0f}p {r['starts']:7d} "
          f"{r['lapped']:6d} ({100 * r['rate']:3.0f}%) "
          f"{'pass' if r['predicted_pass'] else 'fail':>10}  {mark:4}")

hits = sum((r["rate"] > 0.5) == r["predicted_pass"] for r in rows)
print(f"\nprediction correct on {hits}/{len(rows)} champion-track pairs")

if args.json:
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
