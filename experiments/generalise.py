"""Train vs held-out: every champion on its own track, then on generated ones.

The held-out tracks come from src/procgen.py, so nobody picked them. What was
predicted before this ran is in docs/devlog/08-prediction-generated.md: a
champion laps a track iff its tightest corner is at or above that champion's
floor from the corner sweep (docs/results/corner-sweep.json).

  python -m experiments.generalise                              # 100 tracks, seed 7
  python -m experiments.generalise --n 30 --seed 3
  python -m experiments.generalise --extra my.track.json        # add editor-drawn tracks
  python -m experiments.generalise --export data/tracks/generated    # write the set as track files
  python -m experiments.generalise --json docs/results/generalisation.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config import Config
from src.artifacts import RunRecorder
from src.procgen import generate
from src.robustness import assess
from src.track import Track, min_centerline_radius
from src.track_io import save_file
from src.tracks import TRAINING, load

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=100)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--champions", default="data/champions")
p.add_argument("--points", type=int, default=24)
p.add_argument("--offsets", type=int, default=3)
p.add_argument("--extra", nargs="*", default=[], help="track files to add to the held-out set")
p.add_argument("--export", default=None, help="write the generated tracks here as .track.json")
p.add_argument("--json", default=None)
args = p.parse_args()

cfg = Config()
root = Path(args.champions)

sweep = json.loads(Path("docs/results/corner-sweep.json").read_text(encoding="utf-8"))
floor = {c: min((r["radius"] for r in rungs if r["passed"]), default=None) for c, rungs in sweep.items()}

generated = generate(args.n, args.seed, cfg)
if args.export:
    for g in generated:
        save_file(Path(args.export) / f"{g.name}.track.json", g.name, g.centerline, cfg,
                  source=f"neuro-racer procgen seed {args.seed}")
    print(f"wrote {len(generated)} tracks to {args.export}")

held_out: list[Track] = [Track.from_centerline(g.centerline, cfg, name=g.name) for g in generated]
held_out += [load(path, cfg) for path in args.extra]
tight = {t.name: min_centerline_radius(t.centerline) for t in held_out}


def passed(r):
    return r["rate"] > 0.5


results = {"seed": args.seed, "n": len(held_out), "floors": floor, "champions": {}}
print(f"held-out: {len(generated)} generated (seed {args.seed})"
      + (f" + {len(args.extra)} from files" if args.extra else ""))
print(f"{'champion':9} {'floor':>6}  {'train':>6}  {'held-out mean':>13}  {'tracks lapped':>13}  "
      f"{'predicted':>9}  {'accuracy':>8}")

for champ in TRAINING:
    genome = RunRecorder.load_champion(root / champ)
    train = assess(genome, load(champ, cfg), cfg, args.points, args.offsets)
    rows = []
    for t in held_out:
        r = assess(genome, t, cfg, args.points, args.offsets)
        f = floor[champ]
        r |= {"tightest": tight[t.name], "predicted_pass": f is not None and tight[t.name] >= f}
        rows.append(r)
    lapped = sum(passed(r) for r in rows)
    predicted = sum(r["predicted_pass"] for r in rows)
    correct = sum(passed(r) == r["predicted_pass"] for r in rows)
    wider_fail = sum(r["predicted_pass"] and not passed(r) for r in rows)
    tighter_pass = sum(passed(r) and not r["predicted_pass"] for r in rows)
    mean = float(np.mean([r["rate"] for r in rows]))
    fl = f"{floor[champ]:.0f}p" if floor[champ] else "none"
    print(f"{champ:9} {fl:>6}  {100 * train['rate']:5.0f}%  {100 * mean:12.1f}%  "
          f"{lapped:6d} / {len(rows):<4}  {predicted:9d}  {100 * correct / len(rows):7.1f}%")
    results["champions"][champ] = {
        "floor": floor[champ], "train_rate": train["rate"], "held_out_mean_rate": mean,
        "tracks_lapped": lapped, "predicted_lapped": predicted, "correct": correct,
        "passed_tighter_than_floor": tighter_pass, "failed_wider_than_floor": wider_fail,
        "tracks": rows,
    }

total = sum(c["correct"] for c in results["champions"].values())
pairs = len(held_out) * len(TRAINING)
print(f"\nprediction correct on {total}/{pairs} champion-track pairs ({100 * total / pairs:.1f}%)")
for champ, c in results["champions"].items():
    if c["passed_tighter_than_floor"] or c["failed_wider_than_floor"]:
        print(f"  {champ}: {c['passed_tighter_than_floor']} lapped tighter than its floor, "
              f"{c['failed_wider_than_floor']} failed wider than it")

if args.json:
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out}")
