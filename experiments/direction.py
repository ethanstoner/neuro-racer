"""Is the snake champion's one-way bias in the data or the network?

Evaluates the champions of `train.py --direction forward|reverse|both` runs on
snake both ways round, on all seven built-ins both ways, and on the generated
held-out set (procgen seed 7) split by direction. The prediction it tests is in
docs/devlog/10-prediction-direction.md, committed before the runs.

  python -m experiments.direction                         # runs/snake{,-reverse,-both}-seed1..5
  python -m experiments.direction --seeds 1 2 3 --json docs/results/direction.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

# condition -> (run directory prefix, which generation's champion; -1 is the last)
CONDITIONS = {"forward": ("snake", -1), "reverse": ("snake-reverse", -1),
              "both": ("snake-both", -1),
              # budget-matched controls for "both" (devlog 12)
              "forward-400": ("snake-400gen", -1), "snake+chicane": ("snake+chicane", -1),
              # random start pose every generation (devlog 14), one run read twice
              "random-start": ("snake-random", 199), "random-start-400": ("snake-random", -1)}
TIGHT = 51.0   # the one-way snake floor from the corner sweep


def clockwise(points) -> bool:
    """Positive shoelace area in y-down screen coordinates is clockwise on screen."""
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def evaluate(job):
    condition, seed, run_dir, generation, points, offsets = job
    from src.config import Config
    from src.artifacts import RunRecorder
    from src.procgen import generate
    from src.robustness import assess
    from src.track import Track
    from src.tracks import BUILDERS, load

    cfg = Config()
    genome = RunRecorder.load_champion(run_dir, generation)
    rate = lambda t: assess(genome, t, cfg, points, offsets)["rate"]
    out = {"condition": condition, "seed": seed,
           "snake_forward": rate(load("snake", cfg)),
           "snake_reverse": rate(load("snake", cfg, reverse=True)),
           "builtins": {}}
    for name in sorted(BUILDERS):
        out["builtins"][name] = {"forward": rate(load(name, cfg)),
                                 "reverse": rate(load(name, cfg, reverse=True))}
    cw = ccw = n_cw = n_ccw = tight = n_tight = tight_cw = n_tight_cw = 0
    for g in generate(100, 7, cfg):
        passed = rate(Track.from_centerline(g.centerline, cfg, name=g.name)) > 0.5
        if g.check.tightest_radius < TIGHT:
            n_tight += 1
            tight += passed
            if clockwise(g.centerline):
                n_tight_cw += 1
                tight_cw += passed
        if clockwise(g.centerline):
            n_cw += 1
            cw += passed
        else:
            n_ccw += 1
            ccw += passed
    out["generated"] = {"clockwise": cw, "counter_clockwise": ccw, "n_clockwise": n_cw,
                        "n_counter_clockwise": n_ccw, "tight": tight, "n_tight": n_tight,
                        "tight_clockwise": tight_cw, "n_tight_clockwise": n_tight_cw}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--merge", action="store_true",
                   help="replace these conditions' rows in --json and keep the rest")
    p.add_argument("--runs", default="runs")
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--offsets", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(15, os.cpu_count() or 1))
    p.add_argument("--json", default=None)
    a = p.parse_args()

    jobs = []
    for condition in a.conditions:
        prefix, generation = CONDITIONS[condition]
        for seed in a.seeds:
            d = Path(a.runs) / f"{prefix}-seed{seed}"
            if (d / "champions.json").exists():
                jobs.append((condition, seed, str(d), generation, a.points, a.offsets))
            else:
                print(f"missing {d}, skipped")

    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(evaluate, jobs))

    print(f"{'condition':13} {'seed':>4}  {'snake fwd':>9} {'snake rev':>9}  "
          f"{'built-ins fwd':>13} {'built-ins rev':>13}  {'gen cw':>7} {'gen ccw':>7} {'<51px':>6} {'<51 cw':>6}")
    for r in rows:
        bf = np.mean([v["forward"] for v in r["builtins"].values()])
        br = np.mean([v["reverse"] for v in r["builtins"].values()])
        g = r["generated"]
        print(f"{r['condition']:13} {r['seed']:4d}  {100 * r['snake_forward']:8.0f}% "
              f"{100 * r['snake_reverse']:8.0f}%  {100 * bf:12.0f}% {100 * br:12.0f}%  "
              f"{g['clockwise']:3d}/{g['n_clockwise']:<3d} {g['counter_clockwise']:3d}/{g['n_counter_clockwise']:<3d} {g['tight']:2d}/{g['n_tight']} {g['tight_clockwise']:3d}/{g['n_tight_clockwise']}")

    if a.json:
        out = Path(a.json)
        if a.merge and out.exists():
            kept = [r for r in json.loads(out.read_text(encoding="utf-8"))
                    if r["condition"] not in a.conditions]
            rows = kept + rows
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
