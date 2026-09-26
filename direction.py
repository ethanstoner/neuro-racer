"""Is the snake champion's one-way bias in the data or the network?

Evaluates the champions of `train.py --direction forward|reverse|both` runs on
snake both ways round, on all seven built-ins both ways, and on the generated
held-out set (procgen seed 7) split by direction. The prediction it tests is in
docs/devlog/10-prediction-direction.md, committed before the runs.

  python direction.py                         # runs/snake{,-reverse,-both}-seed1..5
  python direction.py --seeds 1 2 3 --json docs/direction.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CONDITIONS = {"forward": "snake", "reverse": "snake-reverse", "both": "snake-both"}


def clockwise(points) -> bool:
    """Positive shoelace area in y-down screen coordinates is clockwise on screen."""
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def evaluate(job):
    condition, seed, run_dir, points, offsets = job
    from config import Config
    from src.artifacts import RunRecorder
    from src.procgen import generate
    from src.robustness import assess
    from src.track import Track
    from src.tracks import BUILDERS, load

    cfg = Config()
    genome = RunRecorder.load_champion(run_dir)
    rate = lambda t: assess(genome, t, cfg, points, offsets)["rate"]
    out = {"condition": condition, "seed": seed,
           "snake_forward": rate(load("snake", cfg)),
           "snake_reverse": rate(load("snake", cfg, reverse=True)),
           "builtins": {}}
    for name in sorted(BUILDERS):
        out["builtins"][name] = {"forward": rate(load(name, cfg)),
                                 "reverse": rate(load(name, cfg, reverse=True))}
    cw = ccw = n_cw = n_ccw = 0
    for g in generate(100, 7, cfg):
        passed = rate(Track.from_centerline(g.centerline, cfg, name=g.name)) > 0.5
        if clockwise(g.centerline):
            n_cw += 1
            cw += passed
        else:
            n_ccw += 1
            ccw += passed
    out["generated"] = {"clockwise": cw, "counter_clockwise": ccw, "n_clockwise": n_cw,
                        "n_counter_clockwise": n_ccw}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--runs", default="runs")
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--offsets", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(15, os.cpu_count() or 1))
    p.add_argument("--json", default=None)
    a = p.parse_args()

    jobs = []
    for condition, prefix in CONDITIONS.items():
        for seed in a.seeds:
            d = Path(a.runs) / f"{prefix}-seed{seed}"
            if (d / "champions.json").exists():
                jobs.append((condition, seed, str(d), a.points, a.offsets))
            else:
                print(f"missing {d}, skipped")

    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(evaluate, jobs))

    print(f"{'condition':9} {'seed':>4}  {'snake fwd':>9} {'snake rev':>9}  "
          f"{'built-ins fwd':>13} {'built-ins rev':>13}  {'gen cw':>7} {'gen ccw':>7}")
    for r in rows:
        bf = np.mean([v["forward"] for v in r["builtins"].values()])
        br = np.mean([v["reverse"] for v in r["builtins"].values()])
        g = r["generated"]
        print(f"{r['condition']:9} {r['seed']:4d}  {100 * r['snake_forward']:8.0f}% "
              f"{100 * r['snake_reverse']:8.0f}%  {100 * bf:12.0f}% {100 * br:12.0f}%  "
              f"{g['clockwise']:3d}/{g['n_clockwise']:<3d} {g['counter_clockwise']:3d}/{g['n_counter_clockwise']:<3d}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
