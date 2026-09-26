"""The real corner floor: a fine sweep down to what the car can physically turn.

tools/corner_sweep.py stopped at 28px in 6-amplitude steps and found floors of
51px (snake) and 131px (chicane). Devlog 13 then had champions lapping 40.6px
corners, the tightest the generated set contains. This sweeps the same wave
shape, scaled to 0.9 so that it stays a legal track (no overlap, inside the
arena) all the way down to an 11.6px corner. That's tighter than the circle the
car can hold at 40px/s (tests/test_premise.py measures 14.3px), so nothing
tighter could be driven by any policy. At full scale the family leaves the arena
below 26px.

Each champion is swept only in the direction(s) it trained in, so the floor
measures corners and not direction (devlog 09-13). A both-ways champion's floor
is the worse of its two directions.

The prediction is in docs/devlog/14-prediction-floors.md.

  python floors.py --json docs/floors.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from direction import CONDITIONS

REVERSE = {"reverse": [True], "both": [False, True]}   # everything else trained clockwise
SCALE = 0.9
_T = np.linspace(0, 2 * np.pi, 600, endpoint=False)


def wave(amplitude: float) -> np.ndarray:
    """tools/corner_sweep.py's family at 0.9 scale: four lobes, tightness set by amplitude alone."""
    r = SCALE * (260 + amplitude * np.cos(4 * _T))
    return np.stack([600 + r * np.cos(_T) * 1.5, 400 + r * np.sin(_T) * 1.0], axis=1)


def rungs(cfg):
    from src.track import Track, min_centerline_radius
    from src.track_io import check
    from src.tracks import reverse_centerline
    out = []
    for amp in range(0, 128, 4):
        c = wave(float(amp))
        ck = check(c, cfg)
        # a rung must be a legal track, or its failures say nothing about the champion
        if ck.self_approach <= cfg.track_width or ck.arena_margin <= 0:
            continue
        track = Track.from_centerline(c, cfg, name=f"wave{amp}")
        rev = Track.from_centerline(reverse_centerline(c), cfg, name=f"wave{amp}-reversed")
        out.append((min_centerline_radius(track.centerline), track, rev))
    return sorted(out, key=lambda r: -r[0])


def sweep(job):
    condition, seed, run_dir, generation, points, offsets = job
    from config import Config
    from src.artifacts import RunRecorder
    from src.robustness import assess

    cfg = Config()
    genome = RunRecorder.load_champion(run_dir, generation)
    result = {"condition": condition, "seed": seed, "directions": {}}
    for reverse in REVERSE.get(condition, [False]):
        rows, floor = [], None
        for radius, fwd, rev in rungs(cfg):
            rate = assess(genome, rev if reverse else fwd, cfg, points, offsets)["rate"]
            rows.append({"radius": radius, "rate": rate})
        # floor = tightest rung reached without a failure on the way down
        for r in rows:
            if r["rate"] <= 0.5:
                break
            floor = r["radius"]
        passed = [r["radius"] for r in rows if r["rate"] > 0.5]
        result["directions"]["reverse" if reverse else "forward"] = {
            "floor": floor, "tightest_passed": min(passed) if passed else None, "rungs": rows}
    floors = [d["floor"] for d in result["directions"].values()]
    result["floor"] = None if None in floors else max(floors)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--runs", default="runs")
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--offsets", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 1))
    p.add_argument("--json", default=None)
    a = p.parse_args()

    from config import Config
    r = rungs(Config())
    print(f"{len(r)} rungs, {r[0][0]:.0f}px down to {r[-1][0]:.0f}px")

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
        results = list(pool.map(sweep, jobs))

    print(f"\n{'condition':17} {'seed':>4}  {'floor':>7}  per direction (floor / tightest passed)")
    for res in results:
        f = f"{res['floor']:.0f}px" if res["floor"] else "none"
        per = "  ".join(f"{k}: {v['floor'] or 0:.0f}/{v['tightest_passed'] or 0:.0f}"
                        for k, v in res["directions"].items())
        print(f"{res['condition']:17} {res['seed']:4d}  {f:>7}  {per}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
