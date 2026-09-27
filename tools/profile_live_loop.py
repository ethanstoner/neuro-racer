"""Evidence for the "it freezes sometimes" report.

scripts/app.py alternates between two things:
  1. simulate the next generation headless, recording every tick
  2. play those recorded frames back at 60fps

Step 1 happens with no pygame event pumping at all, so however long it takes is
a window that does not redraw and does not respond. This measures step 1 across
a real training arc, and how much playback there is to hide it behind.

  python tools/profile_live_loop.py --track snake --generations 60
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from src.config import Config  # noqa: E402
from src.tracks import load  # noqa: E402
from src.net import random_population  # noqa: E402
from src.simulation import run_generation  # noqa: E402
from src.evolve import next_generation, mutation_sigma  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--track", default="snake")
p.add_argument("--generations", type=int, default=60)
p.add_argument("--population", type=int, default=100)
args = p.parse_args()

cfg = Config(population=args.population)
rng = np.random.default_rng(cfg.seed)
track = load(args.track, cfg)
pop = random_population(cfg.population, cfg, rng)

print(f"track={args.track} pop={cfg.population}")
print(f"{'gen':>4} {'sim_s':>7} {'ticks':>6} {'alive':>6} "
      f"{'playback_s@4x':>14} {'freeze_ratio':>13} {'frames_MB':>10}")

worst = 0.0
rows = []
for gen in range(args.generations):
    t0 = time.perf_counter()
    r = run_generation(pop, track, cfg, record=True)
    sim = time.perf_counter() - t0

    f = r.frames
    mb = sum(a.nbytes for a in (f.pos, f.angle, f.alive, f.rays, f.speed,
                                f.activations, f.controls)) / 1e6
    playback = r.ticks_run / 60.0 / 4.0        # 4x speed, 60fps
    ratio = sim / max(playback, 1e-9)
    worst = max(worst, sim)
    rows.append((gen, sim, r.ticks_run, int(r.alive.sum()), playback, ratio, mb))

    if gen % 5 == 0 or sim > 0.5:
        print(f"{gen:4d} {sim:7.3f} {r.ticks_run:6d} {int(r.alive.sum()):6d} "
              f"{playback:14.2f} {ratio:12.1%} {mb:10.1f}")

    pop = next_generation(pop, r.scores, cfg, rng, mutation_sigma(gen, cfg))

sims = [r[1] for r in rows]
print(f"\nworst single freeze : {worst:.3f}s")
print(f"mean sim per gen    : {np.mean(sims):.3f}s")
print(f"gens over 0.25s     : {sum(1 for s in sims if s > 0.25)} of {len(sims)}")
print(f"gens over 1.0s      : {sum(1 for s in sims if s > 1.0)} of {len(sims)}")
print(f"peak frames buffer  : {max(r[6] for r in rows):.1f} MB per generation")
print(f"\nH-key burst (10 gens, no event pump): "
      f"{np.mean(sims[-10:]) * 10:.1f}s of frozen window")
