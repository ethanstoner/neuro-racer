"""Headless training.

  python train.py --track oval --generations 150
  python train.py --track snake --generations 200 --seed 3
"""
import argparse
import time
import numpy as np
from config import Config
from src.tracks import load
from src.net import random_population
from src.simulation import run_generation
from src.evolve import next_generation, mutation_sigma
from src.artifacts import RunRecorder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", default="oval")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    cfg = Config(seed=a.seed, population=a.population)
    rng = np.random.default_rng(cfg.seed)
    track = load(a.track, cfg)
    pop = random_population(cfg.population, cfg, rng)
    out = a.out or f"runs/{a.track}-seed{a.seed}"
    rec = RunRecorder(out, cfg, track_name=a.track)

    print(f"track={a.track}  pop={cfg.population}  seed={cfg.seed}  "
          f"corner={track.length:.0f}px lap")
    print(f"{'gen':>5} {'best':>10} {'mean':>10} {'alive':>6} {'laps':>5} {'lap':>7}")

    start = time.time()
    first_lap_gen = None
    for gen in range(a.generations):
        r = run_generation(pop, track, cfg)
        best_i = int(np.argmax(r.scores))
        lap = float(np.min(r.lap_times))
        rec.record(gen, pop[best_i], r.scores, r.laps, r.lap_times, r.alive)

        if first_lap_gen is None and r.laps.max() > 0:
            first_lap_gen = gen

        if not a.quiet:
            lap_s = f"{lap:.2f}s" if np.isfinite(lap) else "--"
            print(f"{gen:5d} {r.scores.max():10.1f} {r.scores.mean():10.1f} "
                  f"{int(r.alive.sum()):6d} {int(r.laps.max()):5d} {lap_s:>7}")

        pop = next_generation(pop, r.scores, cfg, rng, mutation_sigma(gen, cfg))

    rec.close()
    elapsed = time.time() - start
    print(f"\ndone in {elapsed:.1f}s ({a.generations / elapsed:.1f} gen/s) -> {out}")
    if first_lap_gen is not None:
        print(f"first completed lap: generation {first_lap_gen}")
    else:
        print("no car ever completed a lap")


if __name__ == "__main__":
    main()
