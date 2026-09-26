"""Headless training.

  python train.py --track oval --generations 150
  python train.py --track snake --generations 200 --seed 3
  python train.py --track snake --direction both     # every car drives both ways
  python train.py --track snake chicane              # every car drives both tracks

--direction both scores each car on the track in both directions and ranks it by
the mean. Several --track names do the same across tracks. Either way, a car
only counts as lapping if it laps every one, and its lap time is its slowest.
"""
import argparse
import time
import numpy as np
from config import Config
from src.tracks import load, slug
from src.net import random_population
from src.simulation import run_generation
from src.evolve import next_generation, mutation_sigma
from src.artifacts import RunRecorder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", nargs="+", default=["oval"])
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--direction", choices=("forward", "reverse", "both"), default="forward")
    a = p.parse_args()

    cfg = Config(seed=a.seed, population=a.population)
    rng = np.random.default_rng(cfg.seed)
    directions = {"forward": [False], "reverse": [True], "both": [False, True]}[a.direction]
    tracks = [load(name, cfg, reverse=r) for name in a.track for r in directions]
    track = tracks[0]
    name = "+".join(slug(t) for t in a.track)
    pop = random_population(cfg.population, cfg, rng)
    label = name if a.direction == "forward" else f"{name}-{a.direction}"
    out = a.out or f"runs/{label}-seed{a.seed}"
    rec = RunRecorder(out, cfg, track_name=a.track[0],
                      meta={"direction": a.direction, "tracks": a.track})

    print(f"track={name}  pop={cfg.population}  seed={cfg.seed}  "
          f"corner={track.length:.0f}px lap")
    print(f"{'gen':>5} {'best':>10} {'mean':>10} {'alive':>6} {'laps':>5} {'lap':>7}")

    start = time.time()
    first_lap_gen = None
    for gen in range(a.generations):
        results = [run_generation(pop, t, cfg) for t in tracks]
        scores = np.mean([r.scores for r in results], axis=0)
        laps = np.min([r.laps for r in results], axis=0)
        lap_times = np.max([r.lap_times for r in results], axis=0)
        alive = np.logical_and.reduce([r.alive for r in results])
        best_i = int(np.argmax(scores))
        lap = float(np.min(lap_times))
        rec.record(gen, pop[best_i], scores, laps, lap_times, alive)

        if first_lap_gen is None and laps.max() > 0:
            first_lap_gen = gen

        if not a.quiet:
            lap_s = f"{lap:.2f}s" if np.isfinite(lap) else "--"
            print(f"{gen:5d} {scores.max():10.1f} {scores.mean():10.1f} "
                  f"{int(alive.sum()):6d} {int(laps.max()):5d} {lap_s:>7}")

        pop = next_generation(pop, scores, cfg, rng, mutation_sigma(gen, cfg))

    rec.close()
    elapsed = time.time() - start
    print(f"\ndone in {elapsed:.1f}s ({a.generations / elapsed:.1f} gen/s) -> {out}")
    if first_lap_gen is not None:
        print(f"first completed lap: generation {first_lap_gen}")
    else:
        print("no car ever completed a lap")


if __name__ == "__main__":
    main()
