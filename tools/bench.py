"""How fast is one generation? The whole fast-forward design depends on this.

If a generation takes more than a second or two, something is looping over
cars in Python instead of operating on the whole population as arrays.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from config import Config  # noqa: E402
from src.tracks import load  # noqa: E402
from src.net import random_population  # noqa: E402
from src.simulation import run_generation  # noqa: E402

cfg = Config()
track = load("oval", cfg)

t0 = time.perf_counter()
track2 = load("chicane", cfg)
print(f"track rasterise      {time.perf_counter() - t0:6.2f}s")

for pop_size in (50, 100, 200):
    c = Config(population=pop_size)
    g = random_population(pop_size, c, np.random.default_rng(0))
    t0 = time.perf_counter()
    r = run_generation(g, track, c)
    dt = time.perf_counter() - t0
    print(f"pop {pop_size:4d} plain      {dt:6.2f}s   "
          f"({r.ticks_run} ticks, {r.ticks_run / dt:6.0f} ticks/s)")

g = random_population(100, cfg, np.random.default_rng(0))
t0 = time.perf_counter()
r = run_generation(g, track, cfg, record=True)
print(f"pop  100 recording   {time.perf_counter() - t0:6.2f}s")

t0 = time.perf_counter()
for _ in range(10):
    run_generation(g, track, cfg)
dt = time.perf_counter() - t0
print(f"\n10 generations headless: {dt:.1f}s  ->  {10 / dt:.1f} gen/s")
