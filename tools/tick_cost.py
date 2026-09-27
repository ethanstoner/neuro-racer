"""What one simulation tick costs as the population grows.

Backs the tick-cost table in the README. Every car runs the committed snake
champion, so none crash early and each population size runs the same number of
ticks. Best of several repeats, so a busy machine inflates the numbers less.

  python tools/tick_cost.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from src.artifacts import RunRecorder  # noqa: E402
from src.config import Config  # noqa: E402
from src.simulation import run_generation  # noqa: E402
from src.tracks import load  # noqa: E402

REPEATS = 5
cfg = Config()
track = load("snake", cfg)
genome = RunRecorder.load_champion(os.path.join(ROOT, "data", "champions", "snake"))

rows = []
for n in (1, 10, 100, 500):
    genomes = np.tile(genome, (n, 1)).astype(np.float32)
    best = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        r = run_generation(genomes, track, cfg)
        ms = (time.perf_counter() - t) * 1000 / r.ticks_run
        best = ms if best is None else min(best, ms)
    rows.append((n, best))

base = rows[0][1]
print(f"{'population':>10}  {'ms / tick':>9}  {'us per car-tick':>15}  {'vs. 1 car':>9}")
for n, ms in rows:
    print(f"{n:10d}  {ms:9.3f}  {1000 * ms / n:15.1f}  {ms / base:8.1f}x")
