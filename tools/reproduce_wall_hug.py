"""Reproduce the generation-0 wall-hugging exploit from devlog entry 3.

Collision originally tested the car's centre pixel, so a car could sit with
half its body inside a wall at zero cost -- and leaning on the smooth outer
wall of an oval turned out to be the cheapest way round. A random genome pulled
it off in generation 0.

Setting car_radius to 0 recreates that behaviour exactly, because body_ok is
defined as "within (half_width - car_radius) of the centerline", so a radius of
zero collapses it back onto the raw tarmac mask. Same seed, same population,
same everything else.

  python tools/reproduce_wall_hug.py
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from config import Config  # noqa: E402
from src.tracks import load  # noqa: E402
from src.net import random_population  # noqa: E402
from src.simulation import run_generation  # noqa: E402
from src.render.track_panel import build_track_surface, draw_cars  # noqa: E402
from src.render import palette as P  # noqa: E402

BROKEN = Config(seed=1, population=100, car_radius=0.0)   # the old behaviour
FIXED = Config(seed=1, population=100)                    # 6px car

for label, cfg in (("broken", BROKEN), ("fixed", FIXED)):
    track = load("oval", cfg)
    pop = random_population(cfg.population, cfg, np.random.default_rng(cfg.seed))
    r = run_generation(pop, track, cfg, record=True)
    best = int(np.argmax(r.scores))
    lapped = int((r.laps > 0).sum())
    print(f"generation 0, car_radius={cfg.car_radius:.0f}: "
          f"{lapped} of {cfg.population} completed a lap, "
          f"best score {r.scores.max():,.1f}")

    if label == "broken" and lapped:
        traj = r.frames.pos[:, best]
        speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / cfg.dt
        pygame.init()
        screen = pygame.display.set_mode((cfg.width, cfg.height))
        screen.blit(build_track_surface(track), (0, 0))
        lo, hi = speeds.min(), max(speeds.max(), 1.0)
        for i in range(len(traj) - 1):
            f = (speeds[i] - lo) / max(hi - lo, 1e-6)
            colour = (int(80 + 175 * f), int(150 + 46 * f), int(255 - 191 * f))
            pygame.draw.line(screen, colour, (int(traj[i][0]), int(traj[i][1])),
                             (int(traj[i + 1][0]), int(traj[i + 1][1])), 3)
        draw_cars(screen, r.frames.pos[-1][best:best + 1],
                  r.frames.angle[-1][best:best + 1], np.array([True]), leader=0)
        font = pygame.font.SysFont("consolas", 18)
        screen.blit(font.render(
            f"generation 0, collision ignores car size   "
            f"lap {float(r.lap_times[best]):.2f}s", True, P.TEXT), (16, 16))
        screen.blit(font.render(
            "random weights, riding the outer wall the whole way round",
            True, P.TEXT_DIM), (16, 40))
        out = "docs/devlog/img/wall-hug-gen0.png"
        pygame.image.save(screen, out)
        print(f"  wrote {out}")
        pygame.quit()
