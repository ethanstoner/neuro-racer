"""NeuroRacer -- watch cars teach themselves to drive.

  python main.py --track oval
  python main.py --track snake --seed 3
  python main.py --track oval --shot 12    render generation 12 to PNG and exit

Each generation is simulated headless to completion first, then the recorded
frames are played back at the selected speed. Keeping it in that order means
the rendering path cannot influence the outcome of a run -- what you watch is
exactly what the trainer scored.
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="oval")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--population", type=int, default=100)
parser.add_argument("--generations", type=int, default=100000)
parser.add_argument("--shot", type=int, default=None,
                    help="run headless to generation N, save a PNG, exit")
args = parser.parse_args()

if args.shot is not None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
from config import Config
from src.tracks import load
from src.net import random_population
from src.simulation import run_generation
from src.evolve import next_generation, mutation_sigma
from src.artifacts import RunRecorder
from src.render.track_panel import build_track_surface, draw_cars, draw_rays, draw_ghost
from src.render.net_panel import NetPanel
from src.render.chart_panel import ChartPanel
from src.render.hud import Hud
from src.render import palette as P

# The world is 1200x800 and the track panel shows it 1:1 -- scaling the track
# view would mean scaling every car position and ray too, for no benefit. The
# side panels are added alongside instead, giving a 1600x860 canvas.
CANVAS_W, CANVAS_H = 1600, 860
NET_RECT = (1200, 0, 400, 452)
CHART_RECT = (1200, 452, 400, 348)
HUD_RECT = (0, 800, 1600, 60)

SPEEDS = {pygame.K_1: (1, "1x"), pygame.K_2: (4, "4x"), pygame.K_3: (16, "16x")}
HEADLESS_BURST = 10

cfg = Config(seed=args.seed, population=args.population)
rng = np.random.default_rng(cfg.seed)
track = load(args.track, cfg)
pop = random_population(cfg.population, cfg, rng)
recorder = RunRecorder(f"runs/{args.track}-seed{args.seed}-live", cfg, args.track)

pygame.init()

# Everything is drawn onto a fixed 1600x860 canvas. If the desktop cannot fit
# that, the canvas is scaled down once at flip time -- so the layout never has
# to care what screen it is on, and world coordinates stay 1:1 with pixels.
_desk = pygame.display.Info()
_fit = min(1.0, (_desk.current_w - 80) / CANVAS_W, (_desk.current_h - 120) / CANVAS_H)
WINDOW = (int(CANVAS_W * _fit), int(CANVAS_H * _fit))

window = pygame.display.set_mode(WINDOW)
pygame.display.set_caption(f"NeuroRacer - {args.track}")
screen = pygame.Surface((CANVAS_W, CANVAS_H))
clock = pygame.time.Clock()


def present():
    """Blit the canvas to the window, scaling only if the screen is small."""
    if WINDOW == (CANVAS_W, CANVAS_H):
        window.blit(screen, (0, 0))
    else:
        pygame.transform.smoothscale(screen, WINDOW, window)
    pygame.display.flip()

track_surface = build_track_surface(track)
net_panel = NetPanel(NET_RECT, cfg)
chart_panel = ChartPanel(CHART_RECT, cfg)
hud = Hud(HUD_RECT)

best_history, mean_history, lap_history, lap_time_history = [], [], [], []
best_lap_overall = None
ghost = None
generation = 0
speed_mult, speed_label = 1, "1x"
paused = False
running = True


def evolve_one(record: bool):
    """Run one generation, record artifacts, and advance the population."""
    global pop, generation, best_lap_overall, ghost
    result = run_generation(pop, track, cfg, record=record)
    leader = int(np.argmax(result.scores))

    recorder.record(generation, pop[leader], result.scores, result.laps,
                    result.lap_times, result.alive)
    best_history.append(float(result.scores.max()))
    mean_history.append(float(result.scores.mean()))
    lap_history.append(int(result.laps.max()))

    lap = float(np.min(result.lap_times))
    lap_time_history.append(lap if np.isfinite(lap) else None)
    if np.isfinite(lap):
        best_lap_overall = lap if best_lap_overall is None else min(best_lap_overall, lap)

    champion = pop[leader].copy()
    pop = next_generation(pop, result.scores, cfg, rng, mutation_sigma(generation, cfg))
    generation += 1
    return result, leader, champion


def compose(frames, t, leader, champion):
    """Draw one playback frame."""
    screen.fill(P.BG)
    screen.blit(track_surface, (0, 0))
    draw_ghost(screen, ghost)

    alive = frames.alive[t]
    draw_cars(screen, frames.pos[t], frames.angle[t], alive, leader=leader,
              radius=int(cfg.car_radius))
    if alive[leader]:
        draw_rays(screen, frames.pos[t][leader], frames.angle[t][leader],
                  frames.rays[t][leader], cfg)

    inputs = np.concatenate([frames.rays[t][leader], [0.0]])
    net_panel.draw(screen, champion, inputs,
                   frames.activations[t][leader], frames.controls[t][leader])
    chart_panel.draw(screen, best_history, mean_history, lap_history, lap_time_history)
    hud.draw(screen, generation - 1, int(alive.sum()), cfg.population,
             best_history[-1] if best_history else 0.0, best_lap_overall,
             speed_label, paused, track.name)


# ---------------------------------------------------------------- screenshot
if args.shot is not None:
    for _ in range(args.shot):
        evolve_one(record=False)
    result, leader, champion = evolve_one(record=True)
    frames = result.frames
    compose(frames, min(frames.pos.shape[0] - 1, frames.pos.shape[0] // 2),
            leader, champion)
    out = f"docs/devlog/img/app-{args.track}-gen{args.shot}.png"
    pygame.image.save(screen, out)
    print(f"wrote {out}  (gen {args.shot}, best {best_history[-1]:,.0f}, "
          f"laps {lap_history[-1]}, ticks {result.ticks_run})")
    recorder.close()
    pygame.quit()
    raise SystemExit

# ---------------------------------------------------------------- live loop
while running and generation < args.generations:
    result, leader, champion = evolve_one(record=True)
    frames = result.frames
    total_ticks = frames.pos.shape[0]
    t = 0

    while running and t < total_ticks:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in SPEEDS:
                    speed_mult, speed_label = SPEEDS[event.key]
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_s:
                    path = f"docs/devlog/img/gen-{generation - 1}.png"
                    pygame.image.save(screen, path)
                    print(f"saved {path}")
                elif event.key == pygame.K_h:
                    # Headless burst: no rendering at all, straight through
                    # HEADLESS_BURST generations. This is how you skip the
                    # unwatchable early generations.
                    for _ in range(HEADLESS_BURST):
                        r2, _, _ = evolve_one(record=False)
                        print(f"gen {generation - 1:4d}  best {r2.scores.max():10,.1f}  "
                              f"alive {int(r2.alive.sum()):3d}  laps {int(r2.laps.max())}")
                    t = total_ticks
                    break

        if not running or t >= total_ticks:
            break

        compose(frames, t, leader, champion)
        present()
        clock.tick(60)
        if not paused:
            t += speed_mult

    ghost = frames.pos[:, leader].copy()

recorder.close()
pygame.quit()
