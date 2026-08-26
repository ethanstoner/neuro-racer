"""NeuroRacer -- watch cars teach themselves to drive.

  python main.py --track snake
  python main.py --track snake --population 300 --speed 4
  python main.py --track oval --shot 12     render generation 12 to PNG and exit

Generation N+1 is simulated on a worker thread while generation N plays back,
so the window never stops responding. Simulation still happens strictly before
its own playback, so what you watch is exactly what the trainer scored.
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="snake")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--population", type=int, default=100)
parser.add_argument("--generations", type=int, default=100000)
parser.add_argument("--speed", type=int, default=4, choices=[1, 4, 16],
                    help="starting playback speed; 1x is real time")
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
from src.pump import GenerationPump
from src.render.track_panel import build_track_surface, draw_cars, draw_rays, draw_ghost
from src.render.net_panel import NetPanel
from src.render.chart_panel import ChartPanel
from src.render.hud import Hud
from src.render import palette as P

CANVAS_W, CANVAS_H = 1600, 860
NET_RECT = (1200, 0, 400, 452)
CHART_RECT = (1200, 452, 400, 348)
HUD_RECT = (0, 800, 1600, 60)

SPEEDS = {pygame.K_1: (1, "1x"), pygame.K_2: (4, "4x"), pygame.K_3: (16, "16x")}
BURST = 10

cfg = Config(seed=args.seed, population=args.population)
rng = np.random.default_rng(cfg.seed)
track = load(args.track, cfg)
pop = random_population(cfg.population, cfg, rng)
recorder = RunRecorder(f"runs/{args.track}-seed{args.seed}-live", cfg, args.track)

pygame.init()
_desk = pygame.display.Info()
_fit = min(1.0, (_desk.current_w - 80) / CANVAS_W, (_desk.current_h - 120) / CANVAS_H)
WINDOW = (int(CANVAS_W * _fit), int(CANVAS_H * _fit))
window = pygame.display.set_mode(WINDOW)
pygame.display.set_caption(f"NeuroRacer - {args.track}")
screen = pygame.Surface((CANVAS_W, CANVAS_H))
clock = pygame.time.Clock()

track_surface = build_track_surface(track)
net_panel = NetPanel(NET_RECT, cfg)
chart_panel = ChartPanel(CHART_RECT, cfg)
hud = Hud(HUD_RECT)
status_font = pygame.font.SysFont("consolas", 16)

best_history, mean_history, lap_history, lap_time_history = [], [], [], []
best_lap_overall = None
ghost = None
generation = 0
speed_mult, speed_label = args.speed, f"{args.speed}x"
paused = False
running = True
burst_pending = 0
status = ""


def present():
    if WINDOW == (CANVAS_W, CANVAS_H):
        window.blit(screen, (0, 0))
    else:
        pygame.transform.smoothscale(screen, WINDOW, window)
    pygame.display.flip()


def handle_events():
    """Shared by playback and by the wait-for-simulation idle loop."""
    global running, paused, speed_mult, speed_label, burst_pending
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
                path = f"docs/devlog/img/gen-{max(generation - 1, 0)}.png"
                pygame.image.save(screen, path)
                print(f"saved {path}")
            elif event.key == pygame.K_h:
                # Handled by the main loop, not here: running generations from
                # inside event handling would re-enter the pump.
                burst_pending = BURST


def idle():
    """Called while waiting on the worker. This is what keeps the window alive."""
    handle_events()
    if status:
        bar = pygame.Rect(0, CANVAS_H - 88, CANVAS_W, 24)
        pygame.draw.rect(screen, P.PANEL, bar)
        screen.blit(status_font.render(status, True, P.CAR_LEAD), (14, CANVAS_H - 85))
    present()
    clock.tick(60)


def ingest(result):
    """Record a finished generation and advance the population."""
    global pop, generation, best_lap_overall
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
    return leader, champion


def compose(frames, t, leader, champion):
    screen.fill(P.BG)
    screen.blit(track_surface, (0, 0))
    draw_ghost(screen, ghost)

    alive = frames.alive[t]
    draw_cars(screen, frames.pos[t], frames.angle[t], alive, leader=leader,
              radius=int(cfg.car_radius))
    if alive[leader]:
        draw_rays(screen, frames.pos[t][leader], frames.angle[t][leader],
                  frames.rays[t][leader], cfg)

    inputs = np.concatenate([frames.rays[t][leader], [frames.speed[t][leader]]])
    net_panel.draw(screen, champion, inputs,
                   frames.activations[t][leader], frames.controls[t][leader])
    chart_panel.draw(screen, best_history, mean_history, lap_history, lap_time_history)
    hud.draw(screen, generation - 1, int(alive.sum()), cfg.population,
             best_history[-1] if best_history else 0.0, best_lap_overall,
             speed_label, paused, track.name)


# ---------------------------------------------------------------- screenshot
if args.shot is not None:
    for _ in range(args.shot):
        ingest(run_generation(pop, track, cfg))
    result = run_generation(pop, track, cfg, record=True)
    leader, champion = ingest(result)
    frames = result.frames
    compose(frames, frames.pos.shape[0] // 2, leader, champion)
    out = f"docs/devlog/img/app-{args.track}-gen{args.shot}.png"
    pygame.image.save(screen, out)
    print(f"wrote {out}  (gen {args.shot}, best {best_history[-1]:,.0f}, "
          f"laps {lap_history[-1]}, ticks {result.ticks_run})")
    recorder.close()
    pygame.quit()
    raise SystemExit

# ---------------------------------------------------------------- live loop
pump = GenerationPump(track, cfg)
pump.submit(pop)
status = f"simulating generation {generation}..."

try:
    while running and generation < args.generations:
        result = pump.wait(on_wait=idle)
        if not running:
            break
        leader, champion = ingest(result)

        if burst_pending:
            # Skip ahead without rendering. Each generation is still awaited
            # through idle(), so the window keeps redrawing and responding --
            # the old inline burst froze it for about 17 seconds.
            n = burst_pending
            for i in range(n):
                status = (f"skipping generations  {i + 1}/{n}   "
                          f"now at generation {generation}")
                pump.submit(pop, record=(i == n - 1))
                result = pump.wait(on_wait=idle)
                if not running:
                    break
                leader, champion = ingest(result)
            burst_pending = 0
            if not running:
                break

        # Exactly one generation is always in flight while the previous one
        # plays back. The burst loop above consumes what it submits, so this
        # has to run on both paths or the next wait() finds nothing pending.
        pump.submit(pop)
        status = ""
        frames = result.frames
        if frames is None:
            continue

        total_ticks = frames.pos.shape[0]
        t = 0
        while running and t < total_ticks and not burst_pending:
            handle_events()
            if not running:
                break
            compose(frames, min(t, total_ticks - 1), leader, champion)
            present()
            clock.tick(60)
            if not paused:
                t += speed_mult

        ghost = frames.pos[:, leader].copy()
        status = f"simulating generation {generation}..."
finally:
    pump.close()
    recorder.close()
    pygame.quit()
