"""Drive the track yourself. Arrow keys. Sets the human baseline lap time.

  python drive.py oval
  python drive.py snake --auto 6      headless self-check, no display needed

Deliberately kept in the repo after the AI works: the final devlog entry
compares the evolved champion against these human times.
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("track", nargs="?", default="oval")
p.add_argument("--auto", type=float, default=0.0,
               help="run N seconds of scripted input headless and exit (self-check)")
args = p.parse_args()

if args.auto:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
from config import DEFAULT as CFG
from src.tracks import load
from src.physics import CarState, step
from src.sensors import cast
from src.fitness import wrapped_delta
from src.render.track_panel import build_track_surface, draw_cars, draw_rays
from src.render import palette as P

track = load(args.track, CFG)

pygame.init()
screen = pygame.display.set_mode((CFG.width, CFG.height))
pygame.display.set_caption(f"NeuroRacer - drive {args.track}")
font = pygame.font.SysFont("consolas", 20)
clock = pygame.time.Clock()
surf = build_track_surface(track)

sx, sy, sh = track.start_pose
car = CarState.spawn(1, sx, sy, sh)
prev_p = track.progress[int(sy), int(sx)]
cum = 0.0
elapsed = 0.0
best = None
crashes = 0
show_rays = True


def reset():
    global car, prev_p, cum, elapsed
    car = CarState.spawn(1, sx, sy, sh)
    prev_p = track.progress[int(sy), int(sx)]
    cum = 0.0
    elapsed = 0.0


running = True
ticks = 0
auto_ticks = int(args.auto / CFG.dt)

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            reset()
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_TAB:
            show_rays = not show_rays

    if args.auto:
        # Scripted: full throttle with a steady turn, enough to prove the whole
        # loop runs, laps count, and crashes reset.
        controls = np.float32([[1.0, 0.0, 0.35]])
    else:
        k = pygame.key.get_pressed()
        controls = np.float32([[float(k[pygame.K_UP]), float(k[pygame.K_DOWN]),
                                float(k[pygame.K_RIGHT]) - float(k[pygame.K_LEFT])]])

    car = step(car, controls, CFG)
    elapsed += CFG.dt

    x = int(np.clip(car.pos[0, 0], 0, CFG.width - 1))
    y = int(np.clip(car.pos[0, 1], 0, CFG.height - 1))

    if not track.drivable[y, x]:
        crashes += 1
        reset()
    else:
        p = track.progress[y, x]
        cum += float(wrapped_delta(np.float32([p]), np.float32([prev_p]))[0])
        prev_p = p
        if cum >= 1.0 - 1e-4:
            best = elapsed if best is None else min(best, elapsed)
            cum = 0.0
            elapsed = 0.0

    rays = cast(car.pos, car.angle, track.drivable, CFG)
    screen.blit(surf, (0, 0))
    if show_rays:
        draw_rays(screen, car.pos[0], car.angle[0], rays[0], CFG)
    draw_cars(screen, car.pos, car.angle, np.array([True]), leader=0)

    speed = float(np.linalg.norm(car.vel))
    lines = [
        f"lap {cum * 100:5.1f}%   {elapsed:5.2f}s   {speed:4.0f}px/s   crashes {crashes}",
        f"best {best:.2f}s" if best else "best   --",
        "arrows drive   R reset   TAB rays   ESC quit",
    ]
    for i, s in enumerate(lines):
        screen.blit(font.render(s, True, P.TEXT if i < 2 else P.TEXT_DIM), (16, 16 + i * 24))

    if not args.auto:
        pygame.display.flip()
        clock.tick(60)

    ticks += 1
    if args.auto and ticks >= auto_ticks:
        running = False

if args.auto:
    pygame.image.save(screen, f"docs/devlog/img/drive-{args.track}.png")
    print(f"ran {ticks} ticks on {args.track}")
    print(f"  speed now : {float(np.linalg.norm(car.vel)):.0f} px/s")
    print(f"  lap cum   : {cum * 100:.1f}%")
    print(f"  crashes   : {crashes}")
    print(f"  best lap  : {f'{best:.2f}s' if best else 'none'}")
pygame.quit()
