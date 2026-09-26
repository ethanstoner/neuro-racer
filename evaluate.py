"""Replay a saved champion, optionally on a track it never trained on.

  python evaluate.py --run runs/oval-seed1 --track oval
  python evaluate.py --run runs/oval-seed1 --track snake --shot
  python evaluate.py --run runs/oval-seed1 --gen 0 --shot     # first generation

The --shot form draws the champion's actual trajectory over the track and
writes a PNG, which is how you check that a suspiciously good lap time is a
real lap and not a scoring bug.
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--track", default=None, help="defaults to the training track")
p.add_argument("--gen", type=int, default=-1, help="which champion (-1 = final)")
p.add_argument("--shot", action="store_true", help="save a trajectory PNG")
args = p.parse_args()

if args.shot:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
from config import Config
from src.tracks import load, slug
from src.artifacts import RunRecorder
from src.simulation import run_generation

champs = RunRecorder.load_all(args.run)
entry = champs[args.gen]
track_name = args.track or entry.get("track") or "oval"

cfg = Config(population=1)
track = load(track_name, cfg)
genome = np.array(entry["genome"], dtype=np.float32)[None, :]

r = run_generation(genome, track, cfg, record=True)
traj = r.frames.pos[:, 0]
speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / cfg.dt

trained_on = slug(entry.get("track", "?"))
run_config = os.path.join(args.run, "config.json")
if os.path.exists(run_config):
    import json
    direction = json.load(open(run_config, encoding="utf-8")).get("direction", "forward")
    if direction != "forward":
        # a both-ways champion must not overwrite the forward one's images
        trained_on += f"-{direction}"
print(f"champion gen {entry['generation']} (trained on {trained_on})  ->  {track_name}")
print(f"  survived   : {r.ticks_run * cfg.dt:.2f}s of {cfg.max_episode_seconds:.0f}s max")
print(f"  finished   : {'yes' if r.laps[0] > 0 else 'NO'}")
if r.laps[0] > 0:
    print(f"  lap time   : {r.lap_times[0]:.2f}s")
print(f"  path length: {np.abs(speeds).sum() * cfg.dt:.0f}px  (lap is {track.length:.0f}px)")
print(f"  speed      : mean {speeds.mean():.0f}  min {speeds.min():.0f}  "
      f"max {speeds.max():.0f} px/s")
print(f"  score      : {r.scores[0]:.1f}")

if args.shot:
    import pygame
    from src.render.track_panel import build_track_surface, draw_cars
    from src.render import palette as P

    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    screen.blit(build_track_surface(track), (0, 0))

    # Trajectory coloured by speed: blue slow, yellow fast. Makes it obvious
    # at a glance whether the car is braking for corners.
    lo, hi = speeds.min(), max(speeds.max(), 1.0)
    for i in range(len(traj) - 1):
        f = (speeds[i] - lo) / max(hi - lo, 1e-6)
        colour = (int(80 + 175 * f), int(150 + 46 * f), int(255 - 191 * f))
        pygame.draw.line(screen, colour,
                         (int(traj[i][0]), int(traj[i][1])),
                         (int(traj[i + 1][0]), int(traj[i + 1][1])), 3)

    draw_cars(screen, r.frames.pos[-1], r.frames.angle[-1], np.array([True]), leader=0)
    font = pygame.font.SysFont("consolas", 18)
    label = (f"gen {entry['generation']} on {track_name}   "
             f"{'lap ' + format(float(r.lap_times[0]), '.2f') + 's' if r.laps[0] > 0 else 'DID NOT FINISH'}"
             f"   speed {speeds.min():.0f}-{speeds.max():.0f} px/s")
    screen.blit(font.render(label, True, P.TEXT), (16, 16))
    screen.blit(font.render("blue = slow, yellow = fast", True, P.TEXT_DIM), (16, 40))

    # Name by training track as well as generation: two runs can easily produce
    # a champion at the same generation number, and naming by generation alone
    # silently overwrites one with the other.
    out = (f"docs/devlog/img/eval-{trained_on}{entry['generation']}"
           f"-on-{slug(track_name)}.png")
    pygame.image.save(screen, out)
    print(f"  wrote {out}")
    pygame.quit()
