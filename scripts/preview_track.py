"""Track viewer.

  python -m scripts.preview_track oval            open a window
  python -m scripts.preview_track oval --shot     render once to docs/devlog/img and exit

The --shot path uses SDL's dummy video driver, so it works without a display
and is what the build process uses to verify tracks actually look right.
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("track", nargs="?", default="oval")
p.add_argument("--shot", action="store_true")
p.add_argument("--rays", action="store_true", help="draw sensor rays from the start pose")
args = p.parse_args()

if args.shot:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
from src.config import DEFAULT as CFG
from src.tracks import load, slug
from src.sensors import cast
from src.render.track_panel import build_track_surface, draw_cars, draw_rays

track = load(args.track, CFG)

pygame.init()
screen = pygame.display.set_mode((CFG.width, CFG.height))
pygame.display.set_caption(f"NeuroRacer - {args.track}")
surf = build_track_surface(track)

sx, sy, sh = track.start_pose
pos = np.float32([[sx, sy]])
ang = np.float32([sh])
rays = cast(pos, ang, track.drivable, CFG)


def compose():
    screen.blit(surf, (0, 0))
    if args.rays:
        draw_rays(screen, pos[0], ang[0], rays[0], CFG)
    draw_cars(screen, pos, ang, np.array([True]), leader=0)


if args.shot:
    compose()
    out = f"docs/devlog/img/track-{slug(args.track)}{'-rays' if args.rays else ''}.png"
    pygame.image.save(screen, out)
    print(f"wrote {out}")
    print(f"  drivable px : {track.drivable.sum():,}")
    print(f"  lap length  : {track.length:.0f} px")
    print(f"  start pose  : ({sx:.0f}, {sy:.0f}) heading {np.degrees(sh):.0f} deg")
    print(f"  rays        : {np.round(rays[0], 3)}")
else:
    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        compose()
        pygame.display.flip()
        clock.tick(60)
pygame.quit()
