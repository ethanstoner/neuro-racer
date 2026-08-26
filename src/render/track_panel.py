"""Draws the track and the cars. Rendering only -- no simulation logic."""
import numpy as np
import pygame
from src.render import palette as P


def build_track_surface(track) -> pygame.Surface:
    """Rasterise the mask to a Surface once; blit it every frame after that.

    Rebuilding this per frame would cost far more than the entire physics step.
    """
    h, w = track.drivable.shape
    rgb = np.zeros((w, h, 3), dtype=np.uint8)
    rgb[...] = np.array(P.BG, dtype=np.uint8)
    rgb[track.drivable.T] = np.array(P.TARMAC, dtype=np.uint8)

    # 1px kerb: drivable pixels that have a non-drivable 4-neighbour.
    d = track.drivable
    edge = d & ~(np.roll(d, 1, 0) & np.roll(d, -1, 0)
                 & np.roll(d, 1, 1) & np.roll(d, -1, 1))
    rgb[edge.T] = np.array(P.KERB, dtype=np.uint8)

    surf = pygame.Surface((w, h))
    pygame.surfarray.blit_array(surf, rgb)

    for x, y in track.centerline[::6]:
        surf.set_at((int(x), int(y)), P.CENTER)

    # Start/finish line, drawn across the track at the start pose.
    sx, sy, sh = track.start_pose
    nx, ny = -np.sin(sh), np.cos(sh)
    half = 46
    pygame.draw.line(surf, P.TEXT_DIM,
                     (int(sx - nx * half), int(sy - ny * half)),
                     (int(sx + nx * half), int(sy + ny * half)), 2)
    return surf


def draw_cars(surf, pos, angle, alive, leader=-1, radius=6):
    for i in range(len(pos)):
        colour = P.CAR_LEAD if i == leader else (P.CAR_ALIVE if alive[i] else P.CAR_DEAD)
        x, y = float(pos[i, 0]), float(pos[i, 1])
        nose = (x + np.cos(angle[i]) * radius * 1.9, y + np.sin(angle[i]) * radius * 1.9)
        pygame.draw.circle(surf, colour, (int(x), int(y)), radius)
        pygame.draw.line(surf, colour, (int(x), int(y)), (int(nose[0]), int(nose[1])), 2)


def draw_rays(surf, pos, angle, rays, cfg):
    """Draw one car's sensor rays. Red where the ray found a wall."""
    x, y = float(pos[0]), float(pos[1])
    for k, ang in enumerate(cfg.ray_angles):
        d = float(rays[k]) * cfg.ray_length
        a = float(angle) + ang
        end = (x + np.cos(a) * d, y + np.sin(a) * d)
        hit = rays[k] < 0.999
        pygame.draw.line(surf, P.RAY_HIT if hit else P.RAY,
                         (int(x), int(y)), (int(end[0]), int(end[1])), 1)
        if hit:
            pygame.draw.circle(surf, P.RAY_HIT, (int(end[0]), int(end[1])), 3)


def draw_ghost(surf, trail):
    """Dim trail of the previous generation's champion."""
    if trail is None or len(trail) < 2:
        return
    pts = [(int(x), int(y)) for x, y in trail[::3]]
    if len(pts) > 1:
        pygame.draw.lines(surf, P.GHOST, False, pts, 1)
