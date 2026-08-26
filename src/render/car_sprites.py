"""Actual car sprites instead of circles.

Drawing a body, a windscreen and four wheels per car per frame would be ~700
pygame calls at 100 cars and over 2000 at 300. So each car colour is drawn once
into a small surface and then pre-rotated into a lookup table of fixed angles;
rendering a car becomes a single blit of a cached surface.

At ANGLE_STEPS=72 the angular error is at most 2.5 degrees, which is invisible
at this car size and far cheaper than rotating per frame.
"""
import numpy as np
import pygame

ANGLE_STEPS = 72
_CACHE = {}


def _shade(colour, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in colour)


def _draw_car(length: int, width: int, colour) -> pygame.Surface:
    """One car pointing right (+x), on a transparent surface."""
    pad = 2
    surf = pygame.Surface((length + pad * 2, width + pad * 2), pygame.SRCALPHA)
    x0, y0 = pad, pad

    tyre = _shade(colour, 0.28)
    wheel_w, wheel_h = max(3, length // 4), max(2, width // 4 + 1)
    for wx in (x0 + length // 6, x0 + length - length // 6 - wheel_w):
        for wy in (y0 - 1, y0 + width - wheel_h + 1):
            pygame.draw.rect(surf, tyre, (wx, wy, wheel_w, wheel_h), border_radius=1)

    body = pygame.Rect(x0, y0 + 1, length, width - 2)
    pygame.draw.rect(surf, colour, body, border_radius=max(2, width // 3))
    pygame.draw.rect(surf, _shade(colour, 0.55), body, width=1,
                     border_radius=max(2, width // 3))

    # Windscreen sits forward of centre so the car reads as pointing somewhere.
    glass = pygame.Rect(x0 + length // 2, y0 + 2, max(2, length // 4), max(2, width - 4))
    pygame.draw.rect(surf, _shade(colour, 0.42), glass, border_radius=1)

    # Nose highlight -- the strongest cue for heading at small sizes.
    pygame.draw.rect(surf, _shade(colour, 1.35),
                     (x0 + length - 2, y0 + 2, 2, max(1, width - 4)))
    return surf


def sprites_for(colour, radius: int):
    """Rotation lookup table for one colour, built once and cached."""
    length = max(10, int(radius * 3.2))
    width = max(6, int(radius * 1.9))
    key = (tuple(colour), length, width)
    if key in _CACHE:
        return _CACHE[key]

    base = _draw_car(length, width, colour)
    # pygame rotates counter-clockwise; screen y grows downward, so the angle is
    # negated to keep sprites pointing the same way the physics heading does.
    table = [pygame.transform.rotate(base, -360.0 * i / ANGLE_STEPS)
             for i in range(ANGLE_STEPS)]
    _CACHE[key] = table
    return table


def blit_cars(surf, pos, angle, colour, radius: int, indices=None):
    """Blit many cars of one colour in a single pass."""
    table = sprites_for(colour, radius)
    idx = range(len(pos)) if indices is None else indices
    step = ANGLE_STEPS / (2.0 * np.pi)
    for i in idx:
        sprite = table[int(angle[i] * step) % ANGLE_STEPS]
        rect = sprite.get_rect(center=(int(pos[i, 0]), int(pos[i, 1])))
        surf.blit(sprite, rect)


def clear_cache():
    _CACHE.clear()
