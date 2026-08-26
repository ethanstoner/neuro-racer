"""Runs one generation, headless. Must never import pygame -- a test enforces it.

Every car in the population is advanced in lockstep as arrays. Dead cars stay
in the arrays (masked out) rather than being removed, because compacting the
arrays every tick would cost more than simply computing the dead ones.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from config import Config
from src.physics import CarState, step
from src.sensors import cast
from src.net import forward_with_hidden
from src.fitness import FitnessTracker


@dataclass
class Frames:
    pos: np.ndarray          # (T, N, 2)
    angle: np.ndarray        # (T, N)
    alive: np.ndarray        # (T, N)
    rays: np.ndarray         # (T, N, R)
    activations: np.ndarray  # (T, N, H)
    controls: np.ndarray     # (T, N, 3)


@dataclass
class GenerationResult:
    scores: np.ndarray
    alive: np.ndarray
    laps: np.ndarray
    lap_times: np.ndarray
    ticks_run: int
    frames: Optional[Frames] = None


def run_generation(genomes, track, cfg: Config, record: bool = False) -> GenerationResult:
    n = len(genomes)
    sx, sy, sh = track.start_pose
    car = CarState.spawn(n, sx, sy, sh)
    alive = np.ones(n, dtype=bool)
    tracker = FitnessTracker(n, cfg)

    h, w = track.drivable.shape
    T = cfg.max_ticks

    if record:
        frames = Frames(
            pos=np.zeros((T, n, 2), dtype=np.float32),
            angle=np.zeros((T, n), dtype=np.float32),
            alive=np.zeros((T, n), dtype=bool),
            rays=np.zeros((T, n, cfg.n_rays), dtype=np.float32),
            activations=np.zeros((T, n, cfg.n_hidden), dtype=np.float32),
            controls=np.zeros((T, n, cfg.n_outputs), dtype=np.float32),
        )
    else:
        frames = None

    ticks = 0
    for t in range(T):
        rays = cast(car.pos, car.angle, track.drivable, cfg)
        speed = np.linalg.norm(car.vel, axis=1)
        inputs = np.concatenate([rays, (speed / cfg.max_speed)[:, None]], axis=1)
        controls, hidden = forward_with_hidden(genomes, inputs, cfg)

        # Dead cars must be frozen where they died. Zeroing the controls is not
        # enough -- a dead car still carries its velocity and would coast on
        # across the scenery. Snapshot and restore its state instead.
        prev_pos, prev_vel, prev_angle = car.pos.copy(), car.vel.copy(), car.angle.copy()
        car = step(car, controls * alive[:, None], cfg)
        frozen = ~alive
        if frozen.any():
            car.pos[frozen] = prev_pos[frozen]
            car.vel[frozen] = prev_vel[frozen]
            car.angle[frozen] = prev_angle[frozen]

        ix = np.clip(car.pos[:, 0].astype(np.int32), 0, w - 1)
        iy = np.clip(car.pos[:, 1].astype(np.int32), 0, h - 1)
        # body_ok, not drivable: the car has a size, and touching a wall with
        # any part of it ends the run. Sensors still see the real wall.
        alive &= track.body_ok[iy, ix]

        alive = tracker.update(track.progress[iy, ix], speed, alive, cfg.dt)

        if record:
            frames.pos[t] = car.pos
            frames.angle[t] = car.angle
            frames.alive[t] = alive
            frames.rays[t] = rays
            frames.activations[t] = hidden
            frames.controls[t] = controls

        ticks = t + 1
        if not alive.any():
            break

    if record and ticks < T:
        frames = Frames(frames.pos[:ticks], frames.angle[:ticks], frames.alive[:ticks],
                        frames.rays[:ticks], frames.activations[:ticks],
                        frames.controls[:ticks])

    return GenerationResult(tracker.scores(), alive, tracker.laps,
                            tracker.lap_time, ticks, frames)
