"""Arcade car physics over whole-population arrays. Pure: no I/O, no globals."""
from dataclasses import dataclass
import numpy as np
from config import Config


@dataclass
class CarState:
    pos: np.ndarray      # (N, 2) float32
    vel: np.ndarray      # (N, 2) float32
    angle: np.ndarray    # (N,)   float32, radians

    @classmethod
    def spawn(cls, n: int, x: float, y: float, heading: float) -> "CarState":
        return cls(
            pos=np.tile(np.float32([x, y]), (n, 1)),
            vel=np.zeros((n, 2), dtype=np.float32),
            angle=np.full(n, heading, dtype=np.float32),
        )

    def copy(self) -> "CarState":
        return CarState(self.pos.copy(), self.vel.copy(), self.angle.copy())


def step(s: CarState, controls: np.ndarray, cfg: Config) -> CarState:
    """Advance every car one tick.

    controls[:, 0] throttle 0..1, [:, 1] brake 0..1, [:, 2] steer -1..1.
    Inputs are clipped rather than trusted -- an unevolved network emits
    whatever it likes and must not be able to produce inf or nan downstream.
    """
    throttle = np.clip(controls[:, 0], 0.0, 1.0)
    brake = np.clip(controls[:, 1], 0.0, 1.0)
    steer = np.clip(controls[:, 2], -1.0, 1.0)

    fwd = np.stack([np.cos(s.angle), np.sin(s.angle)], axis=1)
    right = np.stack([-np.sin(s.angle), np.cos(s.angle)], axis=1)

    speed = np.linalg.norm(s.vel, axis=1)

    # Steering authority falls off with speed: the reason braking for corners
    # is a thing the network has to discover rather than something it gets free.
    rate = cfg.max_steer_rate / (1.0 + speed / cfg.steer_speed_falloff)
    angle = s.angle + steer * rate * cfg.dt

    accel = throttle * cfg.accel - brake * cfg.brake
    vel = s.vel + fwd * (accel * cfg.dt)[:, None]
    vel *= (1.0 - cfg.drag * cfg.dt)

    # Split into forward / lateral and damp the lateral part -- grip.
    v_f = (vel * fwd).sum(1)
    v_l = (vel * right).sum(1) * cfg.lateral_grip

    # No reverse gear. Without this, braking past a standstill drives the car
    # backwards, and a car that shuffles back and forth farms distance-based
    # fitness forever without ever going anywhere.
    v_f = np.maximum(v_f, 0.0)

    vel = fwd * v_f[:, None] + right * v_l[:, None]

    # Clamp magnitude, guarding the zero-speed divide.
    mag = np.linalg.norm(vel, axis=1)
    over = mag > cfg.max_speed
    scale = np.ones_like(mag)
    scale[over] = cfg.max_speed / mag[over]
    vel *= scale[:, None]

    return CarState(
        pos=(s.pos + vel * cfg.dt).astype(np.float32),
        vel=vel.astype(np.float32),
        angle=angle.astype(np.float32),
    )
