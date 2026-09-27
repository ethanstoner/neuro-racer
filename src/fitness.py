"""Scoring. The design here is what decides whether the whole project works.

Two rules matter more than the rest:
  1. Crashing must not be punished harder than doing nothing, or generation 1
     converges on parked cars and never explores the track.
  2. Lap time cannot be the objective until laps are achievable, so scoring is
     staged: progress first, then time takes over once a lap is completed.
"""
import numpy as np
from src.config import Config


def wrapped_delta(now: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Progress difference, correct across the start/finish seam.

    Progress is in [0, 1) and wraps. A raw subtraction would score completing
    a lap as -0.99. Deltas beyond +/-0.5 in one tick are physically impossible
    at any reachable speed, so folding into [-0.5, 0.5) is unambiguous.
    """
    return (now - prev + 0.5) % 1.0 - 0.5


class FitnessTracker:
    def __init__(self, n: int, cfg: Config):
        self.cfg = cfg
        self.n = n
        # Accumulators are float64. These sum thousands of small increments over
        # an episode, and in float32 the rounding is enough to land a completed
        # lap at 0.99999994 -- denying a car the lap bonus it actually earned.
        self.cum = np.zeros(n, dtype=np.float64)        # cumulative lap fraction
        self.best_cum = np.zeros(n, dtype=np.float64)
        self.distance = np.zeros(n, dtype=np.float64)
        self.speed_sum = np.zeros(n, dtype=np.float64)
        self.ticks = np.zeros(n, dtype=np.int32)
        self.idle_ticks = np.zeros(n, dtype=np.int32)
        self.laps = np.zeros(n, dtype=np.int32)
        self.lap_time = np.full(n, np.inf, dtype=np.float32)
        self.time = np.zeros(n, dtype=np.float32)
        self.prev = None
        self._idle_limit = int(cfg.idle_seconds / cfg.dt)

    def update(self, progress, speed, alive, dt) -> np.ndarray:
        """Accumulate one tick. Returns the updated alive mask."""
        if self.prev is None:
            self.prev = np.asarray(progress, dtype=np.float32).copy()
            return alive

        a = alive.copy()
        d = wrapped_delta(progress, self.prev) * a
        self.prev = np.where(a, progress, self.prev).astype(np.float32)

        self.cum += d
        self.time += dt * a
        self.ticks += a
        self.distance += speed * dt * a
        self.speed_sum += speed * a

        # Anti-idle: only new forward progress resets the timer, so a car
        # spinning in place or oscillating back and forth still gets culled.
        improved = self.cum > self.best_cum + 1e-4
        self.best_cum = np.maximum(self.best_cum, self.cum)
        self.idle_ticks = np.where(improved, 0, self.idle_ticks + a)
        a &= self.idle_ticks < self._idle_limit

        # Lap completion. Recorded once, at the first crossing. The epsilon
        # keeps the check off a floating-point knife edge; 1e-4 of a lap is a
        # fraction of a pixel, so it cannot credit a lap that was not driven.
        finished = a & (self.cum >= 1.0 - 1e-4) & (self.laps == 0)
        self.laps += finished
        self.lap_time = np.where(finished, self.time, self.lap_time)

        return a

    def scores(self) -> np.ndarray:
        c = self.cfg
        mean_speed = self.speed_sum / np.maximum(self.ticks, 1)
        base = (np.maximum(self.best_cum, 0.0) * c.progress_scale
                + c.w_distance * self.distance
                + c.w_speed * mean_speed)
        lapped = self.laps > 0
        time_score = (c.lap_bonus
                      + (c.max_episode_seconds - self.lap_time) * c.time_bonus_rate)
        return np.where(lapped, time_score, base).astype(np.float32)
