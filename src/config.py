"""Every tunable in one place. No logic lives here."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    # world
    width: int = 1200
    height: int = 800
    track_width: float = 90.0
    dt: float = 1.0 / 60.0
    max_episode_seconds: float = 40.0

    # sensors
    n_rays: int = 7
    ray_spread_deg: float = 90.0      # +/- this, evenly spaced
    ray_length: float = 250.0
    ray_samples: int = 48

    # car
    max_speed: float = 420.0
    accel: float = 600.0
    brake: float = 900.0
    drag: float = 0.4
    lateral_grip: float = 0.92
    max_steer_rate: float = 3.2
    steer_speed_falloff: float = 300.0
    car_radius: float = 6.0

    # network
    n_inputs: int = 8                 # 7 rays + normalised speed
    n_hidden: int = 8
    n_outputs: int = 3                # throttle, brake, steer

    # evolution
    population: int = 100
    elites: int = 5
    tournament_k: int = 3
    mutation_sigma_start: float = 0.30
    mutation_sigma_min: float = 0.05
    mutation_decay: float = 0.97
    mutation_rate: float = 0.15       # fraction of genes touched

    # fitness
    w_distance: float = 0.05
    w_speed: float = 0.02
    lap_bonus: float = 10_000.0
    time_bonus_rate: float = 10.0
    idle_seconds: float = 3.0
    progress_scale: float = 1000.0    # progress is 0..1 per lap; scale to useful units

    seed: int = 1

    @property
    def max_ticks(self) -> int:
        return int(self.max_episode_seconds / self.dt)

    @property
    def ray_angles(self) -> np.ndarray:
        s = np.radians(self.ray_spread_deg)
        return np.linspace(-s, s, self.n_rays)

    @property
    def genome_size(self) -> int:
        return (self.n_inputs * self.n_hidden + self.n_hidden
                + self.n_hidden * self.n_outputs + self.n_outputs)


DEFAULT = Config()
