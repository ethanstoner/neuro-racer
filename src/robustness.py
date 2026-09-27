"""Is it a policy, or one memorised trajectory?

Training starts every car from the same pose facing the same way, so a genome
can score well by encoding a fixed open-loop sequence of turns that happens to
fit the track, without ever reading its sensors. Lap time cannot tell the two
apart -- the fastest champion this project produced was the one that could not
drive.

Dropping the champion at many poses around the lap can. A real policy recovers
from any of them; a memorised trajectory only works from the pose it was born
at. Kept free of pygame so it runs in CI alongside the rest of the trainer.
"""
import numpy as np
from src.config import Config


def spawn_grid(track, cfg: Config, n_points: int = 24, n_offsets: int = 3) -> np.ndarray:
    """Poses spread around the lap and across the track width.

    Headings follow the centerline tangent. Facing a car across the track would
    make the test measure the harness rather than the champion.
    """
    c = track.centerline
    m = len(c)
    idx = np.linspace(0, m, n_points, endpoint=False).astype(int)
    # 0.7 of the usable half-width: far enough off-line to break a memorised
    # trajectory, not so far that the car starts already touching a wall.
    span = (cfg.track_width / 2 - cfg.car_radius) * 0.7
    lateral = np.linspace(-span, span, n_offsets)

    poses = []
    for i in idx:
        tangent = c[(i + 3) % m] - c[i]
        heading = np.arctan2(tangent[1], tangent[0])
        normal = np.array([-np.sin(heading), np.cos(heading)])
        for off in lateral:
            pt = c[i] + normal * off
            poses.append((pt[0], pt[1], heading))
    return np.array(poses, dtype=np.float32)


def random_pose(track, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One spawn pose drawn from the same distribution spawn_grid covers: anywhere
    along the lap, up to 0.7 of the usable half-width off the line, facing along
    the track. Resampled until the car's body fits, which it nearly always does.
    """
    c = track.centerline
    m = len(c)
    span = (cfg.track_width / 2 - cfg.car_radius) * 0.7
    while True:
        i = int(rng.integers(m))
        tangent = c[(i + 3) % m] - c[i]
        heading = np.arctan2(tangent[1], tangent[0])
        normal = np.array([-np.sin(heading), np.cos(heading)])
        pt = c[i] + normal * rng.uniform(-span, span)
        if track.body_ok[int(pt[1]), int(pt[0])]:
            return np.array([pt[0], pt[1], heading], dtype=np.float32)


def assess(genome, track, cfg: Config, points: int = 24, offsets: int = 3) -> dict:
    """Run one genome from every viable pose on `track`; count the laps.

    Imported lazily so that importing this module stays cheap for the tests
    that only exercise spawn_grid.
    """
    from src.simulation import run_generation

    poses = spawn_grid(track, cfg, points, offsets)

    # Only spawn where the car's body actually fits. A pose inside a wall is a
    # broken start, not a failed one, and counting it would understate every
    # champion equally and invisibly.
    ok = track.body_ok[poses[:, 1].astype(int), poses[:, 0].astype(int)]
    poses = poses[ok]
    n = int(ok.sum())
    if n == 0:
        return {"track": track.name, "starts": 0, "survived": 0,
                "lapped": 0, "rate": 0.0, "best_lap": None}

    genomes = np.tile(np.asarray(genome, dtype=np.float32), (n, 1))
    r = run_generation(genomes, track, cfg, starts=poses)
    finite = r.lap_times[np.isfinite(r.lap_times)]
    lapped = int((r.laps > 0).sum())
    return {
        "track": track.name,
        "starts": n,
        "survived": int(r.alive.sum()),
        "lapped": lapped,
        "rate": lapped / n,
        "best_lap": float(finite.min()) if len(finite) else None,
    }
