"""Did the champion learn a policy, or memorise one trajectory?

Training always starts every car from the same pose facing the same way, which
means a genome can score well by encoding a single open-loop sequence of turns
that happens to fit the track -- without ever reading its sensors.

This drops the champion at many points around the lap, at several lateral
offsets from the centerline, and counts how many of those starts it survives.
A real policy recovers from any of them. A memorised trajectory only works from
the one pose it was born at.

  python robustness.py --run runs/oval-seed1
  python robustness.py --run runs/snake-seed1 --track snake
"""
import argparse
import numpy as np
from config import Config
from src.tracks import load, BUILDERS
from src.artifacts import RunRecorder
from src.simulation import run_generation

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--track", default=None)
p.add_argument("--points", type=int, default=24, help="spawn points around the lap")
p.add_argument("--offsets", type=int, default=3, help="lateral offsets per point")
p.add_argument("--all-tracks", action="store_true")
args = p.parse_args()

champs = RunRecorder.load_all(args.run)
entry = champs[-1]
genome = np.array(entry["genome"], dtype=np.float32)


def spawn_grid(track, cfg, n_points, n_offsets):
    """Poses spread around the lap and across the track width."""
    c = track.centerline
    m = len(c)
    idx = np.linspace(0, m, n_points, endpoint=False).astype(int)
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


def assess(track_name):
    cfg = Config()
    track = load(track_name, cfg)
    poses = spawn_grid(track, cfg, args.points, args.offsets)
    n = len(poses)
    genomes = np.tile(genome, (n, 1))

    # Only spawn where the car actually fits.
    ok = track.body_ok[poses[:, 1].astype(int), poses[:, 0].astype(int)]
    poses, genomes, n = poses[ok], genomes[ok], int(ok.sum())

    r = run_generation(genomes, track, cfg, starts=poses)
    survived = int(r.alive.sum())
    lapped = int((r.laps > 0).sum())
    laps_time = r.lap_times[np.isfinite(r.lap_times)]
    return {
        "track": track_name,
        "starts": n,
        "survived": survived,
        "lapped": lapped,
        "best_lap": float(laps_time.min()) if len(laps_time) else None,
    }


trained_on = entry.get("track", "?")
targets = sorted(BUILDERS) if args.all_tracks else [args.track or trained_on]

print(f"champion gen {entry['generation']}, trained on {trained_on}")
print(f"{'track':9} {'starts':>7} {'survived':>9} {'completed a lap':>16} {'best':>8}")
for name in targets:
    r = assess(name)
    lap = f"{r['best_lap']:.2f}s" if r["best_lap"] else "--"
    pct = 100 * r["lapped"] / max(r["starts"], 1)
    print(f"{r['track']:9} {r['starts']:7d} {r['survived']:9d} "
          f"{r['lapped']:9d} ({pct:3.0f}%) {lap:>8}")
