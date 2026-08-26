"""Copy the final champion of each run into `champions/`, which is committed.

`runs/` is gitignored -- it holds a genome for every generation and gets large.
But that means nothing in the repo backs up the lap times in the README. This
exports just the final genome per track, with the config it was trained under
and its measured result, so anyone cloning the repo can replay the champions
and check the numbers rather than taking them on trust.

  python tools/export_champions.py
  python evaluate.py --run champions/snake --track oval
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from config import Config  # noqa: E402
from src.tracks import load, BUILDERS  # noqa: E402
from src.artifacts import RunRecorder  # noqa: E402
from src.simulation import run_generation  # noqa: E402

out_root = ROOT / "champions"
out_root.mkdir(exist_ok=True)
summary = []

for track_name in sorted(BUILDERS):
    run_dir = ROOT / "runs" / f"{track_name}-seed1"
    if not (run_dir / "champions.json").exists():
        print(f"{track_name:9} no run found, skipping")
        continue

    all_champs = RunRecorder.load_all(run_dir)
    final = all_champs[-1]
    first_lap = next((c["generation"] for c in all_champs if c["best_lap"]), None)

    dest = out_root / track_name
    dest.mkdir(exist_ok=True)
    # evaluate.py expects a champions.json, so keep the same shape.
    (dest / "champions.json").write_text(json.dumps([final]), encoding="utf-8")
    shutil.copy(run_dir / "config.json", dest / "config.json")
    shutil.copy(run_dir / "history.csv", dest / "history.csv")

    # Re-measure rather than trusting the recorded number.
    cfg = Config(population=1)
    genome = np.array(final["genome"], dtype=np.float32)[None, :]
    r = run_generation(genome, load(track_name, cfg), cfg)
    lap = float(r.lap_times[0]) if r.laps[0] > 0 else None

    summary.append({
        "track": track_name,
        "generations": len(all_champs),
        "first_lap_generation": first_lap,
        "best_lap": lap,
    })
    print(f"{track_name:9} gen {final['generation']:4d}  "
          f"lap {lap:.2f}s" if lap else f"{track_name:9} did not finish")

(out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nwrote {out_root}")
