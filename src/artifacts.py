"""Run artifacts. Auto-saved, no UI.

Without these the gen-60 champion vanishes when you close the window, and the
generalisation experiment -- train on oval, then run that champion cold on a
track it has never seen -- is impossible.
"""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from config import Config

FIELDS = ["generation", "best", "mean", "median", "alive", "laps", "best_lap"]


class RunRecorder:
    def __init__(self, directory, cfg: Config, track_name: str = "", meta: dict | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.track_name = track_name
        self.champions = []
        self._fh = open(self.dir / "history.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._fh)
        self._csv.writerow(FIELDS)
        info = {"track": track_name, **(meta or {}), "config": asdict(cfg)}
        (self.dir / "config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    def record(self, generation, best_genome, scores, laps, lap_time, alive):
        best_lap = float(np.min(lap_time)) if np.isfinite(lap_time).any() else None
        self.champions.append({
            "generation": int(generation),
            "best_score": float(np.max(scores)),
            "best_lap": best_lap,
            "track": self.track_name,
            "genome": [float(x) for x in best_genome],
        })
        self._csv.writerow([generation, float(np.max(scores)), float(np.mean(scores)),
                            float(np.median(scores)), int(alive.sum()),
                            int(laps.max()), best_lap if best_lap else ""])
        self._fh.flush()
        # Rewritten each generation so a crashed or interrupted run still leaves
        # a complete, valid file behind rather than nothing at all.
        (self.dir / "champions.json").write_text(
            json.dumps(self.champions), encoding="utf-8")

    def close(self):
        self._fh.close()

    @staticmethod
    def load_champion(directory, generation: int = -1) -> np.ndarray:
        data = json.loads((Path(directory) / "champions.json").read_text(encoding="utf-8"))
        return np.array(data[generation]["genome"], dtype=np.float32)

    @staticmethod
    def load_all(directory) -> list:
        return json.loads((Path(directory) / "champions.json").read_text(encoding="utf-8"))
