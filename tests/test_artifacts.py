import json
import numpy as np
from config import Config
from src.artifacts import RunRecorder

CFG = Config()


def _record(rec, gen=0, genome=None, scores=(5.0, 3.0, 1.0), laps=(0, 0, 0),
            lap_time=None, alive=(True, False, False)):
    g = genome if genome is not None else np.arange(CFG.genome_size, dtype=np.float32)
    lt = lap_time if lap_time is not None else np.float32([np.inf] * len(scores))
    rec.record(gen, g, np.float32(scores), np.int32(laps), lt, np.array(alive))
    return g


def test_recorder_writes_champions_and_history(tmp_path):
    rec = RunRecorder(tmp_path, CFG, track_name="oval")
    g = _record(rec)
    rec.close()

    champs = json.loads((tmp_path / "champions.json").read_text())
    assert champs[0]["generation"] == 0
    assert champs[0]["best_score"] == 5.0
    assert champs[0]["track"] == "oval"
    np.testing.assert_allclose(champs[0]["genome"], g)

    lines = (tmp_path / "history.csv").read_text().strip().splitlines()
    assert lines[0].startswith("generation,")
    assert len(lines) == 2


def test_champion_roundtrips_to_a_usable_genome(tmp_path):
    rec = RunRecorder(tmp_path, CFG, "oval")
    g = np.random.default_rng(0).normal(size=CFG.genome_size).astype(np.float32)
    _record(rec, genome=g, scores=(1.0,), laps=(1,), lap_time=np.float32([9.5]),
            alive=(True,))
    rec.close()
    loaded = RunRecorder.load_champion(tmp_path)
    np.testing.assert_allclose(loaded, g, rtol=1e-6)
    assert loaded.shape == (CFG.genome_size,)


def test_champions_file_is_valid_after_every_generation(tmp_path):
    """Rewritten each generation on purpose: an interrupted run must still
    leave a complete, parseable file rather than a truncated one."""
    rec = RunRecorder(tmp_path, CFG, "oval")
    for gen in range(5):
        _record(rec, gen=gen)
        data = json.loads((tmp_path / "champions.json").read_text())
        assert len(data) == gen + 1
        assert all("genome" in d for d in data)
    rec.close()


def test_best_lap_recorded_only_when_finite(tmp_path):
    rec = RunRecorder(tmp_path, CFG, "oval")
    _record(rec, gen=0)                                        # nobody finished
    _record(rec, gen=1, lap_time=np.float32([np.inf, 7.5, np.inf]))
    rec.close()
    champs = RunRecorder.load_all(tmp_path)
    assert champs[0]["best_lap"] is None
    assert champs[1]["best_lap"] == 7.5


def test_config_is_saved_alongside_the_run(tmp_path):
    """A champion is meaningless without the config it was trained under."""
    rec = RunRecorder(tmp_path, CFG, "snake")
    _record(rec)
    rec.close()
    meta = json.loads((tmp_path / "config.json").read_text())
    assert meta["track"] == "snake"
    assert meta["config"]["population"] == CFG.population
    assert meta["config"]["n_hidden"] == CFG.n_hidden


def test_load_champion_by_generation(tmp_path):
    rec = RunRecorder(tmp_path, CFG, "oval")
    first = _record(rec, gen=0, genome=np.zeros(CFG.genome_size, dtype=np.float32))
    last = _record(rec, gen=1, genome=np.ones(CFG.genome_size, dtype=np.float32))
    rec.close()
    np.testing.assert_allclose(RunRecorder.load_champion(tmp_path, 0), first)
    np.testing.assert_allclose(RunRecorder.load_champion(tmp_path, -1), last)
