import numpy as np
from config import Config
from src.net import (unpack, forward_batch, forward_with_hidden,
                     random_population, LAYOUT)

CFG = Config()


def test_random_population_shape():
    g = random_population(CFG.population, CFG, np.random.default_rng(0))
    assert g.shape == (CFG.population, CFG.genome_size) == (100, 99)


def test_unpack_shapes_and_total_size():
    g = random_population(4, CFG, np.random.default_rng(0))
    w1, b1, w2, b2 = unpack(g, CFG)
    assert w1.shape == (4, 8, 8) and b1.shape == (4, 8)
    assert w2.shape == (4, 8, 3) and b2.shape == (4, 3)
    assert sum(int(np.prod(s)) for s in LAYOUT(CFG).values()) == CFG.genome_size


def test_unpack_uses_every_gene_exactly_once():
    """A packing bug that drops or reuses genes would silently shrink the
    search space and be nearly impossible to spot from behaviour alone."""
    g = np.arange(CFG.genome_size, dtype=np.float32)[None, :]
    parts = unpack(g, CFG)
    seen = np.concatenate([p.ravel() for p in parts])
    np.testing.assert_array_equal(np.sort(seen), np.arange(CFG.genome_size))


def test_forward_output_range_and_shape():
    g = random_population(10, CFG, np.random.default_rng(1))
    x = np.random.default_rng(2).uniform(0, 1, (10, 8)).astype(np.float32)
    out = forward_batch(g, x, CFG)
    assert out.shape == (10, 3)
    assert (out[:, 0] >= 0).all() and (out[:, 0] <= 1).all()   # throttle
    assert (out[:, 1] >= 0).all() and (out[:, 1] <= 1).all()   # brake
    assert (out[:, 2] >= -1).all() and (out[:, 2] <= 1).all()  # steer


def test_batch_matches_individual_evaluation():
    """The einsum must agree with evaluating each genome on its own."""
    g = random_population(5, CFG, np.random.default_rng(3))
    x = np.random.default_rng(4).uniform(0, 1, (5, 8)).astype(np.float32)
    batched = forward_batch(g, x, CFG)
    for i in range(5):
        single = forward_batch(g[i:i + 1], x[i:i + 1], CFG)
        np.testing.assert_allclose(batched[i], single[0], rtol=1e-5, atol=1e-6)


def test_identical_genomes_give_identical_output():
    g = random_population(1, CFG, np.random.default_rng(5))
    pair = np.vstack([g, g])
    x = np.tile(np.float32([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), (2, 1))
    out = forward_batch(pair, x, CFG)
    np.testing.assert_array_equal(out[0], out[1])


def test_different_inputs_give_different_outputs():
    """Guards against an all-zero or saturated net that ignores its senses."""
    g = random_population(20, CFG, np.random.default_rng(9))
    a = forward_batch(g, np.zeros((20, 8), dtype=np.float32), CFG)
    b = forward_batch(g, np.ones((20, 8), dtype=np.float32), CFG)
    assert not np.allclose(a, b)


def test_extreme_inputs_stay_finite():
    g = random_population(3, CFG, np.random.default_rng(6)) * 50
    x = np.full((3, 8), 1e6, dtype=np.float32)
    assert np.isfinite(forward_batch(g, x, CFG)).all()


def test_hidden_variant_matches_plain_forward():
    g = random_population(6, CFG, np.random.default_rng(7))
    x = np.random.default_rng(8).uniform(0, 1, (6, 8)).astype(np.float32)
    out, hidden = forward_with_hidden(g, x, CFG)
    np.testing.assert_array_equal(out, forward_batch(g, x, CFG))
    assert hidden.shape == (6, CFG.n_hidden)
    assert (np.abs(hidden) <= 1.0).all(), "tanh hidden layer must be bounded"
