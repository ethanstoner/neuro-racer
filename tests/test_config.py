import numpy as np
from config import Config


def test_genome_size_matches_layer_shapes():
    c = Config()
    assert c.genome_size == 8 * 8 + 8 + 8 * 3 + 3 == 99


def test_ray_angles_are_symmetric_and_centred():
    c = Config()
    a = c.ray_angles
    assert len(a) == 7
    assert a[3] == 0.0
    np.testing.assert_allclose(a, -a[::-1], atol=1e-12)


def test_max_ticks():
    assert Config().max_ticks == 2400
