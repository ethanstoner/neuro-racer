import numpy as np
from config import Config
from src.sensors import cast

CFG = Config()


def corridor_mask(cfg, half_width=40):
    """Horizontal corridor centred on y=400, walls above and below."""
    m = np.zeros((cfg.height, cfg.width), dtype=bool)
    m[400 - half_width:400 + half_width, :] = True
    m[:, 0] = m[:, -1] = False
    return m


def test_side_rays_measure_the_corridor_walls():
    mask = corridor_mask(CFG, 40)
    pos = np.float32([[600.0, 400.0]])
    angle = np.float32([0.0])            # pointing +x, along the corridor
    d = cast(pos, angle, mask, CFG)       # (1, 7) normalised
    tol = 2 * CFG.ray_length / CFG.ray_samples
    # rays 0 and 6 are +/-90 deg: straight at the walls, 40px away
    assert abs(d[0, 0] * CFG.ray_length - 40) <= tol
    assert abs(d[0, 6] * CFG.ray_length - 40) <= tol


def test_forward_ray_sees_nothing_and_saturates():
    mask = corridor_mask(CFG, 40)
    d = cast(np.float32([[600.0, 400.0]]), np.float32([0.0]), mask, CFG)
    assert d[0, 3] == 1.0


def test_diagonal_ray_matches_trigonometry():
    mask = corridor_mask(CFG, 40)
    d = cast(np.float32([[600.0, 400.0]]), np.float32([0.0]), mask, CFG)
    expected = 40.0 / np.sin(np.radians(60.0))   # ray at +/-60 deg
    tol = 2 * CFG.ray_length / CFG.ray_samples
    assert abs(d[0, 5] * CFG.ray_length - expected) <= tol


def test_output_is_normalised_and_finite():
    mask = corridor_mask(CFG, 40)
    rng = np.random.default_rng(3)
    pos = rng.uniform([100, 380], [1100, 420], size=(50, 2)).astype(np.float32)
    ang = rng.uniform(-np.pi, np.pi, 50).astype(np.float32)
    d = cast(pos, ang, mask, CFG)
    assert d.shape == (50, 7)
    assert np.isfinite(d).all() and (d >= 0).all() and (d <= 1).all()


def test_rays_rotate_with_the_car():
    mask = corridor_mask(CFG, 40)
    a = cast(np.float32([[600.0, 400.0]]), np.float32([0.0]), mask, CFG)
    b = cast(np.float32([[600.0, 400.0]]), np.float32([np.pi / 2]), mask, CFG)
    # rotated 90 deg, the formerly-blind forward ray now stares at a wall
    assert b[0, 3] < 1.0 and a[0, 3] == 1.0


def test_out_of_bounds_position_reads_as_wall():
    """A car that left the world must still get finite readings, not an
    IndexError -- early random genomes drive off the map constantly."""
    mask = corridor_mask(CFG, 40)
    d = cast(np.float32([[-50.0, -50.0]]), np.float32([0.0]), mask, CFG)
    assert np.isfinite(d).all()


def test_narrower_corridor_reads_closer():
    wide = cast(np.float32([[600.0, 400.0]]), np.float32([0.0]),
                corridor_mask(CFG, 80), CFG)
    narrow = cast(np.float32([[600.0, 400.0]]), np.float32([0.0]),
                  corridor_mask(CFG, 20), CFG)
    assert narrow[0, 0] < wide[0, 0]


def test_batch_matches_individual():
    mask = corridor_mask(CFG, 40)
    rng = np.random.default_rng(11)
    pos = rng.uniform([200, 380], [1000, 420], size=(8, 2)).astype(np.float32)
    ang = rng.uniform(-np.pi, np.pi, 8).astype(np.float32)
    batched = cast(pos, ang, mask, CFG)
    for i in range(8):
        np.testing.assert_array_equal(batched[i], cast(pos[i:i + 1], ang[i:i + 1], mask, CFG)[0])
