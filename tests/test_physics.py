import numpy as np
from src.config import Config
from src.physics import CarState, step

CFG = Config()


def make(n=4):
    return CarState.spawn(n, x=100.0, y=100.0, heading=0.0)


def full_throttle(n):
    c = np.zeros((n, 3), dtype=np.float32)
    c[:, 0] = 1.0
    return c


def test_full_throttle_accelerates_forward():
    s = make()
    s = step(s, full_throttle(4), CFG)
    assert (s.vel[:, 0] > 0).all()
    np.testing.assert_allclose(s.vel[:, 1], 0.0, atol=1e-6)


def test_speed_saturates_below_max():
    s = make(1)
    for _ in range(4000):
        s = step(s, full_throttle(1), CFG)
    speed = np.linalg.norm(s.vel, axis=1)
    assert speed[0] <= CFG.max_speed + 1e-3
    assert speed[0] > CFG.max_speed * 0.5, "drag is swamping acceleration"


def test_deterministic():
    rng = np.random.default_rng(0)
    controls = rng.uniform(-1, 1, size=(200, 3, 3)).astype(np.float32)
    outs = []
    for _ in range(2):
        s = make(3)
        for c in controls:
            s = step(s, c, CFG)
        outs.append(s.pos.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


def test_adversarial_controls_never_produce_nan():
    """The network outputs arbitrary floats early on. Physics must survive them."""
    s = make(6)
    wild = np.array([[1e6, -1e6, 1e6]] * 6, dtype=np.float32)
    for _ in range(500):
        s = step(s, wild, CFG)
    assert np.isfinite(s.pos).all() and np.isfinite(s.vel).all()
    assert np.linalg.norm(s.vel, axis=1).max() <= CFG.max_speed + 1e-3


def test_steering_is_weaker_at_high_speed():
    """This is what forces the AI to brake for corners."""
    slow = make(1)
    fast = make(1)
    fast.vel[:, 0] = CFG.max_speed
    turn = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    d_slow = step(slow, turn, CFG).angle[0] - slow.angle[0]
    d_fast = step(fast, turn, CFG).angle[0] - fast.angle[0]
    assert abs(d_fast) < abs(d_slow)


def test_lateral_velocity_is_damped():
    s = make(1)
    s.vel[:] = [0.0, 200.0]           # pure sideways, car points along +x
    before = abs(s.vel[0, 1])
    s = step(s, np.zeros((1, 3), dtype=np.float32), CFG)
    assert abs(s.vel[0, 1]) < before


def test_braking_slows_the_car():
    s = make(1)
    for _ in range(120):
        s = step(s, full_throttle(1), CFG)
    fast = np.linalg.norm(s.vel, axis=1)[0]
    brake = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    for _ in range(30):
        s = step(s, brake, CFG)
    assert np.linalg.norm(s.vel, axis=1)[0] < fast


def test_car_never_reverses_under_braking():
    """Brake should stop the car, not drive it backwards -- a reversing car
    would farm distance-based fitness by shuffling back and forth."""
    s = make(1)
    brake = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    for _ in range(200):
        s = step(s, brake, CFG)
    fwd = np.stack([np.cos(s.angle), np.sin(s.angle)], axis=1)
    assert (s.vel * fwd).sum() >= -1e-3
