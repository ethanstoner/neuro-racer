"""A tiny MLP, 8 -> 8 -> 3, evaluated for the whole population at once.

The genome is one flat float32 vector, which makes crossover and mutation
plain array operations. The network is kept small on purpose: at 99 weights
every connection can be drawn individually in the visualiser and stay legible,
and a small search space is exactly what a genetic algorithm wants.
"""
import numpy as np
from src.config import Config


def LAYOUT(cfg: Config) -> dict:
    return {
        "w1": (cfg.n_inputs, cfg.n_hidden),
        "b1": (cfg.n_hidden,),
        "w2": (cfg.n_hidden, cfg.n_outputs),
        "b2": (cfg.n_outputs,),
    }


def unpack(genomes: np.ndarray, cfg: Config):
    """(N, genome_size) -> w1 (N,I,H), b1 (N,H), w2 (N,H,O), b2 (N,O)."""
    n = len(genomes)
    out, i = [], 0
    for shape in LAYOUT(cfg).values():
        size = int(np.prod(shape))
        out.append(genomes[:, i:i + size].reshape((n,) + shape))
        i += size
    return tuple(out)


def random_population(n: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Small initial weights: large random weights saturate tanh immediately,
    which flattens the fitness landscape and stalls the early generations."""
    return rng.normal(0.0, 0.5, size=(n, cfg.genome_size)).astype(np.float32)


def forward_with_hidden(genomes: np.ndarray, x: np.ndarray, cfg: Config):
    """Returns (outputs (N,3), hidden (N,H)).

    The visualiser needs the hidden activations to light up neurons, and the
    simulation needs the outputs. Computing both once avoids doing the forward
    pass twice per tick when recording.
    """
    w1, b1, w2, b2 = unpack(genomes, cfg)
    h = np.tanh(np.einsum("ni,nih->nh", x, w1) + b1)
    o = np.tanh(np.einsum("nh,nho->no", h, w2) + b2)
    out = np.stack([(o[:, 0] + 1.0) * 0.5,        # throttle 0..1
                    (o[:, 1] + 1.0) * 0.5,        # brake    0..1
                    o[:, 2]], axis=1)             # steer   -1..1
    return out.astype(np.float32), h.astype(np.float32)


def forward_batch(genomes: np.ndarray, x: np.ndarray, cfg: Config) -> np.ndarray:
    """x is (N, n_inputs). Returns (N, 3): throttle 0..1, brake 0..1, steer -1..1."""
    return forward_with_hidden(genomes, x, cfg)[0]
