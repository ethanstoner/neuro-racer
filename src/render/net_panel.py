"""Live diagram of the leader's network.

Weights are drawn as individual lines: green positive, red negative, thickness
proportional to magnitude. Neuron fill brightness tracks the current activation,
so you can watch the hidden layer light up as the car approaches a corner.

The network is only 99 weights precisely so this stays legible. A wider hidden
layer would render as a hairball and the panel would stop meaning anything.
"""
import numpy as np
import pygame
from src.net import unpack
from src.render import palette as P

LABELS_IN = ["ray -90", "ray -60", "ray -30", "ray 0", "ray +30", "ray +60",
             "ray +90", "speed"]
LABELS_OUT = ["throttle", "brake", "steer"]

# Weights below this fraction of the largest are not drawn. Without the cull
# the panel is a solid block of lines and no structure is visible.
CULL = 0.08


class NetPanel:
    def __init__(self, rect, cfg):
        self.rect = pygame.Rect(rect)
        self.cfg = cfg
        self.font = pygame.font.SysFont("consolas", 12)
        self.title = pygame.font.SysFont("consolas", 14, bold=True)
        self._coords = self._columns()

    def _columns(self):
        r = self.rect
        top, bottom = r.y + 46, r.bottom - 18
        xs = (r.x + 84, r.centerx + 6, r.right - 78)

        def col(x, n):
            span = bottom - top
            return [(x, top + span * (i + 0.5) / n) for i in range(n)]

        return (col(xs[0], self.cfg.n_inputs),
                col(xs[1], self.cfg.n_hidden),
                col(xs[2], self.cfg.n_outputs))

    def draw(self, surf, genome, inputs, hidden, outputs):
        pygame.draw.rect(surf, P.PANEL, self.rect)
        pygame.draw.line(surf, P.KERB, (self.rect.x, self.rect.y),
                         (self.rect.right, self.rect.y))
        surf.blit(self.title.render("LEADER NETWORK", True, P.TEXT),
                  (self.rect.x + 14, self.rect.y + 12))

        w1, b1, w2, b2 = unpack(genome[None, :], self.cfg)
        w1, w2 = w1[0], w2[0]
        scale = max(float(np.abs(w1).max()), float(np.abs(w2).max()), 1e-6)
        cin, chid, cout = self._coords

        for weights, src, dst in ((w1, cin, chid), (w2, chid, cout)):
            for i, a in enumerate(src):
                for j, b in enumerate(dst):
                    mag = abs(float(weights[i, j])) / scale
                    if mag < CULL:
                        continue
                    base = P.POS if weights[i, j] > 0 else P.NEG
                    fade = 0.25 + 0.75 * mag
                    colour = (int(base[0] * fade), int(base[1] * fade), int(base[2] * fade))
                    pygame.draw.line(surf, colour, a, b, 1 + int(mag * 2.2))

        # Inputs are already 0..1. Hidden is tanh so use magnitude. Outputs are
        # mixed ranges, so normalise each to 0..1 for brightness only.
        out_norm = np.array([outputs[0], outputs[1], abs(outputs[2])], dtype=np.float32)
        self._nodes(surf, cin, np.asarray(inputs), LABELS_IN, "left")
        self._nodes(surf, chid, np.abs(np.asarray(hidden)), None, None)
        self._nodes(surf, cout, out_norm, LABELS_OUT, "right")

    def _nodes(self, surf, coords, values, labels, side):
        for i, (x, y) in enumerate(coords):
            v = float(np.clip(values[i], 0.0, 1.0))
            fill = tuple(int(28 + (c - 28) * v) for c in P.CAR_ALIVE)
            pygame.draw.circle(surf, fill, (int(x), int(y)), 8)
            pygame.draw.circle(surf, P.KERB, (int(x), int(y)), 8, 1)
            if labels:
                t = self.font.render(labels[i], True, P.TEXT_DIM)
                pos = ((x - 14 - t.get_width(), y - 7) if side == "left"
                       else (x + 14, y - 7))
                surf.blit(t, pos)
