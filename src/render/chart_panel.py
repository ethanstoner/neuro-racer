"""Progress by generation.

Two phases, because the thing worth watching changes partway through the run.

Before any car finishes a lap, fitness is the signal: best and mean climb as the
population learns to survive and make progress. The moment a lap is completed,
best fitness leaps by the lap bonus and then barely moves -- plotted raw it is
a flat line pinned to the top of the axis, and everything before it is squashed
into the floor. So once laps exist the chart switches the headline series to
best lap time, which is what is actually being optimised from then on, and
keeps mean fitness alongside on its own scale as the population-health signal.

No matplotlib and no second window: the point of the layout is that the curve
sits next to the cars producing it.
"""
import numpy as np
import pygame
from src.render import palette as P


def _polyline(values, plot, invert=False):
    """Map a series onto the plot rect using its own min/max."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    norm = (v - lo) / (hi - lo)
    if invert:
        norm = 1.0 - norm
    xs = plot.x + plot.width * np.arange(len(v)) / max(len(v) - 1, 1)
    ys = plot.bottom - plot.height * norm
    return [(int(a), int(b)) for a, b in zip(xs, ys)], lo, hi


class ChartPanel:
    def __init__(self, rect, cfg):
        self.rect = pygame.Rect(rect)
        self.cfg = cfg
        self.font = pygame.font.SysFont("consolas", 12)
        self.title = pygame.font.SysFont("consolas", 14, bold=True)

    def draw(self, surf, best_history, mean_history, lap_history, lap_times):
        r = self.rect
        pygame.draw.rect(surf, P.PANEL, r)
        pygame.draw.line(surf, P.KERB, (r.x, r.y), (r.right, r.y))

        finished = [t for t in lap_times if t is not None]
        heading = "BEST LAP TIME" if finished else "FITNESS BY GENERATION"
        surf.blit(self.title.render(heading, True, P.TEXT), (r.x + 14, r.y + 12))

        if len(best_history) < 2:
            surf.blit(self.font.render("collecting...", True, P.TEXT_DIM),
                      (r.x + 14, r.y + 42))
            return

        plot = pygame.Rect(r.x + 16, r.y + 44, r.width - 32, r.height - 84)
        pygame.draw.line(surf, P.KERB, (plot.x, plot.y), (plot.x, plot.bottom))
        pygame.draw.line(surf, P.KERB, (plot.x, plot.bottom), (plot.right, plot.bottom))

        first_lap = next((i for i, n in enumerate(lap_history) if n > 0), None)
        if first_lap is not None:
            x = plot.x + plot.width * first_lap / max(len(best_history) - 1, 1)
            pygame.draw.line(surf, P.CAR_LEAD, (int(x), plot.y), (int(x), plot.bottom), 1)

        # Mean fitness is always shown -- it is the population-health signal,
        # and it keeps improving long after the best has plateaued.
        mean_pts, mean_lo, mean_hi = _polyline(mean_history, plot)
        pygame.draw.lines(surf, P.TEXT_DIM, False, mean_pts, 1)

        legend_y = r.bottom - 30
        if finished:
            # Carry the best-so-far forward through generations with no finisher,
            # so the series has no holes and only ever moves downward.
            running, carried = None, []
            for t in lap_times:
                if t is not None:
                    running = t if running is None else min(running, t)
                carried.append(running if running is not None else np.nan)
            carried = np.array(carried, dtype=np.float64)
            valid = ~np.isnan(carried)
            if valid.sum() >= 2:
                sub = pygame.Rect(plot.x + int(plot.width * np.argmax(valid) /
                                               max(len(carried) - 1, 1)),
                                  plot.y, 1, plot.height)
                pts, lo, hi = _polyline(carried[valid], pygame.Rect(
                    sub.x, plot.y, max(plot.right - sub.x, 1), plot.height), invert=True)
                pygame.draw.lines(surf, P.CAR_LEAD, False, pts, 2)
                surf.blit(self.font.render(f"{lo:.2f}s", True, P.CAR_LEAD),
                          (plot.x + 4, plot.y - 2))
                surf.blit(self.font.render(f"{hi:.2f}s", True, P.CAR_LEAD),
                          (plot.x + 4, plot.bottom - 14))
            surf.blit(self.font.render(
                f"lap {min(finished):.2f}s", True, P.CAR_LEAD), (plot.x, legend_y))
            surf.blit(self.font.render(
                f"mean fit {mean_history[-1]:,.0f}", True, P.TEXT_DIM),
                (plot.x + 110, legend_y))
        else:
            best_pts, lo, hi = _polyline(best_history, plot)
            pygame.draw.lines(surf, P.CAR_ALIVE, False, best_pts, 2)
            surf.blit(self.font.render(f"{hi:,.0f}", True, P.TEXT_DIM),
                      (plot.x + 4, plot.y - 2))
            surf.blit(self.font.render(f"{lo:,.0f}", True, P.TEXT_DIM),
                      (plot.x + 4, plot.bottom - 14))
            surf.blit(self.font.render(
                f"best {best_history[-1]:,.0f}", True, P.CAR_ALIVE), (plot.x, legend_y))
            surf.blit(self.font.render(
                f"mean {mean_history[-1]:,.0f}", True, P.TEXT_DIM),
                (plot.x + 110, legend_y))

        surf.blit(self.font.render(f"gen {len(best_history) - 1}", True, P.TEXT_DIM),
                  (plot.right - 60, legend_y))
