"""Status bar along the bottom."""
import pygame
from src.render import palette as P

HINT = "1/2/3 speed   H headless burst   SPACE pause   S screenshot   ESC quit"


class Hud:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.font = pygame.font.SysFont("consolas", 17)
        self.small = pygame.font.SysFont("consolas", 13)

    def draw(self, surf, gen, alive, total, best, best_lap, speed_label,
             paused, track_name):
        r = self.rect
        pygame.draw.rect(surf, P.PANEL, r)
        pygame.draw.line(surf, P.KERB, (r.x, r.y), (r.right, r.y))

        lap = f"{best_lap:.2f}s" if best_lap else "--"
        state = "PAUSED" if paused else speed_label
        text = (f"GEN {gen:<5d} alive {alive:>3d}/{total:<4d} "
                f"best {best:>10,.0f}  lap {lap:>7}  {track_name:>8}  {state}")
        surf.blit(self.font.render(text, True, P.TEXT), (r.x + 14, r.y + 9))
        surf.blit(self.small.render(HINT, True, P.TEXT_DIM), (r.x + 14, r.y + 33))
