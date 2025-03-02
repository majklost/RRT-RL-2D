import pygame
import pygame.freetype

import time
import pymunk
from pathlib import Path
from pymunk.pygame_util import DrawOptions

from .base_renderer import BaseRenderer
from ..simulator.standard_config import StandardConfig


class EnvRenderer(BaseRenderer):
    """
    Handle inner rendering of the environment.
    """

    def __init__(self, cfg: StandardConfig):
        self.cfg = cfg
        self.clbs = []
        self.initiated = False

    def _delayed_init(self):
        pygame.init()
        width = self.cfg['width']
        height = self.cfg['height']
        self.fps = self.cfg['fps']
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.options = DrawOptions(self.screen)
        self.options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        self.font = self._create_font()

    def register_callback(self, clb):
        self.clbs.append(clb)

    def _additional_render(self, screen):
        for clb in self.clbs:
            clb(screen, self.font)

    def render(self, simulator):
        if not self.initiated:
            self._delayed_init()
            self.initiated = True

        self.screen.fill((255, 255, 255))
        self._additional_render(self.screen)
        simulator.draw_on(self.options)
        self.clock.tick(self.fps)
        pygame.display.flip()

    def close(self):
        pygame.display.quit()
        pygame.quit()
