import pygame
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
        pygame.init()
        width = cfg['width']
        height = cfg['height']
        self.fps = cfg['fps']
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.options = DrawOptions(self.screen)
        self.options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        self.font = self._create_font()

    @staticmethod
    def _create_font():
        pygame.font.init()
        font_path = Path(__file__).parent / 'Arial.ttf'
        return pygame.freetype.Font(font_path, 20)

    def _additional_render(self, screen):
        pass

    def render(self, simulator):
        self.screen.fill((255, 255, 255))
        simulator.draw_on(self.options)
        self._additional_render(self.screen)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.displays.quit()
        pygame.quit()
