import pygame
import pymunk
from pymunk.pygame_util import DrawOptions

from ..simulator.standard_config import STANDARD_CONFIG
from ..simulator.simulator import Simulator
from ..controllers.base_controller import BaseController
from .base_renderer import BaseRenderer


class DebugRenderer(BaseRenderer):
    def __init__(self, cfg=STANDARD_CONFIG, render_constraints=False, realtime=True):
        self._render_constraints = render_constraints
        self._realtime = realtime
        self._fps = cfg['fps']
        w = cfg['width']
        h = cfg['height']
        self.display = pygame.display.set_mode((w, h))
        self.cur_scene = pygame.surface.Surface((w, h))
        self._options = DrawOptions(self.cur_scene)
        self._clock = pygame.time.Clock()
        self._running = True
        if not render_constraints:
            self._options.flags = DrawOptions.DRAW_SHAPES

    def attach_controller(self, controller: BaseController):
        self._controller = controller

    def _additional_drawings(self):
        pass

    def render(self, simulator: Simulator):
        self.cur_scene.fill((255, 255, 255))
        simulator.draw_on(self._options)
        self._additional_drawings()

        self.display.blit(self.cur_scene, (0, 0))
        pygame.display.update()
        if self._realtime:
            self._clock.tick(self._fps)
        if self._controller is not None:
            self._controller.update()
        else:
            self._mark_end_custom()

    def _mark_end_custom(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
