import pygame
import pymunk
from pymunk.pygame_util import DrawOptions

from ..simulator.standard_config import STANDARD_CONFIG
from ..simulator.simulator import Simulator
from ..controllers.direct_controller import DirectController
from .base_renderer import BaseRenderer


class DebugRenderer(BaseRenderer):
    """
    A renderer that can be attached to Simulator directly
    """

    def __init__(self, cfg=STANDARD_CONFIG, render_constraints=False, realtime=True):
        self._render_constraints = render_constraints
        self._realtime = realtime
        self._controller = None
        self._fps = cfg['fps']
        w = cfg['width']
        h = cfg['height']
        self.display = self._get_display(w, h)
        self.cur_scene = pygame.surface.Surface((w, h))
        self._options = DrawOptions(self.cur_scene)
        self._one_time_canvas = pygame.surface.Surface((w, h))
        self._one_time_canvas.fill((255, 255, 255))
        self._clock = pygame.time.Clock()
        self._font = self._create_font()
        self._running = True
        self.clbks = []
        if not render_constraints:
            self._options.flags = DrawOptions.DRAW_SHAPES

    def attach_controller(self, controller: DirectController):
        self._controller = controller

    def attach_draw_clb(self, clb):
        self.clbks.append(clb)

    def one_time_draw(self, clb):
        # self._one_time_canvas.fill((255, 255, 255))
        clb(self._one_time_canvas, self._font)

    def _additional_drawings(self):
        for clb in self.clbks:
            clb(self.cur_scene, self._font)

    def _get_display(self, w, h):
        return pygame.display.set_mode((w, h))

    def _send_to_display(self):
        self.display.blit(self.cur_scene, (0, 0))
        pygame.display.update()
        if self._realtime:
            self._clock.tick(self._fps)
        if self._controller is not None:
            self._controller.update()
        else:
            self._mark_end_custom()

    def render(self, simulator: Simulator):
        self.cur_scene.blit(self._one_time_canvas, (0, 0))
        self._additional_drawings()
        simulator.draw_on(self._options)
        self._send_to_display()

    def _end(self):
        self._running = False
        pygame.quit()
        exit(0)

    def _mark_end_custom(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._end()
