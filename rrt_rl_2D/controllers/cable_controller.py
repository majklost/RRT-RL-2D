import pygame
import numpy as np
from pymunk.pygame_util import from_pygame

from .direct_controller import DirectController
from ..assets import *


class MultiBodyController(DirectController):
    """
    Control a MutliBody object with keyboard input.
    """


    def __init__(self, cable: Cable, moving_force=300, color_change=True):
        self.cable = cable
        self.current = 0
        self.moving_force = moving_force
        self.color_change = color_change
        self.esc_pressed = False

    def _cur_next(self):
        new_idx = (self.current + 1) % len(self.cable.bodies)
        self._set_current(new_idx)

    def _cur_prev(self):
        new_idx = (self.current - 1) % len(self.cable.bodies)
        print(new_idx)
        self._set_current(new_idx)

    def _set_current(self, index):
        if self.color_change:
            self.cable.bodies[self.current].color = pygame.Color("blue")

        self.current = index
        if self.color_change:
            self.cable.bodies[self.current].color = pygame.Color("yellow")

    def update(self):
        force_template = np.zeros(2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB and event.mod & pygame.KMOD_SHIFT:
                    print("shift tab")
                    self._cur_prev()
                elif event.key == pygame.K_TAB:
                    self._cur_next()

            if event.type == pygame.MOUSEBUTTONDOWN:
                p = from_pygame(event.pos, pygame.display.get_surface())
                for i, obj in enumerate(self.cable.bodies):
                    for s in obj.shapes:
                        dist = s.point_query(p).distance
                        if dist < 0:
                            self._set_current(i)
                            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            self.esc_pressed = True
        if keys[pygame.K_LEFT]:
            force_template[0] = -self.moving_force
        if keys[pygame.K_RIGHT]:
            force_template[0] = self.moving_force
        if keys[pygame.K_UP]:
            force_template[1] = -self.moving_force
        if keys[pygame.K_DOWN]:
            force_template[1] = self.moving_force

        self.cable.bodies[self.current].apply_force(force_template)
