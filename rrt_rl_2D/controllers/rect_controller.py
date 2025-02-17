import pygame
import numpy as np
from pymunk.pygame_util import from_pygame

from .direct_controller import DirectController
from ..assets import *


class RectController(DirectController):
    def __init__(self, rect: Rectangle, moving_force=2000):
        self.rect = rect
        self.moving_force = moving_force

    def update(self):
        force_template = np.zeros(2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            force_template[0] = -self.moving_force
        if keys[pygame.K_RIGHT]:
            force_template[0] = self.moving_force
        if keys[pygame.K_UP]:
            force_template[1] = -self.moving_force
        if keys[pygame.K_DOWN]:
            force_template[1] = self.moving_force
        if keys[pygame.K_r]:
            self.rect.orientation += 0.1
        if keys[pygame.K_e]:
            self.rect.orientation -= 0.1

        self.rect.apply_force(force_template)
