import numpy as np
import pygame
from typing import Tuple, Optional

from abc import ABC, abstractmethod


class EnvController(ABC):
    """
    Controls the environemnt and agent via actions
    Used for controlling agent by keyboard in environment
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, obs) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Returns 
        """
        raise NotImplementedError("Method predict not implemented")

    @staticmethod
    def _process_keys():
        force_vector = np.zeros(2)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            force_vector[0] = -1
        if keys[pygame.K_RIGHT]:
            force_vector[0] = 1
        if keys[pygame.K_UP]:
            force_vector[1] = -1
        if keys[pygame.K_DOWN]:
            force_vector[1] = 1
        return force_vector

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)


class CableEnvController(EnvController):
    """
    Standard implementation for environments with cable
    """

    def __init__(self, segnum: int):
        super().__init__()
        self.segnum = segnum
        self.current = 0

    def predict(self, obs):
        force_vector = self._process_keys()
        self._handle_events()
        template = np.zeros((self.segnum, 2))
        template[self.current] = force_vector
        return template.flatten().reshape(1, -1), None

    def _set_current(self, index):
        self.current = index

    def _cur_next(self):
        new_idx = (self.current + 1) % self.segnum
        self._set_current(new_idx)

    def _cur_prev(self):
        new_idx = (self.current - 1) % self.segnum
        self._set_current(new_idx)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB and event.mod & pygame.KMOD_SHIFT:
                    self._cur_prev()
                elif event.key == pygame.K_TAB:
                    self._cur_next()


class RectEnvController(EnvController):
    def predict(self, obs):
        force_vector = self._process_keys()
        self._handle_events()
        return force_vector.flatten().reshape(1, -1), None
