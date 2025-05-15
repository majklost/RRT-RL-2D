from pathlib import Path
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pygame
import pygame.freetype
import importlib.resources

if TYPE_CHECKING:
    from ..simulator.simulator import Simulator


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, simulator: 'Simulator'):
        """
        Renders the current state of the environment
        """
        raise NotImplementedError("Render method must be implemented")

    @staticmethod
    def _create_font():
        pygame.freetype.init()
        with importlib.resources.path("rrt_rl_2D", "Arial.ttf") as font_path:
            return pygame.freetype.Font(font_path, 20)
