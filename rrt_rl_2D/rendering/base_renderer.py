from pathlib import Path
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pygame

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
        pygame.font.init()
        font_path = Path(__file__).parent / 'Arial.ttf'
        return pygame.freetype.Font(font_path, 20)
