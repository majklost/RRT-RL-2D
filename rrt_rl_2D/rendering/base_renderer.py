from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulator.simulator import Simulator


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, simulator: 'Simulator'):
        """
        Renders the current state of the environment
        """
        raise NotImplementedError("Render method must be implemented")
