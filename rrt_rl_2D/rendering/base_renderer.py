from abc import ABC, abstractmethod


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, simulator):
        """
        Renders the current state of the environment
        """
        raise NotImplementedError("Render method must be implemented")
