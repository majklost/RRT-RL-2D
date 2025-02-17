from abc import ABC, abstractmethod


class DirectController(ABC):
    """
    Directly controls the simulation
    """

    @abstractmethod
    def update(self):
        """
        Performs one step of the controller
        """
        raise NotImplementedError("Update method must be implemented")
