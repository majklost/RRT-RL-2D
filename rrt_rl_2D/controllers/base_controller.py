from abc import ABC, abstractmethod


class BaseController(ABC):

    @abstractmethod
    def update(self):
        """
        Performs one step of the controller
        """
        raise NotImplementedError("Update method must be implemented")
