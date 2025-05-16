from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """
    Storage for the nodes of the tree.
    """

    def __init__(self, distancefnc):
        self.distancefnc = distancefnc

    @abstractmethod
    def insert(self, point):
        raise NotImplementedError

    @abstractmethod
    def nearest_neighbour(self, point):
        raise NotImplementedError

    @abstractmethod
    def get_all_nodes(self):
        raise NotImplementedError
