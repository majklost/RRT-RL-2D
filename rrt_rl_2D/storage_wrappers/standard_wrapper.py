from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ..nodes import *
from .base_wrapper import BaseWrapper, BasePath
from ..storages import BaseStorage


class StandardWrapper(BaseWrapper):
    """
    NDim start 2D goal storage
    """

    def __init__(self, storage: BaseStorage, distance_fn, overall_goal: 'GoalNode', cfg):
        super().__init__(cfg=cfg)
        self.storage = storage  # type: BaseStorage
        # must know to be able to check whether it is near the goal
        self.distance_fn = distance_fn
        self.overall_goal = overall_goal
        self.want_next_iter = True
        self._end_node = None
        self.best_dist = float('inf')

    def save_to_storage(self, response):
        """
        Saves the response to the storage.
        """
        for node in response.path:
            self.storage.insert(node)
            if self._check_reached_goal(node):
                break

        return self.want_next_iter

    def _check_reached_goal(self, node):
        """
        Checks if the node is near the goal.
        """
        dist = self.distance_fn(node, self.overall_goal)
        if dist < self.overall_goal.threshold:
            self.want_next_iter = False
            self._end_node = node
            return True

        if dist < self.best_dist:
            self.best_dist = dist
            self._end_node = node
        return False

    def get_all_nodes(self) -> List['VelTreeNode']:
        return self.storage.get_all_nodes()

    def get_path(self) -> BasePath:
        path = []
        if self._end_node is None:
            return BasePath(path, {})
        cur = self._end_node
        while True:
            path.append(cur)
            if cur.parent is None:
                break
            cur = cur.parent

        return BasePath(path, {})

    def get_nearest(self, point: 'GoalNode') -> 'VelTreeNode':
        return self.storage.nearest_neighbour(point)
