from abc import ABC, abstractmethod
from typing import List
from ..planners.base_planner import PlannerResponse
from ..nodes.goal_node import GoalNode
from ..nodes.tree_node import TreeNode


class BaseWrapper:
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def save_to_storage(self, response: PlannerResponse):
        pass

    @abstractmethod
    def get_nearest(self, point: GoalNode) -> TreeNode:
        pass

    @abstractmethod
    def get_path(self) -> 'BasePath':
        pass

    @abstractmethod
    def get_all_nodes(self) -> List[TreeNode]:
        pass


class BasePath:
    def __init__(self, nodes: List[TreeNode], data: dict):
        self.nodes = nodes
        self.data = data
