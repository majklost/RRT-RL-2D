from typing import List

from ..nodes.tree_node import TreeNode
from ..nodes.goal_node import GoalNode
from ..envs.rrt_env import RRTEnv


class BasePlanner:
    def __init__(self, env: RRTEnv):
        """
        Initializes the planner with the environment.
        :param env: The environment in which the planner is working. MUST BE PREPARED WITH NORMS ALREADY
        """
        self.env = env

    def check_path(self, start: TreeNode, goal: GoalNode) -> 'PlannerResponse':
        """
        Checks if a path exists between the start and goal nodes.
        """
        raise NotImplementedError("Check path method must be implemented")


class PlannerResponse:
    def __init__(self, path: List[TreeNode], data: dict):
        """
        Initializes the response from the planner.
        :param path: The path found by the planner.
        :param data: The data associated with the path.
        """
        self.path = path
        self.data = data
