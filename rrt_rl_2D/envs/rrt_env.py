import gymnasium as gym
from ..nodes.tree_node import TreeNode
from ..nodes.goal_node import GoalNode


class RRTEnv(gym.Env):
    def import_start(start: TreeNode):
        """
        Imports the start node into the environment.
        """
        raise NotImplementedError("Import start method must be implemented")

    def import_goal(goal: GoalNode):
        """
        Imports the goal node into the environment.
        """
        raise NotImplementedError("Import goal method must be implemented")

    def export_state() -> TreeNode:
        """
        Exports the state of the environment.
        """
        raise NotImplementedError("Export state method must be implemented")
