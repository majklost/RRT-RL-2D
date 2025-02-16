from ..nodes.tree_node import TreeNode
from ..nodes.goal_node import GoalNode


class BasePlanner:
    def check_path(self, start: TreeNode, goal: GoalNode):
        """
        Checks if a path exists between the start and goal nodes.
        """
        raise NotImplementedError("Check path method must be implemented")
