from .goal_node import GoalNode
from .tree_node import TreeNode


class NodeFactory:
    def __init__(self):
        self.wanted_position = None
        self.wanted_threshold = 30

    def create_goal(self) -> GoalNode:
        assert self.wanted_position is not None, "Before creating goal give position"
        assert self.wanted_threshold is not None, "Before creating goal give threshold"

        return GoalNode(self.wanted_position, self.wanted_threshold)

    def create_tree_node(self) -> TreeNode:
        return TreeNode()
