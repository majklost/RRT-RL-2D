from .node import Node
import numpy as np


class GoalNode(Node):
    """
    Used to represent a goal (of local planner) in the tree.
    """

    def __init__(self, position: np.ndarray, threshold: float):
        super().__init__()
        self.goal = position
        self.threshold = threshold
