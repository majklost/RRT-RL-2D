from .node import Node
import numpy as np


class GoalNode(Node):
    """
    Used to represent a goal (of local planner) in the tree.
    """

    def __init__(self, position: np.ndarray, threshold: float, controllable_idxs: list = None):
        super().__init__()
        self.goal = position
        self.threshold = threshold
        # Controllable indices : List[int]
        self.controllable_idxs = controllable_idxs
