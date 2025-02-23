from .vel_node_manager import VelNodeManager
from ..nodes.goal_node import GoalNode


class ControllableManager(VelNodeManager):
    def __init__(self, cfg, ctrl_idxs=None):
        self.ctrl_idxs = ctrl_idxs
        super().__init__(cfg)

    def create_goal(self) -> GoalNode:
        assert self.wanted_position is not None, "Before creating goal give position"
        assert self.wanted_threshold is not None, "Before creating goal give threshold"
        return GoalNode(self.wanted_position, self.wanted_threshold, controllable_idxs=self.ctrl_idxs)
