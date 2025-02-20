from typing import TYPE_CHECKING

from ..nodes.goal_node import GoalNode
from ..nodes.tree_node import TreeNode
if TYPE_CHECKING:
    from ..envs.rrt_env import BaseEnv
    from ..simulator.standard_config import StandardConfig


class NodeManager:
    def __init__(self, cfg: 'StandardConfig'):
        self.wanted_position = None
        self.wanted_threshold = cfg['threshold']

    def create_goal(self) -> GoalNode:
        assert self.wanted_position is not None, "Before creating goal give position"
        assert self.wanted_threshold is not None, "Before creating goal give threshold"

        return GoalNode(self.wanted_position, self.wanted_threshold)

    def after_step_clb(self, env: 'BaseEnv'):
        pass

    def export(self, env: 'BaseEnv') -> TreeNode:
        tn = TreeNode()
        state = env.map.sim.export()
        tn.agent_pos = env.map.agent.position
        tn.state = state
        return tn
