from .node_manager import NodeManager
from ..nodes import *


class VelNodeManager(NodeManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.velocity_buffer = []

    def after_step_clb(self, env):
        self.velocity_buffer.append(env.map.agent.velocity)

    def export(self, env):
        tn = VelTreeNode()
        state = env.map.sim.export()
        tn.agent_pos = env.map.agent.position
        tn.state = state
        tn.velocity_buffer = self.velocity_buffer
        return tn
