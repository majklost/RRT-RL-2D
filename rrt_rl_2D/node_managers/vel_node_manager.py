from .node_manager import NodeManager
from ..nodes import *
import numpy as np


class VelNodeManager(NodeManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.velocity_buffer = []

    def after_step_clb(self, env):
        self.velocity_buffer.append(env.map.agent.velocity)

    def after_reset_clb(self, env):
        self.velocity_buffer = []

    def export(self, env):
        tn = VelTreeNode()
        state = env.map.sim.export()
        tn.agent_pos = env.map.agent.position
        tn.state = state
        tn.velocity_buffer = np.array(self.velocity_buffer)
        self.velocity_buffer = []
        return tn
