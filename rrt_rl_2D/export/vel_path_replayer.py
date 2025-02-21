import pygame
from ..maps.empty import Empty
from ..storage_wrappers.base_wrapper import BasePath
from ..rendering.debug_renderer import DebugRenderer
from ..nodes import VelTreeNode
import time


class VelPathReplayer:
    def __init__(self, cur_map: Empty, path: BasePath, ):
        self.map = cur_map
        self.path = path
        self.renderer = DebugRenderer(self.map.cfg)

    def replay(self):
        self.map.sim.attach_renderer(self.renderer)
        self.renderer.attach_draw_clb(self.draw)
        # self.map.sim.import_from(self.path.nodes[0].state)
        # print(self.map.agent.position)
        for i in range(0, len(self.path.nodes)):
            node = self.path.nodes[i]
            assert isinstance(node, VelTreeNode), "Node is not VelTreeNode"
            # print(f"node {i} buffer:{len(node.velocity_buffer)}")
            for vel in node.velocity_buffer:
                self.map.sim.step()
                self.map.agent.velocity = vel

            self.map.sim.import_from(node.state)
            self.map.sim.step()

        self.map.agent.velocity = [0, 0]

    def draw(self, screen, font):
        for node in self.path.nodes:
            pygame.draw.circle(screen, (0, 255, 0), node.agent_pos, 5)
        for i in range(1, len(self.path.nodes)):
            node = self.path.nodes[i]
            pygame.draw.line(
                screen, (0, 255, 0), self.path.nodes[i - 1].agent_pos, node.agent_pos, 2)
