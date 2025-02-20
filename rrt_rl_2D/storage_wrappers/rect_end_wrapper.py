import numpy as np
import pygame
from .standard_wrapper import StandardWrapper
from ..nodes import TreeNode


class RectEndWrapper(StandardWrapper):
    def _check_reached_goal(self, node: TreeNode):
        pos = node.agent_pos
        if len(pos.shape) == 1:
            pos = pos.reshape(1, -1)
        diff = pos[:, 0] - self.overall_goal.goal[0]
        if np.abs(diff) < self.overall_goal.threshold:
            self.want_next_iter = False
            self._end_node = node
            return True

    def render_clb(self, screen, font):
        pygame.draw.rect(screen, (0, 100, 255), (self.overall_goal.goal[0] - self.overall_goal.threshold,
                         # width = 3
                                                 0, 2 * self.overall_goal.threshold, self.cfg['height']), 5)
