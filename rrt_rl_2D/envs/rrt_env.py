import gymnasium as gym


from ..nodes.tree_node import TreeNode
from ..nodes.goal_node import GoalNode
from ..maps.empty import Empty
from ..rendering.env_renderer import EnvRenderer
from abc import abstractmethod


class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human', None]}

    def __init__(self, cur_map: Empty, scale_factor, render_mode=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.goal = None  # GoalNode
        self.start = None  # TreeNode

        self.map = cur_map
        if render_mode is not None:
            self.renderer = EnvRenderer(self.map.cfg)

    def render(self):
        if self.renderer is not None:
            self.renderer.render(self.map.sim)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class RRTEnv(BaseEnv):

    def import_start(self, start: TreeNode) -> None:
        """
        Imports the start node into the environment.
        """
        self.map.sim.import_from(start.state)

    def import_goal(self, goal: GoalNode) -> None:
        """
        Imports the goal node into the environment.
        """
        self.goal = goal

    def export_state(self) -> TreeNode:
        """
        Exports the state of the environment.
        """

        tn = TreeNode()
        state = self.map.sim.export()
        tn.state = state
        return tn


class ResetableEnv(BaseEnv):
    def reset_start(self):
        """
        Resets the start node.
        """

    def reset_goal(self):
        """
        Resets the goal node.
        """
        
    def _reset_position(self):
        valid = False
        while not valid:
            pos = self.map.sampler.sample()
            if len(pos.shape) == 1:
                valid = self.map.check_validity(
                    pos.reshape(1, -1))
            else:
                valid = self.map.check_validity(pos)
        return pos
