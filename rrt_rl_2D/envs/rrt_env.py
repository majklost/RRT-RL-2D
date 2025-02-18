import gymnasium as gym
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from ..nodes.tree_node import TreeNode
    from ..nodes.goal_node import GoalNode

from ..nodes.node_factory import NodeFactory
from ..maps.empty import Empty
from ..rendering.env_renderer import EnvRenderer
from abc import abstractmethod


class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human', None]}

    def __init__(self, cur_map: Empty, scale_factor, node_factory: NodeFactory, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.scale_factor = scale_factor
        self.goal = None  # GoalNode
        self.start = None  # TreeNode
        self.node_factory = node_factory  # NodeFactory
        # For warning
        self.reset_called = False
        self.import_called = False

        self.map = cur_map
        if render_mode is not None:
            self.renderer = EnvRenderer(self.map.cfg)
            self.renderer.register_callback(self._additional_render)

    def reset(self, seed=None, options=None):
        self._reset()
        return self._get_observation(), self._get_info()

    def _reset(self):
        pass

    def render(self):
        if self.renderer is not None:
            self.renderer.render(self.map.sim)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def export_state(self) -> 'TreeNode':
        """
        Exports the state of the environment.
        """
        tn = self.node_factory.create_tree_node()
        state = self.map.sim.export()
        tn.state = state
        return tn

    def _on_start_g_change(self):
        """
        Callback that is done everytime a goal or start is changed
        """
        pass

    def _additional_render(self, screen, font, **kwargs):
        pass

    def _get_observation(self):
        """
        Returns the observation of the environment.
        """
        raise NotImplementedError("Get observation method must be implemented")

    def _get_info(self):
        """
        Returns the info of the environment.
        """
        raise NotImplementedError("Get info method must be implemented")


class ImportableEnv(BaseEnv):

    def import_start(self, start: 'TreeNode'):
        """
        Imports the start node into the environment.
        """
        assert not self.reset_called, "Environment should either be reset or import start and goal, not both"

        self.import_called = True

        self._on_start_g_change()
        self.start = start
        self.map.sim.import_from(start.state)
        return self._get_observation(), self._get_info()

    def import_goal(self, goal: 'GoalNode') -> None:
        """
        Imports the goal node into the environment.
        """
        self._on_start_g_change()
        self.goal = goal

    def reset(self, seed=None, options=None):
        return super().reset(seed, options)


class ResetableEnv(BaseEnv):
    """
    Enabling reset of start and end in environment
    """

    def reset_start(self):
        """
        Resets the start node.
        """

        assert not self.import_called, "Environment should either be reset or import start and goal, not both"

        self.reset_called = True
        self._on_start_g_change()
        pos = self._reset_position()
        self.map.agent.position = pos
        self.start = self.export_state()

    def reset_goal(self):
        """
        Resets the goal node.
        """
        self._on_start_g_change()
        pos = self._reset_position()
        if self.goal is None:
            self.node_factory.wanted_position = pos
            self.goal = self.node_factory.create_goal()
        self.goal.goal = pos

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

    def reset(self, seed=None, options=None):
        self.reset_start()
        self.reset_goal()
        return super().reset(seed, options)
