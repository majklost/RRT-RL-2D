import gymnasium as gym
from typing import TYPE_CHECKING
import warnings
import pygame

if TYPE_CHECKING:
    from ..nodes.tree_node import TreeNode
    from ..nodes.goal_node import GoalNode


from ..node_managers.node_manager import NodeManager
from ..maps.empty import Empty
from ..rendering.env_renderer import EnvRenderer
from abc import abstractmethod


class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human', None]}

    def __init__(self, cur_map: Empty, scale_factor, node_manager: NodeManager, render_mode=None, renderer=None):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = renderer
        self.scale_factor = scale_factor
        self.goal = None  # GoalNode
        self.start = None  # TreeNode
        self.node_manager = node_manager  # NodeFactory
        # For warning
        self.reset_called = False
        self.import_called = False
        # For rendering
        self.last_reward = 0
        self.cur_return = 0

        self.map = cur_map
        if render_mode is not None:
            if renderer is None:
                self.set_renderer(EnvRenderer(self.map.cfg))
            else:
                self.set_renderer(renderer)

    def set_renderer(self, renderer: EnvRenderer):
        self.renderer = renderer
        self.renderer.register_callback(self._additional_render)

    def reset(self, seed=None, options=None):
        self._reset()

        return self._get_observation(), self._get_info()

    def _reset(self):
        """
        Place for reseting of inner parts of the environment.
        """
        pass

    def step(self, action):
        self.map.sim.step()
        self.node_manager.after_step_clb(self)

        return self._create_step_return()

    def _create_step_return(self):
        raise NotImplementedError(
            "Create step return method must be implemented")

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
        return self.node_manager.export(self)

    def _additional_render(self, screen, font, **kwargs):
        self._render_return_reward(screen, font)

    def _render_return_reward(self, screen: pygame.Surface, font: pygame.freetype.Font):
        font.render_to(screen, (50, 50),
                       f"Reward: {self.last_reward}")
        font.render_to(screen, (50, 100),
                       f"ReturN: {self.cur_return}", (0, 255, 0))

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

        self.start = start
        self.map.sim.import_from(start.state)
        self.node_manager.after_reset_clb(self)
        return self._get_observation(), self._get_info()

    def import_goal(self, goal: 'GoalNode') -> None:
        """
        Imports the goal node into the environment.
        """
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

        pos = self._reset_position()
        self.map.agent.position = pos
        self.start = self.export_state()

    def reset_goal(self):
        """
        Resets the goal node.
        """

        pos = self._reset_position()
        if self.goal is None:
            self.node_manager.wanted_position = pos
            self.goal = self.node_manager.create_goal()
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
