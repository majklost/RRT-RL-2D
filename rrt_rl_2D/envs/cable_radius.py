import gymnasium as gym
import pygame
import numpy as np
import pymunk

from .rrt_env import BaseEnv, ResetableEnv, ImportableEnv
from ..node_managers.node_manager import NodeManager
from ..samplers import *
from .cable_env import CableEnv


class CableRadius(CableEnv):

    def _render_goal(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.goal.goal, 10)
        pygame.draw.circle(screen, (0, 0, 255), self.goal.goal,
                           self.goal.threshold, 1)
        target_vecs = self._get_target_distance_vecs()
        for i in range(len(target_vecs)):
            pygame.draw.line(screen, (255, 0, 0), self.map.agent.position[i],
                             self.map.agent.position[i] + target_vecs[i], 1)


class CableRadiusNearestObs(CableRadius):
    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 4,), dtype=np.float64)


    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten()))

    def _get_reward(self):
        return super()._get_reward()

    def _additional_render(self, screen, font, **kwargs):
        self._render_obstacles(screen)
        super()._additional_render(screen, font, **kwargs)

    def _render_obstacles(self, screen):
        for i in range(self.agent_len):
            pygame.draw.line(screen, (0, 255, 0), self.map.agent.position[i],
                             self.map.agent.position[i] + self._get_obstacle_distance_vecs()[i], 3)


class CableRadiusNearestObsVel(CableRadiusNearestObs):
    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 6,), dtype=np.float64)

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        velocities = self.map.agent.velocity
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten(), velocities.flatten()))


# Shortcuts so the creation of the environments is easier
class CustomResetableEnv(ResetableEnv, CableRadius):

    def __init__(self, cur_map, scale_factor, node_factory, render_mode=None, renderer=None):
        super().__init__(cur_map, scale_factor, node_factory,
                         render_mode=render_mode, renderer=renderer)

        self.custom_sampler = NDIMSampler((self.map.MARGIN, self.map.MARGIN), (
            self.map.cfg['width'] - self.map.MARGIN, self.map.cfg["height"] - self.map.MARGIN))

    def reset_goal(self):
        pos = self.custom_sampler.sample()
        if self.goal is None:
            self.node_manager.wanted_position = pos
            self.goal = self.node_manager.create_goal()
        self.goal.goal = pos


class CableRadiusI(ImportableEnv, CableRadius):
    """
    Standard Importable from CableRadius
    """

    pass


class CableRadiusR(CustomResetableEnv, CableRadius):
    """
    Standard Resetable from CableRadius, custom reset_goal
    """


class CableRadiusNearestObsI(ImportableEnv, CableRadiusNearestObs):
    """
    Standard Importable from CableRadiusNearestObs
    """

    pass


class CableRadiusNearestObsVelI(ImportableEnv, CableRadiusNearestObsVel):
    """
    Standard Importable from CableRadiusNearestObsVel
    """

    pass


class CableRadiusNearestObsR(CustomResetableEnv, CableRadiusNearestObs):
    """
    Standard Resetable from CableRadiusNearestObs, custom reset_goal
    """

    pass


class CableRadiusNearestObsVelR(CustomResetableEnv, CableRadiusNearestObsVel):
    """
    Standard Resetable from CableRadiusNearestObsVel, custom reset_goal
    """

    pass
