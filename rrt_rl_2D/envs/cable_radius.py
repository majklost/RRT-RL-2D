import gymnasium as gym
import pygame
import numpy as np
import pymunk

from .rrt_env import BaseEnv, ResetableEnv, ImportableEnv
from ..node_managers.node_manager import NodeManager
from ..samplers import *


class CableRadius(BaseEnv):
    def __init__(self, cur_map, scale_factor, node_factory, render_mode=None, renderer=None):
        super().__init__(cur_map, scale_factor, node_factory,
                         render_mode=render_mode, renderer=renderer)

        self.agent_len = len(self.map.agent.bodies)
        self._set_filter()
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.last_start = None
        self._reset()

    def step(self, action):
        action = self._process_action(action)
        for i in range(self.agent_len):
            force = action[i * 2: i * 2 + 2]
            if np.linalg.norm(force) > 1:
                force /= np.linalg.norm(force)
            force *= self.scale_factor
            self.map.agent.bodies[i].apply_force(force)

        return super().step(action)

    def _create_step_return(self):
        obs = self._get_observation()
        reward, done = self._get_reward()
        info = self._get_info()
        return obs, reward, done, False, info

    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 2,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.agent_len * 2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        vecs = self.goal.goal - self.map.agent.position
        if len(vecs.shape) == 1:  # Hack to be compatible with rectangle
            vecs = vecs.reshape(1, -1)
        return vecs

    def _get_observation(self):
        target_distance_vecs = self._get_target_distance_vecs()
        return target_distance_vecs.flatten()

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1), where=np.linalg.norm(distances, axis=1) > self.goal.threshold)

    def _get_reward(self):
        if self.map.agent.outer_collision_idxs:
            self.fail = True
            return 0, True

        distances = self._get_target_distance_vecs()

        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            self.reached = True
            return 1000, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential  # return zero
        reward = potential - self.last_target_potential

        self.last_target_potential = potential
        return reward, False

    def _get_info(self):
        return {'goal': self.goal, 'fail': self.fail, 'reached': self.reached}

    def _process_action(self, action):
        return action

    def _set_filter(self):
        for b in self.map.agent.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    def _render_goal(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.goal.goal, 10)
        pygame.draw.circle(screen, (0, 0, 255), self.goal.goal,
                           self.goal.threshold, 1)
        target_vecs = self._get_target_distance_vecs()
        for i in range(len(target_vecs)):
            pygame.draw.line(screen, (255, 0, 0), self.map.agent.position[i],
                             self.map.agent.position[i] + target_vecs[i], 1)

    def _additional_render(self, screen, font, **kwargs):
        self._render_goal(screen)

    def _reset(self):
        self.last_target_potential = 0
        self.fail = False
        self.reached = False


class CableRadiusI(ImportableEnv, CableRadius):
    """
    Standard Importable from CableRadius
    """

    pass


class CableRadiusR(ResetableEnv, CableRadius):
    """
    Standard Resetable from CableRadius, custom reset_goal
    """

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
