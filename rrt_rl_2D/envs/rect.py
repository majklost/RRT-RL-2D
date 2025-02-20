import gymnasium as gym
import pygame
import numpy as np
import pygame.freetype
import pymunk

from .rrt_env import BaseEnv, ResetableEnv, ImportableEnv
from ..nodes.node_factory import NodeFactory
from ..samplers import *


class RectEnv(BaseEnv):
    """
    Env with rectangle, 2 degrees of freedom
    """

    def __init__(self, cur_map, scale_factor, node_factory=NodeFactory(), render_mode=None):
        super().__init__(cur_map, scale_factor, node_factory, render_mode=render_mode)
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        self._reset()

    def step(self, action):
        if np.linalg.norm(action) > 1:
            action /= np.linalg.norm(action)
        action *= self.scale_factor
        self.map.agent.bodies[0].apply_force(action)

        self.map.sim.step()
        obs = self._get_observation()
        reward, done = self._get_reward()
        self.cur_return += reward
        info = self._get_info()
        return obs, reward, done, False, info

    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(2,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        return self.goal.goal - self.map.agent.position

    def _get_observation(self):
        target_distance_vecs = self._get_target_distance_vecs()
        return target_distance_vecs.flatten()

    def _calc_potential(self, distances):
        p = np.linalg.norm(distances)
        return -p if p > self.goal.threshold else 0

    def _get_reward(self):
        if self.map.agent.collision_data:
            self.fail = True
            return -1000, True

        distances = self._get_target_distance_vecs()

        if np.linalg.norm(distances) < self.goal.threshold:
            self.reached = True
            return 1000, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential
        reward = potential - self.last_target_potential
        self.last_target_potential = potential
        self.last_reward = reward
        return reward, False

    def _get_info(self):
        return {'goal': self.goal, 'fail': self.fail, 'reached': self.reached}

    def _additional_render(self, screen, font, **kwargs):
        super()._additional_render(screen, font, **kwargs)
        self._render_goal(screen)

    def _render_goal(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.goal.goal, 10)
        pygame.draw.circle(screen, (0, 0, 255), self.goal.goal,
                           self.goal.threshold, 1)
        target_vec = self._get_target_distance_vecs()
        pygame.draw.line(screen, (255, 0, 0), self.map.agent.position,
                         self.map.agent.position + target_vec, 1)

    def _reset(self):
        self.last_target_potential = 0
        self.fail = False
        self.reached = False
        self.last_reward = 0
        self.cur_return = 0


class RectEnvI(ImportableEnv, RectEnv):
    pass


class RectEnvR(ResetableEnv, RectEnv):
    pass
