import gymnasium as gym
import numpy as np
import pymunk
import warnings

from .rrt_env import BaseEnv, ResetableEnv, RRTEnv
from ..nodes import *


class CableRadius(RRTEnv, ResetableEnv):
    def __init__(self, cur_map, scale_factor, node_factory=NodeFactory(), render_mode=None):
        super().__init__(cur_map, scale_factor, node_factory, render_mode=render_mode)

        self.agent_len = len(self.map.agent.bodies)
        self._set_filter()
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.last_start = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_target_potential = 0
        self.reset_start()
        self.reset_goal()
        return self._get_observation(), self._get_info()

    def step(self, action):
        action = self._process_action(action)
        for i in range(self.agent_len):
            force = action[i * 2: i * 2 + 2]
            if np.linalg.norm(force) > 1:
                force /= np.linalg.norm(force)
            force *= self.scale_factor
            self.map.agent.bodies[i].apply_force(force)
        self.map.sim.step()
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
        return self.goal.goal - self.map.agent.position

    def _get_observation(self):
        target_distance_vecs = self._get_target_distance_vecs()
        return target_distance_vecs.flatten()

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1), where=np.linalg.norm(distances, axis=1) > self.goal.threshold)

    def _get_reward(self):
        distances = self._get_target_distance_vecs()
        if len(distances.shape) == 1:
            distances = distances.reshape(1, -1)

        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            return 1000, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential  # return zero
        reward = potential - self.last_target_potential

        self.last_target_potential = potential
        return reward, False

    def _get_info(self):
        return {'goal': self.goal}

    def _process_action(self, action):
        return action

    def _set_filter(self):
        for b in self.map.agent.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    def _on_start_g_change(self):
        self.last_target_potential = 0
