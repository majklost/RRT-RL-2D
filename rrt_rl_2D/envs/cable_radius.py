import gymnasium as gym
import numpy as np
import pymunk
import warnings

from .rrt_env import BaseEnv
from ..nodes import *


class CableRadius(BaseEnv):
    def __init__(self, cur_map, scale_factor, render_mode=None):
        super().__init__(cur_map, scale_factor, render_mode=render_mode)

        self.agent_len = len(self.map.agent.bodies)
        self._set_filter()
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.last_start = None
        self.reset()

    def export_state(self):
        state = self.map.sim.export()
        return TreeNode(state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_target_potential = 0
        try:
            self.map.reset_start()
            self.map.reset_goal()
        except NotImplementedError:
            warnings.warn("Reset start method not implemented on map")
            if self.last_start is not None:
                self.import_start(self.last_start)

    def import_start(self, start: TreeNode):
        self.start = start
        self.map.sim.import_from(start.state)

    def import_goal(self, goal: Goal2D):
        self.goal = goal

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
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        return self.cur_goal.goal - self.map.cable.position

    def _get_observation(self):
        target_distance_vecs = self._get_target_distance_vecs()
        return target_distance_vecs.flatten()

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1), where=np.linalg.norm(distances, axis=1) > self.cur_goal.threshold)

    def _get_reward(self):
        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.radius):
            self.success = True
            return 1000, True

        potential = self._calc_potential(distances)
        reward = potential - self.last_target_potential
        self.last_target_potential = potential
        return reward, False

    def _get_info(self):
        return {'goal': self.goal, 'last_action': self.last_action, 'success': self.success}

    def _process_action(self, action):
        return action

    def _set_filter(self):
        for b in self.map.agent.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
