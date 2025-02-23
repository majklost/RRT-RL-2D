# Trying to copy cable radius from previous package
import gymnasium as gym
import pygame
import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np

from ..simulator.standard_config import STANDARD_CONFIG
from ..samplers.ndim_sampler import NDIMSampler
from ..maps import *
from ..rendering.env_renderer import EnvRenderer


class CableRadiusEmpty(gym.Env):
    metadata = {'render.modes': ['human', None], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()
        pygame.init()
        cfg = STANDARD_CONFIG.copy()
        cfg['seg_num'] = 10
        self.cur_map = AlmostEmpty(cfg)
        self.render_mode = render_mode
        self.renderer = None

        self.controllable_idxs = list(range(len(self.cur_map.agent.bodies)))
        self.controllable_num = len(self.controllable_idxs)
        self._set_filter()
        self.width = self.cur_map.cfg['width']
        self.height = self.cur_map.cfg['height']

        self.radius = self.cur_map.agent.length / 2

        self.goal_sampler = NDIMSampler([self.radius, self.radius], [
            self.width - self.radius, self.height - self.radius])

        self.goal = None
        self.reset_start()
        self._reset_goal()

        self.scale_factor = 600
        self.last_target_potential = 0
        self.last_info = None
        self.success = False

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

    def reset_start(self):
        """
        Resets the start node.
        """

        pos = self._reset_position()
        self.cur_map.agent.position = pos

    def _reset_position(self):
        valid = False
        while not valid:
            pos = self.cur_map.sampler.sample()
            if len(pos.shape) == 1:
                valid = self.cur_map.check_validity(
                    pos.reshape(1, -1))
            else:
                valid = self.cur_map.check_validity(pos)
        return pos

    def _reset_goal(self):
        """
        Creates 2D goal point
        """
        self.goal = self.goal_sampler.sample()

    def _set_filter(self):
        for b in self.cur_map.agent.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 6,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        return self.goal - self.cur_map.agent.position

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        velocities = self.cur_map.agent.velocity
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten(), velocities.flatten()))

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1), where=np.linalg.norm(distances, axis=1) > self.radius)

    def _process_action(self, action):
        return action

    def step(self, action):
        action = self._process_action(action)
        self.last_action = action
        for i in range(len(self.controllable_idxs)):
            idx = self.controllable_idxs[i]
            # print(i, len(action), i*2+2)
            force = action[i * 2:i * 2 + 2]
            if np.linalg.norm(force) > 1:
                force /= np.linalg.norm(force)
            force *= self.scale_factor
            self.cur_map.agent.bodies[idx].apply_force(force)
        self.cur_map.sim.step()
        obs = self._get_observation()
        reward, done = self._get_reward()
        self.cur_return += reward
        self.last_reward = reward

        self.last_info = self._get_info()
        return obs, reward, done, False, self.last_info

    def _get_reward(self):
        if self.cur_map.agent.outer_collision_idxs:
            return -1000, True

        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.radius):
            self.success = True
            return 100000, True

        potential = self._calc_potential(distances)
        reward = 10 * (potential - self.last_target_potential) - 20
        self.last_target_potential = potential
        return reward, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.success = False
        self.last_reward = None
        self.cur_return = 0
        self.reset_start()
        # self.map.cable.position = self.start_pos
        self._reset_goal()

        self.last_target_potential = self._calc_potential(
            self._get_target_distance_vecs())

        self.last_info = self._get_info()
        return self._get_observation(), self.last_info

    def _get_info(self):
        return {'goal': self.goal, 'success': self.success}

    def render(self):
        if self.renderer is None:
            self.renderer = EnvRenderer(self.cur_map.cfg)
        self.renderer.render(self.cur_map.sim)

    def close(self):
        self.renderer.close()
    

    def _get_obstacle_distance_vecs(self):
        responses = np.array([self.cur_map.sim._space.point_query_nearest(
            x.tolist(), (self.width**2 + self.height**2)**0.5, self.my_filter).point for x in self.cur_map.agent.position])
        return responses - self.cur_map.agent.position
