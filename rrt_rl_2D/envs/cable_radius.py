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
        self.last_action = None
        self.last_reward = 0
        self.cur_return = 0
        self._reset()

    def step(self, action):
        action = self._process_action(action)
        self.last_action = action
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
        self.last_reward = reward
        self.cur_return += reward
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
            return -1000, True

        distances = self._get_target_distance_vecs()

        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            self.reached = True
            return 100000, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential  # return zero
        reward = 10 * (potential - self.last_target_potential) - 20

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

    def _render_return(self, screen, font):
        font.render_to(screen, (50, 50),
                       f"Reward: {self.last_reward}")
        font.render_to(screen, (50, 150),
                       f"Return: {self.cur_return}")

    def _render_forces(self, screen):
        for i in range(self.agent_len):
            force = self.last_action[i * 2: i * 2 + 2]
            pygame.draw.line(screen, (255, 0, 0), self.map.agent.position[i],
                             self.map.agent.position[i] + force // 2, 2)

    def _additional_render(self, screen, font, **kwargs):
        self._render_goal(screen)
        self._render_forces(screen)
        self._render_return(screen, font)

    def _reset(self):
        self.last_target_potential = 0
        self.fail = False
        self.reached = False
        self.last_reward = 0
        self.cur_return = 0


class CableRadiusNearestObs(CableRadius):
    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 4,), dtype=np.float64)

    def _get_obstacle_distance_vecs(self):
        responses = np.array([self.map.sim._space.point_query_nearest(
            x.tolist(), (self.map.cfg['height']**2 + self.map.cfg['width']**2)**0.5, self.my_filter).point for x in self.map.agent.position])
        return responses - self.map.agent.position

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
