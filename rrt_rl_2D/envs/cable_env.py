import gymnasium as gym
import pygame
import numpy as np
import pymunk
from .rrt_env import BaseEnv, ResetableEnv, ImportableEnv
from ..samplers import *


class CableEnv(BaseEnv):
    """
    Base env
    """

    def __init__(self, cur_map, scale_factor, node_factory, render_mode=None):
        super().__init__(cur_map, scale_factor, node_factory,
                         render_mode=render_mode)

        self.agent_len = len(self.map.agent.bodies)
        self._set_filter()
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.last_start = None
        self.last_actions = None
        self.last_reward = 0
        self.cur_return = 0
        self._reset()

    def step(self, action):
        self.last_actions = np.zeros((self.agent_len, 2))
        if self.goal.controllable_idxs is not None:
            idxs = self.goal.controllable_idxs
        else:
            idxs = range(self.agent_len)

        for i in idxs:
            cur_action = action[i * 2: i * 2 + 2]
            cur_action = self._process_action(cur_action, i)
            self.last_actions[i, :] = cur_action
            self.map.agent.bodies[i].apply_force(cur_action)

        self.last_actions = np.array(self.last_actions)
        return BaseEnv.step(self, action)

    def _process_action(self, cur_action, idx):
        if np.linalg.norm(cur_action) > 1:
            cur_action /= np.linalg.norm(cur_action)
        cur_action *= self.scale_factor

        return cur_action

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

        if self.goal.controllable_idxs is not None:
            idxs = self.goal.controllable_idxs
            distances = distances[idxs]

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

    def _set_filter(self):
        for b in self.map.agent.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    def _render_goal(self, screen):
        for i in range(len(self.goal.goal)):
            pygame.draw.circle(screen, (0, 0, 255), self.goal.goal[i], 2)
            if i != 0:
                pygame.draw.line(screen, (0, 0, 255),
                                 self.goal.goal[i - 1], self.goal.goal[i], 1)

    def _render_return(self, screen, font):
        font.render_to(screen, (50, 50),
                       f"Reward: {self.last_reward}")
        font.render_to(screen, (50, 150),
                       f"Return: {self.cur_return}")

    def _render_forces(self, screen):
        if self.goal.controllable_idxs is not None:
            idxs = self.goal.controllable_idxs
        else:
            idxs = range(self.agent_len)

        for i in idxs:
            force = self.last_actions[i]
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

    def _get_obstacle_distance_vecs(self):
        responses = np.array([self.map.sim._space.point_query_nearest(
            x.tolist(), (self.map.cfg['height']**2 + self.map.cfg['width']**2)**0.5, self.my_filter).point for x in self.map.agent.position])
        return responses - self.map.agent.position


class CableEnvNaive(CableEnv):
    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 4,), dtype=np.float64)

    def _get_observation(self):
        return np.concatenate([self.map.agent.position.flatten(), self.goal.goal.flatten()])


class CablePIDEnv(CableEnv):
    """
    Takes the action as the desired velocity of the agent.
    """

    def _process_action(self, cur_action, idx):
        if np.linalg.norm(cur_action) > 1:
            cur_action /= np.linalg.norm(cur_action)
        cur_action *= self.scale_factor

        vel = self.map.agent.bodies[idx].velocity
        error = cur_action - vel
        cur_action = 80 * error
        return cur_action


class CableInnerAngles(CableEnv):
    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        part1 = np.ones(self.agent_len * 2) * limit
        part2 = np.ones(self.agent_len - 1) * np.pi
        low = -np.concatenate((part1, part2))
        high = np.concatenate((part1, part2))
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def _get_observation(self):
        vecs = self._get_target_distance_vecs()
        angles = self.map.agent.angles_between()
        return np.concatenate((vecs.flatten(), angles))


class CableBigTest(CableEnv):
    def _create_observation_space(self):
        # Calculate appropriate dimensions and limits
        # [positions, segment vectors, sin/cos angles, goal vectors, obstacle distances]
        total_dims = 4 * 2 * self.agent_len - 2
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_dims,), dtype=np.float64)

    def _get_observation(self):

        # Segment vectors
        segment_vecs_diffs = np.roll(
            self.map.agent.position, -1, axis=0) - self.map.agent.position

        # Angles with sin/cos representation
        angles = self.map.agent.angles_between()
        angle_features = np.array([[np.sin(a), np.cos(a)]
                                   for a in angles]).flatten()

        # Goal information
        goal_vec = self.goal.goal - self.map.agent.position

        return np.concatenate((segment_vecs_diffs.flatten(), angle_features.flatten(), goal_vec.flatten(), goal_vec.flatten()))

    def _get_reward(self):
        # Handle collisions
        if self.map.agent.outer_collision_idxs:
            return -10, True  # Smaller penalty

        distances = self._get_target_distance_vecs()

        # Success condition
        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            return 10, True  # Smaller positive reward

        # Calculate potential-based reward
        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential

        # Potential-based reward without constant penalty
        reward = 0.5 * (potential - self.last_target_potential)

        # Add configuration reward (encourage natural cable shapes)
        angles = self.map.agent.angles_between()
        # Reward straighter configurations
        angle_reward = 0.1 * np.sum(np.cos(angles))
        reward += angle_reward

        return reward, False


class CableEnvNaiveR(ResetableEnv, CableEnvNaive):
    pass


class CableEnvNaiveI(ImportableEnv, CableEnvNaive):
    pass


class CableBigTestR(ResetableEnv, CableBigTest):
    pass


class CableBigTestI(ImportableEnv, CableBigTest):
    pass


class CableInnerAnglesR(ResetableEnv, CableInnerAngles):
    pass


class CableInnerAnglesI(ImportableEnv, CableInnerAngles):
    pass


class CableEnvR(ResetableEnv, CableEnv):
    pass


class CableEnvI(ImportableEnv, CableEnv):
    pass


class CablePIDEnvR(ResetableEnv, CablePIDEnv):
    pass


class CablePIDEnvI(ImportableEnv, CablePIDEnv):
    pass
