import gymnasium as gym
import numpy as np
from .cable_env import CableEnv
from .rrt_env import ResetableEnv, ImportableEnv


class BlendEnv(CableEnv):
    # RL learns only how much to blend repulsive force from obstacle and attractive force from goal.

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.agent_len,), dtype=np.float64)

    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 4,), dtype=np.float64)

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten()))

    def step(self, action):
        coefs = (self._process_action(action) + 1) / 2
        target_vecs = self._get_target_distance_vecs()
        obstacle_vecs = -self._get_obstacle_distance_vecs()  # repulsive force

        target_vecs /= np.linalg.norm(target_vecs, axis=1, keepdims=True)
        obstacle_vecs /= np.linalg.norm(obstacle_vecs, axis=1, keepdims=True)

        force = coefs[:, None] * target_vecs + \
            (1 - coefs)[:, None] * obstacle_vecs
        force *= self.scale_factor

        return super().step(force.flatten())

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
            return 1000, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential  # return zero
        reward = 10 * (potential - self.last_target_potential) - 20

        self.last_target_potential = potential
        return reward, False


class BlendEnvR(ResetableEnv, BlendEnv):
    pass


class BlendEnvI(ImportableEnv, BlendEnv):
    pass
