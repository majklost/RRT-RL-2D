import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from .cable_env import CableEnv
from .rrt_env import BaseEnv, ResetableEnv, ImportableEnv


class LastEnv(CableEnv):
    """
    Last big meaningful test of the cable to see if there is a chance of success.
    """

    def _create_observation_space(self):
        positions_dim = self.agent_len * 2  # x,y for each node
        # vector between consecutive nodes
        segments_dim = (self.agent_len - 1) * 2
        angles_dim = (self.agent_len - 1) * 2  # sin,cos for each inner angle
        goal_dim = 3 * self.agent_len  # goal vector (x,y) + distance

        # distance + vector(x,y) for each obstacle
        obstacle_dim = self.agent_len * 2 + self.agent_len

        total_dims = positions_dim + segments_dim + angles_dim + goal_dim + obstacle_dim

        # Use generous bounds for all dimensions
        limit = max(self.map.cfg['width'], self.map.cfg['height']) * 2
        return gym.spaces.Box(low=-limit, high=limit, shape=(total_dims,), dtype=np.float64)

    def _get_observation(self):
        # 1. POSITIONS - global cable state (absolute positions)
        positions = np.array(self.map.agent.position).flatten()

        # 2. SEGMENTS - local relationships between nodes
        segments = np.diff(self.map.agent.position, axis=0).flatten()

        # 3. ANGLES - with sin/cos representation to avoid discontinuities
        angles = self.map.agent.angles_between()
        angle_features = np.array([[np.sin(a), np.cos(a)]
                                  for a in angles]).flatten()

        # 4. GOAL - relative vector and distance
        goal_vectors = self._get_target_distance_vecs()
        goal_distances = np.linalg.norm(goal_vectors, axis=1)
        goal_vectors = goal_vectors.flatten()

        # 5. OBSTACLES - closest distances and vectors
        obstacle_features = self._get_obstacle_distance_vecs()

        obstacle_distances = np.linalg.norm(obstacle_features, axis=1)
        obstacle_features = obstacle_features.flatten()

        # Combine all features
        return np.concatenate([
            positions,           # Global position context
            segments,            # Local segment relationships
            angle_features,      # Angular information (sin/cos)
            goal_distances,       # Distance to goal
            goal_vectors,     # Distance to goal
            obstacle_features,    # Obstacle awareness
            obstacle_distances     # Obstacle awareness
        ])

    def _get_reward(self):
        # Handle collisions
        if self.map.agent.outer_collision_idxs:
            return -10, True  # Smaller penalty than original -1000

        distances = self._get_target_distance_vecs()

        # Success condition
        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            return 10, True  # Smaller positive reward than original 1000

        # Calculate potential-based reward
        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential
            return -1, False  # Small initial penalty

        # Progress reward with smaller time penalty
        reward = 0.5 * (potential - self.last_target_potential) - \
            5  # Reduced from -20

        # Add configuration reward (encourage natural cable shapes)
        angles = self.map.agent.angles_between()
        # Reward straighter configurations
        angle_reward = 0.1 * np.sum(np.cos(angles))
        reward += angle_reward

        self.last_target_potential = potential
        return reward, False


class LastEnvR(ResetableEnv, LastEnv):
    pass


class LastEnvI(ImportableEnv, LastEnv):
    pass


if __name__ == '__main__':

    from ..node_managers.controllable_manager import ControllableManager
    from ..simulator.standard_config import STANDARD_CONFIG
    from ..maps import Empty



    node_factory = ControllableManager(STANDARD_CONFIG)
    cur_map = Empty(STANDARD_CONFIG)
    env = LastEnvR(cur_map, 300, node_factory)
    check_env(env)
    env.close()
