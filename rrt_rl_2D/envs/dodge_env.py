import gymnasium as gym
import numpy as np
from .rrt_env import BaseEnv, ResetableEnv, ImportableEnv
from .cable_env import CableEnv
# In this env, cable is driven to target by standard force techniques, goal of the RL is to drive around obstacles
# Multiple environments are set up - with or without velocity


class DodgeEnv(CableEnv):
    def __init__(self, cur_map, scale_factor, node_factory, render_mode=None):
        super().__init__(cur_map, scale_factor, node_factory, render_mode=render_mode)

    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 2 + 1,), dtype=np.float64)

    def _normalized_step_num(self):
        """
        Returns normalized step number in range [-1, 1]
        """
        return (self.step_num - 500) / 500

    def _get_observation(self):
        obstacle = self._get_obstacle_distance_vecs()
        return np.concatenate([obstacle.flatten(), [self._normalized_step_num()]])

    def _get_reward(self):
        if self.map.agent.outer_collision_idxs:
            self.fail = True
            return 0, True

        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            self.reached = True
            return 0, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential  # return zero

        reward = potential - self.last_target_potential
        self.last_target_potential = potential
        return reward, False

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.agent_len * 2,), dtype=np.float64)

    def step(self, action):
        self.last_actions = np.zeros((self.agent_len, 2))
        if self.goal.controllable_idxs is not None:
            idxs = self.goal.controllable_idxs
        else:
            idxs = range(self.agent_len)
        advised_actions = self._get_target_distance_vecs().flatten()

        for i in idxs:
            cur_action = action[i * 2: i * 2 + 2]
            advised_action = advised_actions[i * 2: i * 2 + 2]
            cur_action = self._process_action(cur_action, i)
            advised_action = self._process_action(advised_action, i)
            cur_action = self._process_action(
                2 * cur_action + advised_action, i)
            self.last_actions[i, :] = cur_action
            self.map.agent.bodies[i].apply_force(cur_action)

        self.last_actions = np.array(action).reshape((self.agent_len, 2))
        return BaseEnv.step(self, action)


class DodgeEnvVel(DodgeEnv):
    def _create_observation_space(self):
        limit = max(self.map.cfg['width'], self.map.cfg['height'])
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.agent_len * 4 + 1,), dtype=np.float64)

    def _get_observation(self):
        obstacle = self._get_obstacle_distance_vecs()
        return np.concatenate([obstacle.flatten(), self.map.agent.velocity.flatten(), [self._normalized_step_num()]])


class DodgeEnvPenalty(DodgeEnv):
    def _get_reward(self):
        if self.map.agent.outer_collision_idxs:
            self.fail = True
            return -1000, True

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


class DodgeEnvReduction(DodgeEnv):
    def _calc_transition(self):
        return np.tanh(1.8 * self._normalized_step_num()) / np.tanh(1.8)

    def step(self, action):
        self.last_actions = np.zeros((self.agent_len, 2))
        if self.goal.controllable_idxs is not None:
            idxs = self.goal.controllable_idxs
        else:
            idxs = range(self.agent_len)
        advised_actions = self._get_target_distance_vecs().flatten()

        for i in idxs:
            cur_action = action[i * 2: i * 2 + 2]
            advised_action = advised_actions[i * 2: i * 2 + 2]
            cur_action = self._process_action(cur_action, i)
            advised_action = self._process_action(advised_action, i)
            cur_action = self._process_action(
                (2 - 2 * self._calc_transition()) * cur_action + advised_action, i)
            self.last_actions[i, :] = cur_action
            self.map.agent.bodies[i].apply_force(cur_action)

        self.last_actions = np.array(action).reshape((self.agent_len, 2))
        return BaseEnv.step(self, action)


class DodgeEnvPenaltyReduction(DodgeEnvReduction):
    def _get_reward(self):
        if self.map.agent.outer_collision_idxs:
            self.fail = True
            return -500, True

        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.goal.threshold):
            self.reached = True
            return 500, True

        potential = self._calc_potential(distances)
        if self.last_target_potential == 0:
            self.last_target_potential = potential  # return zero

        reward = potential - self.last_target_potential
        self.last_target_potential = potential
        return reward, False


class DodgeEnvReductionVel(DodgeEnvVel, DodgeEnvReduction):
    def step(self, action):
        return DodgeEnvReduction.step(self, action)


class DodgeEnvPenaltyReductionR(ResetableEnv, DodgeEnvPenaltyReduction):
    pass


class DodgeEnvPenaltyReductionI(ImportableEnv, DodgeEnvPenaltyReduction):
    pass


class DodgeEnvPenaltyR(ResetableEnv, DodgeEnvPenalty):
    pass


class DodgeEnvPenaltyI(ImportableEnv, DodgeEnvPenalty):
    pass


class DodgeEnvReductionR(ResetableEnv, DodgeEnvReduction):
    pass


class DodgeEnvReductionI(ImportableEnv, DodgeEnvReduction):
    pass


class DodgeEnvReductionVelR(ResetableEnv, DodgeEnvVel, DodgeEnvReduction):
    pass


class DodgeEnvReductionVelI(ImportableEnv, DodgeEnvVel, DodgeEnvReduction):
    pass


class DodgeEnvR(ResetableEnv, DodgeEnv):
    pass


class DodgeEnvI(ImportableEnv, DodgeEnv):
    pass


class DodgeEnvVelR(ResetableEnv, DodgeEnvVel):
    pass


class DodgeEnvVelI(ImportableEnv, DodgeEnvVel):
    pass
