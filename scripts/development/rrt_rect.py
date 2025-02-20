import numpy as np
import pygame
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv as Dumm
from rrt_rl_2D import *
from rrt_rl_2D.envs.rect import RectEnvI
from rrt_rl_2D.manual_models.base_model import BaseManualModel
from rrt_rl_2D.rendering.env_renderer import EnvRenderer
from rrt_rl_2D.rendering.null_renderer import NullRenderer
from rrt_rl_2D.utils.seed_manager import init_manager

cfg = STANDARD_CONFIG.copy()
cfg['checkpoint_period'] = 20
cfg['seed_env'] = 27
cfg['seed_plan'] = 20
cfg['threshold'] = 20
init_manager(cfg['seed_env'], cfg['seed_plan'])
node_manager = node_managers.VelNodeManager(cfg)


class MyMap(RectangleEmpty, ThickStones):
    pass


class LinearModel(BaseManualModel):
    def predict(self, obs):
        return obs, None


def distance_fnc(n1, n2):

    if isinstance(n1, GoalNode):
        if isinstance(n2, GoalNode):
            raise ValueError("Both nodes are goal nodes in comparision.")
        else:
            return np.linalg.norm(n2.agent_pos - n1.goal)

    elif isinstance(n2, GoalNode):
        return np.linalg.norm(n1.agent_pos - n2.goal)
    return np.linalg.norm(n1.agent_pos - n2.agent_pos)


storage = storages.GNAT(distance_fnc)
sampler = NDIMSampler((0, 0), (cfg["width"], cfg["height"]))


dummy_renderer = NullRenderer()


renderer = EnvRenderer(cfg)


def maker():
    return RectEnvI(MyMap(cfg), 1000, VelNodeManager(cfg), render_mode='human', renderer=NullRenderer())


env = make_vec_env(maker, 1, vec_env_cls=Dumm)
overall_goal = GoalNode(
    (cfg['width'] - 200, cfg['height'] // 2), cfg['threshold'])
s_wrapper = storage_wrappers.StandardWrapper(
    storage, distance_fnc, overall_goal, cfg)

planner = VecEnvPlanner(env, LinearModel(), cfg)

start_node = env.env_method("export_state")
response = PlannerResponse(start_node, {})

s_wrapper.save_to_storage(response)


points = []


def custom_clb(screen, font):
    for p in points:
        pygame.draw.circle(screen, (0, 255, 0), p, 5)


renderer.register_callback(custom_clb)

for i in range(15000):
    if not s_wrapper.want_next_iter:
        print("Goal reached")
        break
    qrand_raw = sampler.sample()
    node_manager.wanted_position = qrand_raw
    qrand = node_manager.create_goal()
    nearest = s_wrapper.get_nearest(qrand)
    response = planner.check_path(nearest, qrand)
    for node in response.path:
        points.append(node.agent_pos)

    if response.data.get('timeout', False):
        print("Timeout")
    # if i == 500:
    #     env.env_method("set_renderer", renderer)

    s_wrapper.save_to_storage(response)
    if i % 100 == 0:
        print("Iteration: ", i)
env.env_method("set_renderer", renderer)
env.render()

input()


path = s_wrapper.get_path()
test_env = maker()

env.close()
