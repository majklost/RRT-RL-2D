import numpy as np
import pygame
from pathlib import Path
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from rrt_rl_2D import *
from rrt_rl_2D.makers import Blend
from rrt_rl_2D.envs.blend_env import BlendEnvI
from rrt_rl_2D.manual_models.base_model import BaseManualModel
from rrt_rl_2D.rendering.env_renderer import EnvRenderer
from rrt_rl_2D.rendering.null_renderer import NullRenderer
from rrt_rl_2D.utils.seed_manager import init_manager
from rrt_rl_2D.export.vel_path_replayer import VelPathReplayerCable
from rrt_rl_2D.utils.save_manager import load_manager, get_run_paths

from rrt_rl_2D.RL.training_utils import create_multi_env

EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)


cfg = STANDARD_CONFIG.copy()
# cfg['seg_num'] = 60
# cfg['cable_length'] = 400

cfg['checkpoint_period'] = 20
cfg['seed_env'] = 25
# cfg['seed_plan'] = 115
cfg['seed_plan'] = 15
cfg['threshold'] = 20
init_manager(cfg['seed_env'], cfg['seed_plan'])

ctrl_idxs = None
# ctrl_idxs = [0]
# ctrl_idxs = [0, cfg['seg_num'] // 2, cfg['seg_num'] - 1]
node_manager = node_managers.ControllableManager(cfg, ctrl_idxs)
node_manager.wanted_threshold = cfg['threshold']


class MyMap(ThickStones):
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


cur_map = MyMap(cfg)
storage = storages.GNAT(distance_fnc)
sampler = BezierSampler(cur_map.agent.length, cfg['seg_num'], (0, 0, 0),
                        (cfg["width"], cfg["height"], 2 * np.pi))


def raw_maker():
    return BlendEnvI(cur_map, 600, VelNodeManager(cfg), render_mode='human', renderer=NullRenderer())


paths = get_run_paths('cable-blend-blend_basic', run_cnt=3)
maker, _ = Blend.standard_stones(raw_maker)


model = PPO.load(paths['model_best'], device='cpu')


env = create_multi_env(maker, 1, normalize_path=paths['norm'])
overall_goal = GoalNode(
    (cfg['width'] - 200, cfg['height'] // 2), threshold=250)
s_wrapper = storage_wrappers.rect_end_wrapper.RectEndWrapper(
    storage, distance_fnc, overall_goal, cfg)

planner = VecEnvPlanner(env, model, cfg)

start_node = env.env_method("export_state")
response = PlannerResponse(start_node, {})

s_wrapper.save_to_storage(response)


chpoints = []


def custom_clb(screen, font):
    for chpoint in chpoints:
        for p in chpoint:
            pygame.draw.circle(screen, (0, 255, 0), p, 5)
        for i in range(len(chpoint) - 1):
            pygame.draw.line(screen, (0, 255, 0), chpoint[i], chpoint[i + 1])
    s_wrapper.render_clb(screen, font)


dummy = NullRenderer()
renderer = EnvRenderer(cfg)
renderer.register_callback(custom_clb)
# env.env_method("set_renderer", renderer)
try:
    for i in range(10000):
        if not s_wrapper.want_next_iter:
            print("Goal reached")
            break

        qrand_raw = sampler.sample()
        node_manager.wanted_position = qrand_raw
        qrand = node_manager.create_goal()
        nearest = s_wrapper.get_nearest(qrand)
        response = planner.check_path(nearest, qrand)
        for node in response.path:
            chpoints.append(node.agent_pos)

        if response.data.get('timeout', False):
            print("Timeout")
        # if i == 1000:
        #     # pass
        #     env.env_method("set_renderer", renderer)

        s_wrapper.save_to_storage(response)
        if i % 100 == 0:
            print("Iteration: ", i)
finally:
    env.env_method("set_renderer", renderer)
    env.render()
    input()
    env.close()

path = s_wrapper.get_path()
print("Path length: ", len(path.nodes))
replayer = VelPathReplayerCable(cur_map, path)
replayer.replay()
