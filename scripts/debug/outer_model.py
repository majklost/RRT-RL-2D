# from stable_baselines3 import PPO
# from pathlib import Path

# from rrt_rl_2D.makers.makers import *
# from rrt_rl_2D.RL.players import play_model

# maker, _ = CableRadius.obs_vel_stronger_fast(render_mode='human')
# model_path = Path('./scripts/debug/test_model.zip')
# norm_path = Path('./scripts/debug/test_norms.pkl')
# play_model(model_path, norm_path, maker, normalize=True)

from pathlib import Path
import numpy as np
import pygame
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from rrt_rl_2D.RL.training_utils import standard_wrap, create_callback_list, get_name, create_multi_env
from rrt_rl_2D.utils.save_manager import load_manager, get_run_paths
from rrt_rl_2D import *
from rrt_rl_2D.envs.cable_radius import CableRadiusI
from rrt_rl_2D.manual_models.base_model import BaseManualModel
from rrt_rl_2D.rendering.env_renderer import EnvRenderer
from rrt_rl_2D.rendering.null_renderer import NullRenderer
from rrt_rl_2D.utils.seed_manager import init_manager
from rrt_rl_2D.export.vel_path_replayer import VelPathReplayerCable
from rrt_rl_2D.makers.makers import CableRadiusMaker


EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-radius-'

VANILLA = True


cfg = STANDARD_CONFIG.copy()
cfg['checkpoint_period'] = 120
cfg['seed_env'] = 21
cfg['seed_plan'] = 20
# cfg['seed_plan'] = 15
cfg['seed_plan'] = 1332
cfg['threshold'] = cfg['cable_length'] / 2
cfg['seg_num'] = 10
init_manager(cfg['seed_env'], cfg['seed_plan'])
node_manager = node_managers.VelNodeManager(cfg)
node_manager.wanted_threshold = cfg['threshold']


class MyMap(StandardStones):
    pass


class LinearModel(BaseManualModel):
    def predict(self, obs, **kwargs):
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
cur_map = MyMap(cfg)


def raw_maker():
    return CableRadiusNearestObsVelI(cur_map, 300, VelNodeManager(cfg), render_mode='human', renderer=NullRenderer())


maker = standard_wrap(raw_maker, max_episode_steps=1000)
# maker, _ = CableRadius.obs_vel_stronger_stones(
#     render_mode='human', resetable=False)

paths = get_run_paths("cable-radius-obs_vel_stones_relearn")


overall_goal = GoalNode(
    (cfg['width'] - 200, cfg['height'] // 2), threshold=250)
s_wrapper = storage_wrappers.rect_end_wrapper.RectEndWrapper(
    storage, distance_fnc, overall_goal, cfg)


if VANILLA:
    env = create_multi_env(maker, 1, normalize=False)
    planner = VecEnvPlanner(env, LinearModel(), cfg)
else:
    env = create_multi_env(maker, 1, normalize_path=paths['norm'])
    env.training = False
    model = PPO.load(paths['model_best'], device='cpu', env=env)
    planner = VecEnvPlanner(env, model, cfg)


start_node = env.env_method("export_state")
response = PlannerResponse(start_node, {})

s_wrapper.save_to_storage(response)


chpoints = []


def custom_clb(screen, font):
    for chpoint in chpoints:
        for p in chpoint:
            pygame.draw.circle(screen, (0, 255, 0), p, 5)
    s_wrapper.render_clb(screen, font)


renderer = EnvRenderer(cfg)
renderer.register_callback(custom_clb)
env.env_method("set_renderer", renderer)

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
        chpoints.append(node.agent_pos)

    if response.data.get('timeout', False):
        print("Timeout")
    # if i == 100:
    #     env.env_method("set_renderer", renderer)

    s_wrapper.save_to_storage(response)
    if i % 100 == 0:
        print("Iteration: ", i)

env.env_method("set_renderer", renderer)
env.render()
input()
env.close()

path = s_wrapper.get_path()
print("Path length: ", len(path.nodes))
replayer = VelPathReplayerCable(cur_map, path)
replayer.replay()
