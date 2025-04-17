import numpy as np
import pygame
from pathlib import Path
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from rrt_rl_2D.CLIENT.makers import DodgeEnvPenaltyReductionMaker, DodgeEnvMaker, DodgeEnvPenaltyMaker
from rrt_rl_2D import *
from rrt_rl_2D.manual_models.base_model import BaseManualModel
from rrt_rl_2D.rendering.env_renderer import EnvRenderer
from rrt_rl_2D.rendering.null_renderer import NullRenderer
from rrt_rl_2D.utils.seed_manager import init_manager
from rrt_rl_2D.export.vel_path_replayer import VelPathReplayerCable
from rrt_rl_2D.utils.save_manager import load_manager, get_run_paths

from rrt_rl_2D.RL.training_utils import create_multi_env
from rrt_rl_2D.export.vel_path_saver import VelPathSaver


EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)


cfg = STANDARD_CONFIG.copy()
# cfg['seg_num'] = 60
cfg['cable_length'] = 300

cfg['checkpoint_period'] = 20
cfg['seed_env'] = 50
# cfg['seed_plan'] = 115
cfg['seed_plan'] = 60
cfg['threshold'] = 20
init_manager(cfg['seed_env'], cfg['seed_plan'])

MAP_NAME = 'AlmostEmpty'


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

paths = get_run_paths('cable-RRT-run', run_cnt=7)

maker_factory = DodgeEnvPenaltyMaker(MAP_NAME, cfg)

maker, maker_name, stuff = maker_factory.first_try()
node_manager = stuff['nm']

model = PPO.load(paths['model_last'], device='cpu')


env = create_multi_env(maker, 1, normalize_path=paths['norm'])
cur_map = env.env_method("get_map")[0]


sampler = BezierSampler(cur_map.agent.length, cfg['seg_num'], (0, 0, 0),
                        (cfg["width"], cfg["height"], 2 * np.pi))


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
    for i in range(6000):
        if not s_wrapper.want_next_iter:
            print("Goal reached")
            print("Iterations: ", i)
            print("steps sum: ", planner.step_cnt)
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
save = VelPathSaver(maker_name, path, cfg, MAP_NAME, {}, __file__)
save.save(".", "test")
replayer = VelPathReplayerCable(cur_map, path)
replayer.replay()
