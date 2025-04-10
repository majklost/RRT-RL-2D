# script allowing learning of planner inside RRT
import numpy as np
from typing import TYPE_CHECKING
import pygame
from pathlib import Path
import time
from torch import nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from rrt_rl_2D.CLIENT.makers import DodgeEnvPenaltyReductionMaker, DodgeEnvMaker, DodgeEnvPenaltyMaker
from rrt_rl_2D import *
from rrt_rl_2D.manual_models.base_model import BaseManualModel
from rrt_rl_2D.rendering.env_renderer import EnvRenderer
from rrt_rl_2D.rendering.null_renderer import NullRenderer
from rrt_rl_2D.utils.seed_manager import init_manager
from rrt_rl_2D.export.vel_path_replayer import VelPathReplayerCable
from rrt_rl_2D.utils.save_manager import load_manager, get_run_paths, get_paths
from rrt_rl_2D.RL.training_utils import create_multi_env, get_name, SaveModelCallback
from rrt_rl_2D.export.vel_path_saver import VelPathSaver
from rrt_rl_2D.simulator.standard_config import StandardConfig
if TYPE_CHECKING:
    from rrt_rl_2D.simulator.standard_config import StandardConfig

EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-RRT-'


cfg = STANDARD_CONFIG.copy()
# cfg['seg_num'] = 60
cfg['cable_length'] = 300

cfg['checkpoint_period'] = 20
cfg['seed_env'] = 50
# cfg['seed_plan'] = 115
cfg['seed_plan'] = 60
cfg['threshold'] = 20
init_manager(cfg['seed_env'], cfg['seed_plan'])

MAP_NAME = 'NonConvex'


def distance_fnc(n1, n2):

    if isinstance(n1, GoalNode):
        if isinstance(n2, GoalNode):
            raise ValueError("Both nodes are goal nodes in comparision.")
        else:
            return np.linalg.norm(n2.agent_pos - n1.goal)

    elif isinstance(n2, GoalNode):
        return np.linalg.norm(n1.agent_pos - n2.goal)
    return np.linalg.norm(n1.agent_pos - n2.agent_pos)


# storage = storages.GNAT(distance_fnc)

paths = get_run_paths('cable-standard-dodgePenalty', run_cnt=2)

maker_factory = DodgeEnvPenaltyMaker(MAP_NAME, cfg)

maker, maker_name, stuff = maker_factory.first_try()
node_manager = stuff['nm']


env = create_multi_env(maker, 1, normalize_path=paths['norm'])
cur_map = env.env_method("get_map")[0]


sampler = BezierSampler(cur_map.agent.length, cfg['seg_num'], (0, 0, 0),
                        (cfg["width"], cfg["height"], 2 * np.pi))


overall_goal = GoalNode(
    (cfg['width'] - 200, cfg['height'] // 2), threshold=250)
start_node = env.env_method("export_state")
# s_wrapper = storage_wrappers.rect_end_wrapper.RectEndWrapper(
#     storage, distance_fnc, overall_goal, cfg)


# start_node = env.env_method("export_state")
# response = PlannerResponse(start_node, {})

# s_wrapper.save_to_storage(response)


renderer = EnvRenderer(cfg)

# env.env_method("set_renderer", renderer)


class RRTManager:
    def __init__(self, max_iter=float("inf")):
        self.iter_num = 0
        self.max_iter = max_iter
        self.storage = None
        self.wrapper = None
        self._clear()

    def RRTSample(self):
        if self.iter_num % 100 == 0:
            print("Iteration: ", self.iter_num)
        self.iter_num += 1

        qrand_raw = sampler.sample()
        node_manager.wanted_position = qrand_raw
        qrand = node_manager.create_goal()
        nearest = self.s_wrapper.get_nearest(qrand)
        return nearest, qrand

        # response = planner.check_path(nearest, qrand)

    def RRTStep(self, response):
        self.s_wrapper.save_to_storage(response)
        if not self.s_wrapper.want_next_iter or self.iter_num >= self.max_iter:
            print("End of one RRT episode")
            self.iter_num = 0
            self._clear()

    def _clear(self):
        print("Clearing storage")

        self.storage = storages.GNAT(distance_fnc)
        self.s_wrapper = storage_wrappers.rect_end_wrapper.RectEndWrapper(
            self.storage, distance_fnc, overall_goal, cfg)
        response = PlannerResponse(start_node, {})
        self.s_wrapper.save_to_storage(response)


class RRTCallback(BaseCallback):
    def __init__(self, rrtmngr, cfg: StandardConfig, verbose=0):
        super().__init__(verbose)
        self.max_steps = cfg['max_steps']
        self.period = cfg['checkpoint_period']
        self.cur_step_cnt = 0
        self.rrtmngr = rrtmngr
        self.cur_start = None

        self.exports = []

    def _on_training_start(self):
        print("INIT START")
        assert self.training_env.num_envs == 1, "Planner only supports single environment"
        nearest, qrand = self.rrtmngr.RRTSample()
        self._env_import(nearest, qrand)
        print("INIT END")

    def _ending_step(self):
        """Reset start and goal"""
        self._link_nodes(self.exports, self.cur_start)

        response = PlannerResponse(self.exports, dict(
            reached=False, timeout=False, fail=False))
        self.rrtmngr.RRTStep(response)
        nearest, qrand = self.rrtmngr.RRTSample()
        self._env_import(nearest, qrand)

    def _env_import(self, start, goal):
        # print("IMPORTING")
        self.training_env.env_method("import_start", start)
        self.training_env.env_method("import_goal", goal)
        self.cur_start = start
        self.exports = []

    def _env_export(self):
        return self.training_env.env_method("export_state")[0]

    def _link_nodes(self, exports, start):
        if len(exports) == 0:
            return
        exports[0].parent = start
        for i in range(len(exports) - 1):
            exports[i + 1].parent = exports[i]

    def _standard_step(self):
        if self.cur_step_cnt % self.period == 0 and self.cur_step_cnt != 0:
            self.exports.append(self._env_export())

    def _on_step(self) -> bool:
        assert "dones" in self.locals, "Dones not found in locals"
        is_done = self.locals["dones"][0]
        self.training_env.env_method("render")
        is_timeout = self.max_steps == self.cur_step_cnt
        if is_done or is_timeout:
            self.cur_step_cnt = 0
            self._ending_step()
        else:
            self.cur_step_cnt += 1
            self._standard_step()
        return True


data = {
    "map_name": MAP_NAME,
    "cfg": cfg
}


def run():
    save_paths = get_paths(get_name(BASE_NAME),
                           "comment", maker_name, data=data)
    # model = PPO.load(paths['model_last'], device='cpu', env=env)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=64, gamma=0.999, learning_rate=1.434646320811716e-04, clip_range=0.4, n_epochs=4, gae_lambda=0.98,
                policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=nn.Tanh,
    ),)

    model.tensorboard_log = save_paths['tb']
    rrtmngr = RRTManager(8_000)
    callback = RRTCallback(rrtmngr, cfg)

    checkpoint_callback = SaveModelCallback(
        save_paths['model_last'], save_freq=20000)
    print("Starting learning")
    model.learn(20_000_000, callback=[callback, checkpoint_callback],)


if __name__ == "__main__":
    run()
