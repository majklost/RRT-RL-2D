"""
Script to run the quantitative analysis of performance
Based on previous files
4 maps
"""
import numpy as np
import time
from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from typing import TypedDict


from rrt_rl_2D.CLIENT import *
from rrt_rl_2D.simulator.standard_config import StandardConfig
from rrt_rl_2D import *
from rrt_rl_2D.envs.rrt_env import BaseEnv
from rrt_rl_2D.utils.save_manager import load_manager, get_run_paths
from rrt_rl_2D.RL import *
from rrt_rl_2D.utils.seed_manager import init_manager
from rrt_rl_2D.manual_models import BaseManualModel

MAX_STEPS_NUM = 400_000
TIMEOUT = 2 * 3600


class Returns(TypedDict):
    env: VecEnv
    model: PPO
    sampler: BezierSampler
    node_manager: node_managers.NodeManager
    cur_map: Empty
    cfg: dict
    maker_name: str


def create_cfg():
    cfg = STANDARD_CONFIG.copy()
    cfg['cable_length'] = 300

    cfg['checkpoint_period'] = 20
    cfg['seed_env'] = 50
    return cfg


def distance_fnc(n1, n2):

    if isinstance(n1, GoalNode):
        if isinstance(n2, GoalNode):
            raise ValueError("Both nodes are goal nodes in comparision.")
        else:
            return np.linalg.norm(n2.agent_pos - n1.goal)

    elif isinstance(n2, GoalNode):
        return np.linalg.norm(n1.agent_pos - n2.goal)
    return np.linalg.norm(n1.agent_pos - n2.agent_pos)


def blender(map_name, cfg: StandardConfig, **kwargs):
    cfg['threshold'] = 20
    paths = get_run_paths('cable-blend-blend_basic', run_cnt=3)

    maker_factory = BlendMaker(map_name, cfg)
    maker, maker_name, stuff = maker_factory.analyzable()
    node_manager = stuff['nm']
    model = PPO.load(paths['model_best'], device='cpu')
    env = create_multi_env(maker, 1, normalize_path=paths['norm'])
    cur_map = env.env_method("get_map")[0]
    sampler = BezierSampler(cur_map.agent.length, cfg['seg_num'], (0, 0, 0),
                            (cfg["width"], cfg["height"], 2 * np.pi))
    ret: Returns = {
        "env": env,
        "model": model,
        "sampler": sampler,
        "node_manager": node_manager,
        "cur_map": cur_map,
        "cfg": cfg,
        "maker_name": maker_name
    }
    return ret


def cable(map_name, cfg: StandardConfig, **kwargs):
    cfg['threshold'] = 20

    class LinearModel(BaseManualModel):
        def predict(self, obs, **kwargs):
            return obs, None
    ctrl_idxs = kwargs.get('ctrl_idxs', None)
    if ctrl_idxs is not None:
        raise NotImplementedError("Controlled nodes not implemented")

    maker_factory = StandardCableMaker(map_name, cfg)
    maker, maker_name, stuff = maker_factory.analyzable()
    node_manager = stuff['nm']
    env = create_multi_env(maker, 1, normalize=False)
    cur_map = env.env_method("get_map")[0]
    sampler = BezierSampler(cur_map.agent.length, cfg['seg_num'], (0, 0, 0),
                            (cfg["width"], cfg["height"], 2 * np.pi))
    ret: Returns = {
        "env": env,
        "model": LinearModel(),
        "sampler": sampler,
        "node_manager": node_manager,
        "cur_map": cur_map,
        "cfg": cfg,
        "maker_name": maker_name
    }
    return ret


def radius_dummy(map_name, cfg: StandardConfig, **kwargs):
    cfg['threshold'] = cfg['cable_length'] // 2

    class LinearModel(BaseManualModel):
        def predict(self, obs, **kwargs):
            return obs, None

    maker_factory = CableRadiusMaker(map_name, cfg)
    maker, maker_name, stuff = maker_factory.analyzable()
    node_manager = stuff['nm']
    env = create_multi_env(maker, 1, normalize=False)
    cur_map = env.env_method("get_map")[0]
    sampler = NDIMSampler((0, 0), (cfg["width"], cfg["height"]))
    ret: Returns = {
        "env": env,
        "model": LinearModel(),
        "sampler": sampler,
        "node_manager": node_manager,
        "cur_map": cur_map,
        "cfg": cfg,
        "maker_name": maker_name
    }
    return ret


def radius_rl(map_name, cfg: StandardConfig, **kwargs):
    cfg['threshold'] = cfg['cable_length'] // 2
    rpaths = get_run_paths('cable-radius-obs_vel_stronger_fast', run_cnt=16)
    model = PPO.load(rpaths['model_last'], device='cpu')
    maker_factory = CableRadiusMaker(map_name, cfg)
    maker, maker_name, stuff = maker_factory.analyzable()
    node_manager = stuff['nm']
    env = create_multi_env(maker, 1, normalize_path=rpaths['norm'])
    cur_map = env.env_method("get_map")[0]
    sampler = NDIMSampler((0, 0), (cfg["width"], cfg["height"]))
    ret: Returns = {
        "env": env,
        "model": model,
        "sampler": sampler,
        "node_manager": node_manager,
        "cur_map": cur_map,
        "cfg": cfg,
        "maker_name": maker_name
    }
    return ret


def rect_fnc(map_name, cfg: StandardConfig, **kwargs):

    class LinearModel(BaseManualModel):
        def predict(self, obs, **kwargs):
            return obs, None
    maker_factory = RectMaker(map_name, cfg)
    maker, maker_name, stuff = maker_factory.analyzable()
    node_manager = stuff['nm']
    env = create_multi_env(maker, 1, normalize=False)
    cur_map = env.env_method("get_map")[0]
    sampler = NDIMSampler((0, 0), (cfg["width"], cfg["height"]))
    ret: Returns = {
        "env": env,
        "model": LinearModel(),
        "sampler": sampler,
        "node_manager": node_manager,
        "cur_map": cur_map,
        "cfg": cfg,
        "maker_name": maker_name
    }
    return ret


def main(cur_args):
    cfg = create_cfg()
    cfg["seed_plan"] = cur_args.seed
    init_manager(cfg['seed_env'], cfg['seed_plan'])
    returns: Returns = TXT2MODEL[cur_args.mode](cur_args.map_name, cfg)

    cfg = returns['cfg']
    env = returns['env']

    storage = storages.GNAT(distance_fnc)
    overall_goal = GoalNode(
        (cfg['width'] - 200, cfg['height'] // 2), threshold=250)
    s_wrapper = storage_wrappers.rect_end_wrapper.RectEndWrapper(
        storage, distance_fnc, overall_goal, cfg)

    planner = VecEnvPlannerA(env, returns['model'], cfg)
    start_node = env.env_method("export_state")
    reponse = PlannerResponse(start_node, {})
    s_wrapper.save_to_storage(reponse)
    chpoints = []
    # analytics
    iter_cnt = 0
    storage_time = 0
    reachead_goal = False

    start_t = time.perf_counter()
    while planner.step_cnt < MAX_STEPS_NUM:
        if not s_wrapper.want_next_iter:
            reachead_goal = True
            print("Goal reached")
            break
        qrand_raw = returns['sampler'].sample()
        returns['node_manager'].wanted_position = qrand_raw
        qrand = returns['node_manager'].create_goal()
        stor_time = time.perf_counter()
        nearest = s_wrapper.get_nearest(qrand)
        storage_time += time.perf_counter() - stor_time
        response = planner.check_path(nearest, qrand)
        for node in response.path:
            chpoints.append(node.agent_pos)
        s_wrapper.save_to_storage(response)
        iter_cnt += 1
        if iter_cnt % 100 == 0:
            print("Iterations: ", iter_cnt)
            print("steps sum: ", planner.step_cnt)

        if iter_cnt % 1000 == 0:
            if time.perf_counter() - start_t > TIMEOUT:
                print("Timeout")
                break

    end_t = time.perf_counter()
    print(
        f"Time elapsed: {end_t - start_t}, Iterations: {iter_cnt}, reached goal: {reachead_goal}")
    env.close()

    result_cnts = planner.result_cnts

    path = s_wrapper.get_path()
    saver = VelPathSaver(maker_name=returns['maker_name'], path=path, cfg=cfg, map_name=cur_args.map_name,
                         data={"nodes": chpoints}, script_name=__file__)

    cur_analytics: analytics.StandardAnalytics = {
        'tot_time': end_t - start_t,
        'iterations': iter_cnt,
        'finished': reachead_goal,
        'collided_cnt': result_cnts['collided'],
        'timeout_cnt': result_cnts['timeouts'],
        'reached_cnt': result_cnts['reached'],
        'steps_sum': planner.step_cnt,
        'sim_time': returns["cur_map"].sim.step_time,
        'storage_time': storage_time,
        'node_cnt': planner.node_cnt,
        'env_times': planner.env_times,
        'path_node_cnt': len(path.nodes),
    }
    saver.save(cur_args.path_path, cur_args.name)
    save(cur_analytics, cur_args.analytics_path, cur_args.name)


TXT2MODEL = {
    "Blend": blender,
    "Cable": cable,
    "RadiusDummy": radius_dummy,
    "RadiusRL": radius_rl,
    "Rect": rect_fnc,


}


if __name__ == "__main__":
    EXPERIMENTS_PATH = Path(
        __file__).parent.parent.parent / "experiments" / 'RL'
    EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
    load_manager(EXPERIMENTS_PATH)

    parser = ArgumentParser(
        prog="quantitative",
        description="Run the quantitative analysis of the performance"
    )
    parser.add_argument("mode", type=str, choices=TXT2MODEL.keys())
    parser.add_argument("map_name", type=str, choices=maps.str2map.keys())
    parser.add_argument("path_path", type=str)
    parser.add_argument("analytics_path", type=str)
    parser.add_argument("--name", type=str, default="PlannedPath")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
