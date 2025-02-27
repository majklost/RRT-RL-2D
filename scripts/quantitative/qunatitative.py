"""
Script to run the quantitative analysis of performance
Based on previous files
4 maps
"""
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3 import PPO


from rrt_rl_2D.CLIENT import *
from rrt_rl_2D import *
from rrt_rl_2D.utils.save_manager import load_manager, get_run_paths
from rrt_rl_2D.RL import *

MAX_STEPS_NUM = 500000


def create_cfg():
    cfg = STANDARD_CONFIG.copy()
    cfg['cable_length'] = 300

    cfg['checkpoint_period'] = 20
    cfg['seed_env'] = 50


def distance_fnc(n1, n2):

    if isinstance(n1, GoalNode):
        if isinstance(n2, GoalNode):
            raise ValueError("Both nodes are goal nodes in comparision.")
        else:
            return np.linalg.norm(n2.agent_pos - n1.goal)

    elif isinstance(n2, GoalNode):
        return np.linalg.norm(n1.agent_pos - n2.goal)
    return np.linalg.norm(n1.agent_pos - n2.agent_pos)


def blender(cfg, map_name, **kwargs):
    paths = get_run_paths('cable-blend-blend_basic', run_cnt=3)
    maker, maker_name, stuff = BlendMaker.first_try(map_name, cfg)
    node_manager = stuff['nm']
    model = PPO.load(paths['model_best'], device='cpu')
    env = create_multi_env(maker, 1, normalize_path=paths['norm'])
    cur_map = env.env_method("get_map")[0]
    sampler = BezierSampler(cur_map.agent.length, cfg['seg_num'], (0, 0, 0),
                            (cfg["width"], cfg["height"], 2 * np.pi))
    

def cable(cfg, map_name, **kwargs):
    pass


def radius(cfg, map_name, **kwargs):
    pass


def rect_fnc(cfg, map_name, **kwargs):
    pass


def main(cur_args):
    cfg = create_cfg()

    storage = storages.GNAT(distance_fnc)
    overall_goal = GoalNode(
        (cfg['width'] - 200, cfg['height'] // 2), threshold=250)
    s_wrapper = storage_wrappers.rect_end_wrapper.RectEndWrapper(
        storage, distance_fnc, overall_goal, cfg)


TXT2MODEL = {
    "Blend": blender,
    "Cable": cable,
    "Radius": radius,
    "Rect": rect_fnc,

}


if __name__ == "__main__":
    EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments" / 'RL'
    EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
    load_manager(EXPERIMENTS_PATH)

    parser = ArgumentParser(
        prog="quantitative",
        description="Run the quantitative analysis of the performance"
    )
    parser.add_argument("mode", type=str, choices=TXT2MODEL.keys())
    parser.add_argument("map_name", type=str, choices=maps.str2map.keys())
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
