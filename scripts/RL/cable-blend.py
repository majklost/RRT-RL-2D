import numpy as np
import pygame
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch import nn

from rrt_rl_2D import *
from rrt_rl_2D.RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name

from rrt_rl_2D.makers import BlendMaker

from rrt_rl_2D.utils.save_manager import load_manager, get_paths, get_run_paths

EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-blend-'


def blend_basic():
    maker, maker_name = BlendMaker.first_try()
    paths = get_paths(get_name(BASE_NAME), 'comment', maker_name)
    env = create_multi_env(
        maker, 32, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)

    ch_clb, eval_clb = create_callback_list(paths=paths, eval_env=eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=4000000, callback=[ch_clb, eval_clb])
    print("Training done")


if __name__ == '__main__':
    blend_basic()
