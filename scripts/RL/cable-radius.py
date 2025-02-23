import numpy as np
import pygame
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch import nn

from rrt_rl_2D import *
from rrt_rl_2D.RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name

from rrt_rl_2D.makers import CableRadius, Debug

from rrt_rl_2D.utils.save_manager import load_manager, get_paths, get_run_paths

EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-radius-'


def obs_vel_stronger_fast():
    maker, maker_name = CableRadius.obs_vel_stronger_fast()
    paths = get_paths(get_name(BASE_NAME),
                      'disabled reset when creating envs', maker_name)
    env = create_multi_env(
        maker, 32, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)

    ch_clb, eval_clb = create_callback_list(paths=paths, eval_env=eval_env)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=32, gamma=0.9999, learning_rate=7.134646320811716e-05, clip_range=0.4, n_epochs=4, gae_lambda=0.98,
                policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh),
    )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[ch_clb, eval_clb])
    print("Training done")


def obs_vel_stronger_stones():
    maker, maker_name = CableRadius.obs_vel_stronger_stones()
    paths = get_paths(get_name(BASE_NAME), 'comment', maker_name)
    env = create_multi_env(
        maker, 32, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)

    ch_clb, eval_clb = create_callback_list(paths=paths, eval_env=eval_env)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=32, gamma=0.9999, learning_rate=7.134646320811716e-05, clip_range=0.4, n_epochs=4, gae_lambda=0.98,
                policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh),
    )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[ch_clb, eval_clb])
    print("Training done")


def obs_vel_stones_relearn():
    """
    Use learned policy from almost empty map to learn on stones map
    """
    maker, maker_name = CableRadius.obs_vel_stronger_stones()
    paths = get_paths(get_name(BASE_NAME), 'comment', maker_name)
    env = create_multi_env(
        maker, 32, normalize=True)
    base_paths = get_run_paths('cable-radius-obs_vel_stronger_fast')
    eval_env = create_multi_env(
        maker, 1, normalize=True, normalize_path=base_paths['norm'])

    ch_clb, eval_clb = create_callback_list(paths=paths, eval_env=eval_env)

    model = PPO.load(base_paths['model_best'], env=env,
                     verbose=0, tensorboard_log=paths['tb'], device='cpu')

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[ch_clb, eval_clb])
    print("Training done")


def debug_radius():
    maker, maker_name = Debug.debug_radius_fast()
    paths = get_paths(get_name(BASE_NAME), 'comment', maker_name)
    env = create_multi_env(
        maker, 32, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)

    ch_clb, eval_clb = create_callback_list(paths=paths, eval_env=eval_env)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=32, gamma=0.9999, learning_rate=7.134646320811716e-05, clip_range=0.4, n_epochs=4, gae_lambda=0.98,
                policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh),
    )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[ch_clb, eval_clb])
    print("Training done")


if __name__ == '__main__':
    # obs_vel_stronger_fast()
    # obs_vel_stronger_stones()
    # debug_radius()
    obs_vel_stones_relearn()
