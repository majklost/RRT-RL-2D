from pathlib import Path
from stable_baselines3 import PPO
from torch import nn

from rrt_rl_2D import *
from rrt_rl_2D.CLIENT.makers import StandardCableMaker, CableInnerAnglesMaker
from rrt_rl_2D.utils.save_manager import load_manager, get_paths, get_run_paths
from rrt_rl_2D.RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name


EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-standard-'


def inner_angles():
    cfg = STANDARD_CONFIG.copy()
    MAP_NAME = 'Empty'
    cfg['threshold'] = 20
    maker, maker_name, objects = CableInnerAnglesMaker(
        MAP_NAME, cfg=cfg, resetable=True).first_try()
    data = {
        "map_name": MAP_NAME,
        "cfg": cfg
    }
    paths = get_paths(get_name(BASE_NAME), 'comment', maker_name, data=data)
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
    model.learn(total_timesteps=4_000_000, callback=[ch_clb, eval_clb])
    print("Training done")


def standard():
    cfg = STANDARD_CONFIG.copy()
    MAP_NAME = 'Empty'
    cfg['threshold'] = 20
    maker, maker_name, _ = StandardCableMaker(
        MAP_NAME, cfg=cfg, resetable=True).first_try()
    data = {
        "map_name": MAP_NAME,
        "cfg": cfg
    }
    paths = get_paths(get_name(BASE_NAME), 'comment', maker_name, data=data)
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
    model.learn(total_timesteps=4_000_000, callback=[ch_clb, eval_clb])
    print("Training done")


if __name__ == '__main__':
    # inner_angles()
    standard()
