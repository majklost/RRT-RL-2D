from pathlib import Path
from stable_baselines3 import PPO
from torch import nn

from rrt_rl_2D import *
from rrt_rl_2D.CLIENT.makers.makers import LastEnvMaker
from rrt_rl_2D.utils.save_manager import load_manager, get_paths, get_run_paths
from rrt_rl_2D.RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name


EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-last-'


def last():
    PER_STAGE = 8_000_000
    script_folder = Path(__file__).parent
    cfg = STANDARD_CONFIG.copy()
    cfg['threshold'] = 20
    MAP_NAME0 = 'Empty'
    MAP_NAME1 = 'AlmostEmpty'
    MAP_NAME2 = 'StandardStones'
    MAP_NAME3 = 'Piped'
    num_envs = 64
    device = 'cpu'
    data = {
        "map_name": 'AlmostEmpty',
        "cfg": cfg
    }
    maker0, maker_name0, objects0 = LastEnvMaker(
        MAP_NAME0, cfg=cfg, resetable=True).first_try()
    maker1, maker_name1, objects1 = LastEnvMaker(
        MAP_NAME1, cfg=cfg, resetable=True).first_try()
    maker2, maker_name2, objects2 = LastEnvMaker(
        MAP_NAME2, cfg=cfg, resetable=True).first_try()
    maker3, maker_name3, objects3 = LastEnvMaker(
        MAP_NAME3, cfg=cfg, resetable=True).first_try()
    paths = get_paths(get_name(BASE_NAME), 'dry_run', maker_name1, data=data)

    env0 = create_multi_env(maker0, num_envs, normalize=True)

    eval_env = create_multi_env(maker3, 1, normalize=True)
    ch_clb, eval_clb = create_callback_list(paths=paths, eval_env=eval_env)

    model = PPO("MlpPolicy", env0, verbose=0, tensorboard_log=paths['tb'], device=device,
                batch_size=2048, gamma=0.999, learning_rate=3e-4,
                clip_range=0.2, n_epochs=10, gae_lambda=0.95,
                policy_kwargs=dict(
                    net_arch=dict(pi=[512, 256, 256], vf=[512, 256, 256]),
                    activation_fn=nn.ReLU,
    ),
    )

    # learning starts
    print("Stage 0: Training model")
    model.learn(total_timesteps=PER_STAGE, callback=[
                ch_clb, eval_clb], reset_num_timesteps=False)
    model.save(script_folder / 'last_model_0')
    env0.save(script_folder / 'last_env_0_norms')
    env0.close()
    print("Stage 0: Training done")

    env1 = create_multi_env(maker1, num_envs, normalize=True,
                            normalize_path=script_folder / 'last_env_0_norms')

    print("Stage 1: Training model")
    model.set_env(env1)
    model.learn(total_timesteps=PER_STAGE, callback=[
                ch_clb, eval_clb], reset_num_timesteps=False)
    model.save(script_folder / 'last_model_1')
    env1.save(script_folder / 'last_env_1_norms')
    env1.close()
    print("Stage 1: Training done")

    env2 = create_multi_env(maker2, num_envs, normalize=True,
                            normalize_path=script_folder / 'last_env_1_norms')

    print("Stage 2: Training model")
    model.set_env(env2)
    model.learn(total_timesteps=PER_STAGE, callback=[
                ch_clb, eval_clb], reset_num_timesteps=False)
    model.save(script_folder / 'last_model_2')
    env2.save(script_folder / 'last_env_2_norms')
    env2.close()
    print("Stage 2: Training done")

    env3 = create_multi_env(maker3, num_envs, normalize=True,
                            normalize_path=script_folder / 'last_env_2_norms')

    print("Stage 3: Training model")
    model.set_env(env3)
    model.learn(total_timesteps=PER_STAGE, callback=[
                ch_clb, eval_clb], reset_num_timesteps=False)
    model.save(script_folder / 'last_model_3')
    env3.save(script_folder / 'last_env_3_norms')
    env3.close()
    print("Stage 3: Training done")


if __name__ == '__main__':
    last()
