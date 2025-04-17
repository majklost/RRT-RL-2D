
from stable_baselines3.common.env_util import make_vec_env
from rrt_rl_2D.envs.cable_radius import CableRadiusNearestObsR
from rrt_rl_2D import *
from rrt_rl_2D.CLIENT.makers import CableRadiusMaker
from rrt_rl_2D.controllers.env_controller import CableEnvController
from rrt_rl_2D.rendering import EnvRenderer

cfg = STANDARD_CONFIG.copy()
mf = CableRadiusMaker("AlmostEmpty", cfg, resetable=True)
maker, maker_name, objects = mf.first_try()
nm = objects['nm']


renderer = EnvRenderer(cfg)
env = make_vec_env(maker, 1)

my_map = env.env_method("get_map")[0]
env.env_method('set_renderer', renderer)
renderer._delayed_init()

controller = CableEnvController(segnum=cfg['seg_num'])
obs = env.reset()

for i in range(10000):
    action, _ = controller.predict(obs)
    print(my_map.agent.angles_between()[0])
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()
    env.render()
