from stable_baselines3.common.env_util import make_vec_env


from rrt_rl_2D.envs.cable_radius import CableRadiusNearestObsR
from rrt_rl_2D import *
from rrt_rl_2D.controllers.env_controller import CableEnvController

cfg = STANDARD_CONFIG.copy()
my_map = AlmostEmpty(cfg)
nm = NodeManager(cfg)


def maker():
    return CableRadiusNearestObsR(my_map, 1000, nm, render_mode='human')


env = make_vec_env(maker, 1)

controller = CableEnvController(segnum=cfg['seg_num'])
obs = env.reset()

for i in range(10000):
    action, _ = controller.predict(obs)
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()
    env.render()
