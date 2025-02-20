# Give goal and start points to the environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from rrt_rl_2D.maps import *
from rrt_rl_2D.assets.cable import Cable
from rrt_rl_2D.envs.cable_radius import CableRadiusR, CableRadiusI
from rrt_rl_2D.controllers.env_controller import CableEnvController, RectEnvController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG
from rrt_rl_2D.nodes import GoalNode
s = STANDARD_CONFIG.copy()
s['seed_env'] = 27
s['seed_plan'] = 20
s['seg_num'] = 1
s['cable_length'] = 40


class MyMap(ThickStones):
    pass


map = MyMap(s)


def maker():
    return CableRadiusI(map, 500, render_mode='human')


goal = GoalNode((s['width'] - 200, s['height'] // 2), 100)
env = make_vec_env(maker, 1)

# controller = RectEnvController()
controller = CableEnvController(segnum=s['seg_num'])


env.env_method("import_goal", goal)
states = env.env_method("export_state")
state = states[0]
env.env_method("import_start", state)

obs = env.reset()


for i in range(10000):
    action, _ = controller.predict(obs)

    obs, reward, done, info = env.step(action)
    if done[0]:
        env.env_method("import_start", state)
        obs = env.reset()

    # if i % 200 == 0:
    #     env.env_method("import_start", state)

    # if i == 100:
    #     states = env.env_method("export_state")
    #     state = states[0]
    # elif i % 100 == 0:
    #     env.env_method("import_start", state)

    env.render()
