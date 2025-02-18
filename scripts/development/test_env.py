# Give goal and start points to the environment
from stable_baselines3.common.env_util import make_vec_env


from rrt_rl_2D.maps import *
from rrt_rl_2D.envs.cable_radius import CableRadius
from rrt_rl_2D.controllers.env_controller import CableEnvController, RectEnvController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG

s = STANDARD_CONFIG.copy()
s['seed_env'] = 27
s['seed_plan'] = 20


class MyMap(ThickStones):
    pass


map = MyMap(s)


def maker():
    return CableRadius(map, 1000, render_mode='human')


env = make_vec_env(maker, 1)

# controller = RectEnvController()
controller = CableEnvController(segnum=s['seg_num'])


obs = env.reset()
states = env.env_method("export_state")
state = states[0]

for i in range(10000):
    action, _ = controller.predict(obs)

    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

    # if i % 200 == 0:
    #     env.env_method("import_start", state)

    # if i == 100:
    #     states = env.env_method("export_state")
    #     state = states[0]
    # elif i % 100 == 0:
    #     env.env_method("import_start", state)

    env.render()
