# Give goal and start points to the environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from rrt_rl_2D.maps import *
from rrt_rl_2D.envs.rect import RectEnvI, RectEnvR
from rrt_rl_2D.controllers.env_controller import CableEnvController, RectEnvController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG
from rrt_rl_2D.nodes import GoalNode
from rrt_rl_2D.node_managers import VelNodeManager

s = STANDARD_CONFIG.copy()
s['seed_env'] = 27
s['seed_plan'] = 20


class MyMap(RectangleEmpty, Empty):
    pass


map = MyMap(s)

nm = VelNodeManager(cfg=s)


def maker():
    return RectEnvI(map, 1000, nm, render_mode='human')


goal = GoalNode(
    (s['width'] - 200, s['height'] // 2), threshold=250)
env = make_vec_env(maker, 1)

controller = RectEnvController()

env.env_method("import_goal", goal)
states = env.env_method("export_state")
state = states[0]
env.env_method("import_start", state)

obs = env.reset()


for i in range(10000):

    action, _ = controller.predict(obs)

    obs, reward, done, info = env.step(action)
    if done[0]:
        print(map.agent.collision_data)
        print(info)
        env.env_method("import_start", state)
        obs = env.reset()


    env.render()
