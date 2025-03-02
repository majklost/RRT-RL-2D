# Give goal and start points to the environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from rrt_rl_2D.rendering.env_renderer import EnvRenderer
from rrt_rl_2D.CLIENT.makers.makers import RectPIDMaker
from rrt_rl_2D.controllers.env_controller import CableEnvController, RectEnvController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG
from rrt_rl_2D.nodes import GoalNode
from rrt_rl_2D.node_managers import VelNodeManager
from rrt_rl_2D.RL.training_utils import create_multi_env

s = STANDARD_CONFIG.copy()
s['seed_env'] = 27
s['seed_plan'] = 20


nm = VelNodeManager(cfg=s)


maker_factory = RectPIDMaker('Empty', s, resetable=True)
maker, maker_name, _ = maker_factory.first_try()

goal = GoalNode(
    (s['width'] - 200, s['height'] // 2), threshold=250)
env = create_multi_env(maker, 1, normalize=False)
controller = RectEnvController()
renderer = EnvRenderer(s)
renderer._delayed_init()
env.env_method("set_renderer", renderer)

cur_map = env.env_method("get_map")[0]


obs = env.reset()


for i in range(10000):

    action, _ = controller.predict(obs)
    print(cur_map.agent.velocity)

    obs, reward, done, info = env.step(action)
    if done[0]:
        print(cur_map.agent.collision_data)
        print(info)
        # env.env_method("import_start", state)
        obs = env.reset()
    env.render()
