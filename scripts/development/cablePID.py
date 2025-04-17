"""Cable movement with PID regulator"""

from rrt_rl_2D.RL.training_utils import create_multi_env
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG
from rrt_rl_2D.CLIENT.makers.makers import PIDCableMaker
from rrt_rl_2D.controllers.env_controller import CableEnvController
from rrt_rl_2D.rendering.env_renderer import EnvRenderer

cfg = STANDARD_CONFIG.copy()


maker_factory = PIDCableMaker('Empty', cfg, resetable=True)
maker, maker_name, _ = maker_factory.first_try()

env = create_multi_env(maker, 1, normalize=False)
renderer = EnvRenderer(cfg)
renderer._delayed_init()
env.env_method("set_renderer", renderer)
cur_map = env.env_method("get_map")[0]
controller = CableEnvController(segnum=cfg['seg_num'])
obs = env.reset()
for i in range(10000):
    action, _ = controller.predict(obs)
    obs, reward, done, info = env.step(action)
    # print(cur_map.agent.velocity)
    if done[0]:
        obs = env.reset()
    env.render()
