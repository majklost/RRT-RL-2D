import pygame
from rrt_rl_2D.manual_models.blend_manual import BlendManualModel
from rrt_rl_2D import *
from rrt_rl_2D.CLIENT import *
from rrt_rl_2D.RL.training_utils import create_multi_env
from rrt_rl_2D.rendering.env_renderer import EnvRenderer

cfg = STANDARD_CONFIG.copy()
cfg['threshold'] = 20
maker, maker_name, _ = BlendMaker(
    'StandardStones', cfg, resetable=True).first_try()
env = create_multi_env(maker, 1, normalize=False)
cur_map = env.env_method('get_map')[0]
model = BlendManualModel(cfg['seg_num'])
renderer = EnvRenderer(cfg)
renderer._delayed_init()
env.env_method('set_renderer', renderer)
obs = env.reset()

for i in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()
    env.render()
    if pygame.event.get(pygame.QUIT):
        break
