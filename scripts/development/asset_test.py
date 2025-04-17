import numpy as np
import matplotlib.pyplot as plt
import pygame

from rrt_rl_2D.maps import *
from rrt_rl_2D.rendering.debug_renderer import DebugRenderer
from rrt_rl_2D.controllers.cable_controller import CableController
from rrt_rl_2D.controllers.singlebody_controller import SinglebodyController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG


class MyMap(Piped):
    pass


cfg = STANDARD_CONFIG.copy()
cfg['seg_num'] = 20
m = MyMap(cfg)
sim = m.sim
dr = DebugRenderer(render_constraints=False)
# controller = SinglebodyController(m.agent)
controller = CableController(m.agent, moving_force=1000)

dr.attach_controller(controller)
sim.attach_renderer(dr)


force_buffer = []
pos1 = []
pos2 = []
check = m.sim.export()

for i in range(10000):
    sim.step()
    if m.agent.outer_collision_idxs:
        print("Collision ", i)
