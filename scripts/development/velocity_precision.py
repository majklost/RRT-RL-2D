"""Test how much differs velocity descripted path form original path"""
import numpy as np
import matplotlib.pyplot as plt
import pygame

from rrt_rl_2D.maps import *
from rrt_rl_2D.rendering.debug_renderer import DebugRenderer
from rrt_rl_2D.controllers.cable_controller import MultiBodyController
from rrt_rl_2D.controllers.singlebody_controller import SinglebodyController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG


class MyMap(RectangleEmpty, Piped):
    pass


cfg = STANDARD_CONFIG.copy()
cfg['seg_num'] = 40
m = MyMap(cfg)
sim = m.sim
dr = DebugRenderer()
controller = SinglebodyController(m.agent)
# controller = CableController(m.agent)

dr.attach_controller(controller)
sim.attach_renderer(dr)


force_buffer = []
pos1 = []
pos2 = []
check = m.sim.export()

for i in range(1000):

    print(m.cfg)
    pos1.append(m.agent.position)
    sim.step()
    force_buffer.append(m.agent.velocity)
    if m.agent.outer_collision_idxs:
        print("Collision ", i)


dr._controller = None
m.sim.import_from(check)
for i in range(len(force_buffer)):
    pos2.append(m.agent.position)
    sim.step()

    m.agent.velocity = force_buffer[i]


errors = [np.max(np.linalg.norm(p1 - p2, axis=1))
          for p1, p2 in zip(pos1, pos2)]
plt.plot(errors)
print(max(errors))
plt.show()
