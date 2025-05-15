"""
Script for testing asset behavior (e.g. foam and cable)
Press arrow keys to move, click on masses to move with the given mass
"""

import numpy as np
import pygame

from rrt_rl_2D.maps import *
from rrt_rl_2D.rendering.debug_renderer import DebugRenderer
from rrt_rl_2D.controllers.cable_controller import MultiBodyController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG


RENDER_CONSTRAINTS = False

# For Foam testing
# class MyMap(FoamEmpty, Piped):
#     pass

# For different map
# class MyMap(FoamEmpty, StandardStones):
#     pass


# For Cable testing
class MyMap(Piped):
    pass

# Spring Cable


class MyMap(SpringEmpty, Piped):
    pass


cfg = STANDARD_CONFIG.copy()
cfg['seg_num'] = 20
m = MyMap(cfg)
sim = m.sim
dr = DebugRenderer(render_constraints=RENDER_CONSTRAINTS)
controller = MultiBodyController(m.agent, moving_force=1000)


dr.attach_controller(controller)
sim.attach_renderer(dr)

for i in range(10000):
    if controller.esc_pressed:
        pygame.image.save(dr.display, "test.png")
        break
    sim.step()
