# create pictures of maps for thesis
import numpy as np
import matplotlib.pyplot as plt
import pygame

from rrt_rl_2D.maps import *
from rrt_rl_2D.rendering.debug_renderer import DebugRenderer
from rrt_rl_2D.controllers.cable_controller import MultiBodyController
from rrt_rl_2D.controllers.singlebody_controller import SinglebodyController
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG

map_classes = [Empty, Piped, StandardStones, ThickStones, NonConvex]

for map_cls in map_classes:
    cfg = STANDARD_CONFIG.copy()
    cfg['seed_env'] = 50
    m = map_cls(cfg)
    cur_name = map_cls.__name__

    sim = m.sim
    dr = DebugRenderer()
    # controller = SinglebodyController(m.agent)
    controller = MultiBodyController(m.agent)

    dr.attach_controller(controller)
    sim.attach_renderer(dr)
    sim.step()
    pygame.image.save(dr.display, cur_name + ".png")
    print("saved")
