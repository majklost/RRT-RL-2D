from rrt_rl_2D.maps import *
from rrt_rl_2D.rendering.debug_renderer import DebugRenderer
from rrt_rl_2D.controllers.cable_controller import CableController
from rrt_rl_2D.controllers.rect_controller import RectController


class MyMap(RectangleEmpty, ResetableEmpty, NonConvex):
    pass


m = MyMap()
sim = m.sim
dr = DebugRenderer()
controller = RectController(m.agent)
# controller = CableController(m.agent)

dr.attach_controller(controller)
sim.attach_renderer(dr)

for i in range(1000):
    sim.step()
    if i % 100 == 0:
        m.reset_start()
