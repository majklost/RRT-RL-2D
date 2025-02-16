import numpy as np
import pymunk

from ..simulator.standard_config import STANDARD_CONFIG
from ..simulator.simulator import Simulator
from ..assets import *
from ..utils.seed_manager import init_manager
from ..samplers import *


RECTDIM = 30  # Dimension of the rectangle in the RectangleEmpty class


class Empty:
    """
    Defines the basic environment for the simulation.
    """

    def __init__(self, cfg=STANDARD_CONFIG):
        self.cfg = cfg
        self._modify_config()
        self._calc_spacing_constants()
        self._init_manager()
        self._cur_start_points = None
        self._cur_goal_points = None
        self.fixed_objects = []
        self.movable_objects = []
        self._add_agent()
        self._add_assets()
        self._sim = self.create_sim()

        self._sampler = self._create_sampler()
        self._add_goal_points()

    def _modify_config(self):
        pass

    def _add_goal_points(self):
        self._cur_goal_points = self._sampler.sample(
            self.END[0], self.END[1], angle=0, fixed_shape=True)

    def _calc_spacing_constants(self):
        self.EMPTY = 350
        self.START = (self.EMPTY - 10,
                      self.cfg["height"] // 2 - self.cfg['cable_length'] // 2)
        self.END = (self.cfg["height"] - self.EMPTY //
                    2, self.cfg['height'] // 2)
        self.MARGIN = 50  # Margin around boundings where sampler will not sample

    def _add_assets(self):
        self._add_boundings()

    def _add_boundings(self):
        boundings = Boundings(self.cfg["width"], self.cfg["height"])
        self.fixed_objects.append(boundings)

    def _create_sampler(self):
        lower_bounds = np.array([self.MARGIN, self.MARGIN, 0])
        upper_bounds = np.array(
            [self.cfg["width"] - self.MARGIN, self.cfg["height"] - self.MARGIN, 2 * np.pi])
        return BezierSampler(self.cfg["cable_length"], self.cfg["seg_num"], lower_bounds, upper_bounds)

    def _add_agent(self):
        self.agent = Cable(
            self.START, self.cfg["cable_length"], self.cfg["seg_num"], thickness=5, angle=np.pi)
        self.agent.color = (0, 0, 255)
        self.agent.set_collision_type(1)
        self._cur_start_points = self.agent.position.copy()
        self.movable_objects.append(self.agent)

    def create_sim(self):
        return Simulator(self.movable_objects, self.fixed_objects, self.cfg)

    def _init_manager(self):
        init_manager(self.cfg['seed_env'], self.cfg['seed_plan'])

    @property
    def start_points(self):
        return self._cur_start_points

    @property
    def goal_points(self):
        return self._cur_goal_points

    @property
    def sim(self):
        return self._sim


class ResetableEmpty(Empty):
    """
    Changes map to be resetable. Adds reset_start and reset_goal methods.
    Must be placed after RectangleEmpty in the inheritance list.
    """

    def reset_start(self):
        valid = False
        while not valid:
            pos = self._sampler.sample()
            valid = self._check_validity(pos)
        self.agent.position = pos
        self._cur_start_points = pos

    def reset_goal(self):
        valid = False
        while not valid:
            pos = self._sampler.sample()
            valid = self._check_validity(pos)
        self._cur_goal_points = pos

    def _check_validity(self, pos):
        b = pymunk.Body()
        circ = pymunk.Circle(b, 10)
        for p in pos:
            if not (self.MARGIN < p[0] < self.cfg["width"] - self.MARGIN and self.MARGIN < p[1] < self.cfg["height"] - self.MARGIN):
                return False
            b.position = p.tolist()
            res = self.sim.shape_query(circ)
            if res:
                return False
        return True


class RectangleEmpty(Empty):
    """
    When added as superclass before Empty, it will create a rectangle agent instead of a cable.
    Must be placed before Empty in the inheritance list.
    Must be placed before ResetableEmpty in the inheritance list.
    """

    def _add_agent(self):
        self.agent = Rectangle(self.START, RECTDIM,
                               RECTDIM, pymunk.Body.DYNAMIC)

        self.movable_objects.append(self.agent)

    def _create_sampler(self):
        lower_bounds = np.array([self.MARGIN, self.MARGIN])
        upper_bounds = np.array(
            [self.cfg["width"] - self.MARGIN, self.cfg["height"] - self.MARGIN])
        return NDIMSampler(lower_bounds, upper_bounds)

    def _check_validity(self, pos):
        b = pymunk.Body()
        rect = pymunk.Poly.create_box(b, (RECTDIM, RECTDIM))
        p = pos
        if not (self.MARGIN < p[0] < self.cfg["width"] - self.MARGIN and self.MARGIN < p[1] < self.cfg["height"] - self.MARGIN):
            return False
        b.position = p.tolist()
        res = self.sim.shape_query(rect)
        if res:
            return False
        return True

    def _add_goal_points(self):
        self._cur_goal_points = self.END
