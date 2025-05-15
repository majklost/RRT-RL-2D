import numpy as np
import pymunk
import warnings

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

    def __init__(self, cfg=STANDARD_CONFIG, sim_cls=Simulator):
        self.cfg = cfg
        self._modify_config()
        self._calc_spacing_constants()
        self._init_manager()

        self.fixed_objects = []
        self.movable_objects = []
        self._add_agent()
        self._add_assets()
        self._sim = self._create_sim(sim_cls)

        self._sampler = self._create_sampler()

    def _modify_config(self):
        pass

    def _calc_spacing_constants(self):
        self.EMPTY = 350
        self.START = (self.EMPTY - 100,
                      self.cfg["height"] // 2 - self.cfg['cable_length'] // 2)
        self.END = (self.cfg["width"] - self.EMPTY //
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
            self.START, self.cfg["cable_length"], self.cfg["seg_num"], thickness=5, angle=np.pi / 2)
        self.agent.color = (0, 0, 255)
        self.agent.set_collision_type(1)
        self.movable_objects.append(self.agent)

    def _create_sim(self, sim_cls):
        return sim_cls(self.movable_objects, self.fixed_objects, self.cfg)

    def _init_manager(self):
        init_manager(self.cfg['seed_env'], self.cfg['seed_plan'])

    @property
    def sim(self):
        return self._sim

    @property
    def sampler(self):
        """
        Sampler that samples points for specific agent type
        """
        return self._sampler

    def check_validity(self, pos: list | np.ndarray):
        """
        Checks if the given positions are valid for the given dimensions.
        :param pos: List of positions to check
        :param dimensions: Dimensions of the object to check - (radius,) for circle, (width, height) for rectangle
        :return: True if the positions are valid, False otherwise
        """
        dimensions = self.cfg["check_dimensions"]
        b = pymunk.Body()
        if len(dimensions) == 1:
            # Circle
            shape = pymunk.Circle(b, dimensions[0])

        elif len(dimensions) == 2:
            # Rectangle
            shape = pymunk.Poly.create_box(b, dimensions)
        else:
            raise ValueError("Invalid dimensions for validity check")

        for p in pos:
            if not (self.MARGIN < p[0] < self.cfg["width"] - self.MARGIN and self.MARGIN < p[1] < self.cfg["height"] - self.MARGIN):
                return False
            b.position = p.tolist()
            res = self.sim.shape_query(shape)
            if res:
                return False
        return True


class ResetableEmpty(Empty):
    """
    Changes map to be resetable. Adds reset_start and reset_goal methods.
    Must be placed after RectangleEmpty in the inheritance list.
    """

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

    def _get_width(self):
        return RECTDIM

    def _get_height(self):
        return RECTDIM

    def _add_agent(self):
        self.agent = Rectangle(self.START, self._get_width(),
                               self._get_height(), pymunk.Body.DYNAMIC)

        self.movable_objects.append(self.agent)

    def _create_sampler(self):
        lower_bounds = np.array([self.MARGIN, self.MARGIN])
        upper_bounds = np.array(
            [self.cfg["width"] - self.MARGIN, self.cfg["height"] - self.MARGIN])
        return NDIMSampler(lower_bounds, upper_bounds)


class CircleEmpty(Empty):
    """
    When added as superclass before Empty, it will create a circle agent instead of a cable.
    Must be placed before Empty in the inheritance list.
    Must be placed before ResetableEmpty in the inheritance list.
    """

    def _get_radius(self):
        return 20

    def _add_agent(self):
        self.agent = Circle(self.START, self._get_radius(),
                            pymunk.Body.DYNAMIC)

        self.movable_objects.append(self.agent)

    def _create_sampler(self):
        lower_bounds = np.array([self.MARGIN, self.MARGIN])
        upper_bounds = np.array(
            [self.cfg["width"] - self.MARGIN, self.cfg["height"] - self.MARGIN])
        return NDIMSampler(lower_bounds, upper_bounds)


class FoamEmpty(Empty):
    """
    When added as superclass before Empty, it will create a foam agent instead of a cable.
    Must be placed before Empty in the inheritance list.
    Must be placed before ResetableEmpty in the inheritance list.
    """

    def _get_width(self):
        return 200

    def _get_height(self):
        return 300

    def _add_agent(self):
        dims = np.array([self._get_width(), self._get_height()])
        self.agent = Foam(self.START - dims / 2, dims,
                          masspoint_per_length=.02, mass_radius=10)

        self.movable_objects.append(self.agent)


class SpringEmpty(Empty):
    """
    When added as superclass before Empty, it will create a foam agent instead of a cable.
    Must be placed before Empty in the inheritance list.
    Must be placed before ResetableEmpty in the inheritance list.
    """

    def _get_width(self):
        return 200

    def _get_height(self):
        return 300

    def _add_agent(self):
        self.agent = SpringCable(
            self.START, self.cfg["cable_length"], self.cfg["seg_num"], thickness=5, angle=np.pi / 2)
        self.agent.color = (0, 0, 255)
        self.agent.set_collision_type(1)
        self.movable_objects.append(self.agent)
