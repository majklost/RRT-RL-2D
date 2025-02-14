import numpy as np
import pymunk
from ..assets import *

from .empty import Empty
from ..utils.common_utils import deg2rad


STATIC = pymunk.Body.STATIC


class NonConvex(Empty):
    def _add_assets(self):
        self._add_non_convex()
        return super()._add_assets()

    def _add_non_convex(self):
        self._add_v_shape(
            np.array([self.cfg['width'] // 2, self.cfg['height'] // 2]), 420, 35)
        self._add_v_shape(
            np.array([self.cfg['width'] // 2 - 400, self.cfg['height'] // 3]), 400, 60)
        self._add_v_shape(
            np.array([self.cfg['width'] // 2 + 200, self.cfg['height'] // 2 - 300]), 200, 30)
        self._add_v_shape(
            np.array([self.cfg['width'] // 2 + 200, self.cfg['height'] // 2 + 300]), 200, 30)
        self._add_v_shape(
            np.array([self.cfg['width'] // 2 - 600, self.cfg['height'] // 2 + 300]), 200, 40)

        for i in range(5):
            self._add_v_shape(
                np.array([self.cfg['width'] - self.EMPTY - 200, self.cfg['height'] - i * 200]), 80, 40)

    def _add_v_shape(self, pos, length, angle):
        rect1 = Rectangle(pos + (0, length // 4), length, 20, STATIC)
        rect1.orientation = np.pi - deg2rad(angle)
        rect2 = Rectangle(pos + (0, -length // 4), length, 20, STATIC)
        rect2.orientation = np.pi + deg2rad(angle)
        self.fixed_objects.append(rect1)
        self.fixed_objects.append(rect2)
