import numpy as np

from ..assets import *

from .empty import Empty


class ThickStones(Empty):
    def _add_assets(self):
        self._add_stones()
        return super()._add_assets()

    def _add_stones(self):
        stones = RandomObstacleGroup(
            np.array([self.EMPTY + 120, 190]), self.cfg['width'] // 6, self.cfg['height'] // 3.5, 3, 3, radius=180)
        stones.color = (100, 100, 100)
        self.fixed_objects.append(stones)


class StandardStones(Empty):
    def _add_assets(self):
        self._add_stones()
        return super()._add_assets()

    def _add_stones(self):
        stones = RandomObstacleGroup(
            np.array([self.EMPTY + 120, 30]), self.cfg['width'] // 6, self.cfg['height'] // 3.5, 4, 4, radius=130)
        stones.color = (100, 100, 100)
        self.fixed_objects.append(stones)


class AlmostEmpty(Empty):
    def _add_assets(self):
        self._add_stones()
        return super()._add_assets()

    def _add_stones(self):
        stones = RandomObstacleGroup(
            np.array([self.EMPTY + 120, 60]), self.cfg['width'] // 4, self.cfg['height'] // 2.5, 3, 3, radius=100)
        stones.color = (100, 100, 100)
        self.fixed_objects.append(stones)
