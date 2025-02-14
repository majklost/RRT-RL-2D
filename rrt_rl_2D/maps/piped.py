import numpy as np
import pymunk


STATIC = pymunk.Body.STATIC

from ..assets import *
from ..utils.common_utils import deg2rad
from .empty import Empty


class Piped(Empty):
    def _add_assets(self):
        self._add_pipes()
        return super()._add_assets()

    def _add_pipes(self):
        blockage = Rectangle(
            np.array([self.EMPTY + 50, self.cfg['height'] // 4 - 50]), self.cfg['height'] // 2 - 80, 20, STATIC)
        blockage2 = Rectangle(np.array(
            [self.EMPTY + 50, self.cfg['height'] - (self.cfg['height'] // 4 - 50)]), self.cfg['height'] // 2 - 80, 20, STATIC)
        blockage.orientation = blockage2.orientation = np.pi / 2
        self.fixed_objects.append(blockage)
        self.fixed_objects.append(blockage2)
        self._create_pipe(
            np.array([self.EMPTY + 200, self.cfg['height'] // 2]), 300, 0, 150)
        self._add_v_shape(
            np.array([self.cfg['width'] // 2 - 150, self.cfg['height'] // 2]), 200, 120)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2 - 220, self.cfg['height'] // 2 - 200]), 250, 20, STATIC)
        rec.orientation = -deg2rad(60)
        self.fixed_objects.append(rec)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2 - 220, self.cfg['height'] // 2 + 180]), 250, 20, STATIC)
        rec.orientation = deg2rad(45)
        self.fixed_objects.append(rec)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2, self.cfg['height'] // 2 + 280]), 400, 20, STATIC)
        self.fixed_objects.append(rec)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2, self.cfg['height'] // 2 + 200]), 400, 20, STATIC)
        rec.orientation = deg2rad(45)
        self.fixed_objects.append(rec)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2 + 80, self.cfg['height'] // 2 - 280]), 700, 20, STATIC)
        self.fixed_objects.append(rec)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2 + 20, self.cfg['height'] // 2 - 120]), 300, 20, STATIC)
        self.fixed_objects.append(rec)
        rec = Rectangle(
            np.array([self.cfg['width'] // 2 + 400, self.cfg['height'] // 2 - 120]), 400, 20, STATIC)
        rec.orientation = -deg2rad(70)
        self.fixed_objects.append(rec)

    def _create_pipe(self, pos, length, angle, width):
        pipe1 = Rectangle(pos + (0, width // 2), length, 20, STATIC)
        pipe1.orientation = angle
        pipe2 = Rectangle(pos - (0, width // 2), length, 20, STATIC)
        pipe2.orientation = angle
        self.fixed_objects.append(pipe1)
        self.fixed_objects.append(pipe2)

    def _add_v_shape(self, pos, length, angle):
        rect1 = Rectangle(pos + (0, length // 3), length, 20, STATIC)
        rect1.orientation = np.pi - deg2rad(angle)
        rect2 = Rectangle(pos + (0, -length // 3), length, 20, STATIC)
        rect2.orientation = np.pi + deg2rad(angle)
        self.fixed_objects.append(rect1)
        self.fixed_objects.append(rect2)
