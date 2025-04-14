import pymunk
import numpy as np

from .singlebody import SingleBody


class Circle(SingleBody):
    def __init__(self, pos: np.array, radius: float, body_type, track_collisions=True):
        super().__init__(body_type, track_collisions)
        shape = pymunk.Circle(self._body, radius)
        self.shapes = [shape]
        self.position = pos
