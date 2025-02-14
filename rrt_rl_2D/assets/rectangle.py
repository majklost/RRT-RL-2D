import pymunk
import numpy as np


from .singlebody import SingleBody


class Rectangle(SingleBody):
    def __init__(self,
                 pos: np.array,
                 w: float,
                 h: float,
                 body_type,
                 sensor=False):
        super().__init__(body_type=body_type)
        shape = pymunk.Poly.create_box(self._body, (w, h))
        shape.sensor = sensor
        self.shapes = [shape]
        self.position = pos
