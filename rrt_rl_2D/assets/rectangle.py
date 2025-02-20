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
        if sensor:
            track_col = False
        else:
            track_col = True
        super().__init__(body_type=body_type, track_collisions=track_col)
        shape = pymunk.Poly.create_box(self._body, (w, h))
        shape.sensor = sensor
        self.shapes = [shape]
        self.position = pos
