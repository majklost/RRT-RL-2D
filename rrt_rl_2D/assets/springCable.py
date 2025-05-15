import numpy as np
import warnings
import pymunk


from ..utils.common_utils import rot_matrix
from .multibody import MultiBody
from .rectangle import Rectangle
from .circle import Circle
from .cable import Cable
from .spring_params import SpringParams

springs = SpringParams(stiffness=500, damping=2)

USE_CIRCLES = True


class SpringCable(Cable):
    def __init__(self, pos, length, num_links, thickness, angle=0, max_angle=np.pi / 2, stiffness=100.0, damping=10.0):
        # Initialize spring properties
        self.springParams = springs
        super().__init__(pos, length, num_links, thickness, angle, max_angle)

    def _create_objects(self, pos):
        rm = rot_matrix(self.angle)
        for i in range(self.num_links):
            if USE_CIRCLES:
                r = Circle(pos + rm @ np.array([i * self.segment_length, 0]), self.thickness,
                           pymunk.Body.DYNAMIC)
            else:
                r = Rectangle(pos + rm @ np.array([i * self.segment_length, 0]), self.segment_length, self.thickness,
                              pymunk.Body.DYNAMIC)
                r.orientation = self.angle + np.pi

            self.append(r)

    def _create_pivots(self):
        # Override to create springs instead of pivots
        x = self.empty_len
        for i in range(self.num_links - 1):
            spring = pymunk.constraints.DampedSpring(
                self.bodies[i + 1].body, self.bodies[i].body,
                (x + self.segment_length / 4, 0), (-x - self.segment_length / 4, 0),
                self.segment_length / 4,  # Rest length
                self.springParams.stiffness, self.springParams.damping
            )
            self.pivots.append(spring)

            rotary_limit = pymunk.constraints.RotaryLimitJoint(
                self.bodies[i + 1].body, self.bodies[i].body,
                -self.max_angle, self.max_angle
            )
            self.angular_springs.append(rotary_limit)
