import pymunk
import numpy as np
import pygame

from ..simulator.collision_data import CollisionData
from ..utils.common_utils import rot_matrix


class SingleBody:
    def __init__(self, body_type, track_collisions=True):
        super().__init__()
        self._collision_data = None  # type: CollisionData
        self.shapes = []
        self._body = pymunk.Body(body_type=body_type, mass=0, moment=0)
        self._color = (0, 0, 0, 0)
        self.density = .01
        self.collision_clb = None
        self._manual_force = np.array([0, 0])
        self.track_colisions = track_collisions

    @property
    def collision_data(self):
        return self._collision_data

    @property
    def body(self) -> pymunk.Body:
        """
        Pymunk representation of the object
        :return:
        """
        return self._body

    @property
    def bodies(self):
        return [self]

    @property
    def density(self):
        return self._density

    @property
    def position(self):
        return np.array(self._body.position)

    @property
    def orientation(self):
        return self._body.angle

    @density.setter
    def density(self, density):
        self._density = density
        for s in self.shapes:
            s.density = density

    @property
    def velocity(self):
        return np.array(self._body.velocity)

    @property
    def angular_velocity(self):
        return self._body.angular_velocity

    @property
    def friction(self):
        return self._body.friction

    @property
    def color(self):
        return self._color  # Return the _color attribute

    @body.setter
    def body(self, body):
        self._body = body

    def apply_force_middle(self, force):
        # force = rot_matrix(-self.orientation) @ force
        # self._body.apply_force_at_local_point((force[0],force[1]), (0,0))
        self._body.apply_force_at_world_point(
            (force[0], force[1]), self._body.position)

    def apply_force(self, force: np.array, global_coords=True):

        if len(force) == 4:
            direction = force[:2].tolist()
            pos = force[2:].tolist()
        elif len(force) == 2:
            direction = force.tolist()
            pos = [0, 0]
        else:
            raise ValueError("Force must be 4 or 2 element array")

        if global_coords:
            rm = rot_matrix(-self.orientation)
            direction = np.dot(rm, direction).tolist()

        self._body.apply_force_at_local_point(direction, pos)

    def add_to_space(self, space: pymunk.Space):
        """
        Simulator provides the space, and the object adds itself to the space
        Pymunk does not allow add same shape to multiple spaces
        :param space:
        :return:
        """

        space.add(self._body)
        for shape in self.shapes:
            shape.density = self._density
            space.add(shape)

    def set_collision_type(self, collision_type):
        for s in self.shapes:
            s.collision_type = collision_type

    def set_ID(self, ID: tuple[int, ...], moveable: bool = True):
        """
        Used by simulator to set the ID of the object, so that it can be identified in the simulator
        :param ID:
        :param moveable:
        :return:
        """
        if moveable:
            self._body.moveId = ID
        else:
            self._body.fixedId = ID

    def get_manual_force(self) -> np.array:
        """
        :return: Force applied to the CoG of given body
        """
        return np.array([self._body.force.x, self._body.force.y])

    def save_manual_forces(self):
        """
        Save the manual forces applied to the body
        :return:
        """
        self._manual_force = np.array([self._body.force.x, self._body.force.y])

    def get_body(self, tup_index: tuple[int, ...]):
        if len(tup_index) != 0:
            raise ValueError("Tried to get subbody of single body object")
        return self

    def link_body(self, body, index: tuple[int, ...]):
        if len(index) != 0:
            raise ValueError("Linking of subbody of single body object")
        self._body = body
        self.shapes = body.shapes
        self._collision_data = None

    def collision_start(self, collsion_data):
        self._collision_data = collsion_data

    def collision_end(self, collsion_data):
        self._collision_data = None

    def collision_clear(self):
        self._collision_data = None

    @position.setter
    def position(self, value: np.array):
        self._body.position = value[0], value[1]

    @orientation.setter
    def orientation(self, value):
        self._body.angle = value

    @velocity.setter
    def velocity(self, value):
        self._body.velocity = value[0], value[1]

    @angular_velocity.setter
    def angular_velocity(self, value):
        self._body.angular_velocity = value

    @friction.setter
    def friction(self, value):
        self._body.friction = value

    @color.setter
    def color(self, color):

        self._color = color
        for s in self.shapes:
            s.color = pygame.Color(color)
