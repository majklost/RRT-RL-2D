import numpy as np
import warnings
import pymunk


from ..utils.common_utils import rot_matrix
from .multibody import MultiBody
from .rectangle import Rectangle


class Cable(MultiBody):
    def __init__(self, pos, length, num_links, thickness, angle=0, max_angle=np.pi / 2):
        super().__init__()
        self.angle = angle
        self.num_links = num_links
        self.thickness = thickness
        self.max_angle = max_angle
        self.density = 0.005
        # segments length does not sum up to LENGTH !!!
        self.segment_length = length / self.num_links
        self.pivots = []
        self.angular_springs = []
        # the length of the empty space between segments so cable can bend
        self.empty_len = self.thickness / \
            (3 * np.tan((np.pi - self.max_angle) / 2))
        print(f"empty len: {self.empty_len}, thickness: {self.thickness}")

        if self.empty_len > self.thickness:
            warnings.warn(
                "Empty length is greater than thickness, cable can stuck in itself")

        self.length = length + 2 * (self.num_links - 1) * self.empty_len
        self._create_cable(pos)
        self.track_colisions = True

    def _create_cable(self, pos):
        self._create_objects(pos)
        self._create_pivots()

    def _create_objects(self, pos):
        rm = rot_matrix(self.angle)
        for i in range(self.num_links):
            r = Rectangle(pos + rm @ np.array([i * self.segment_length, 0]), self.segment_length, self.thickness,
                          pymunk.Body.DYNAMIC)
            r.orientation = self.angle + np.pi
            self.append(r)

    def _create_pivots(self):
        x = self.empty_len
        # print(x)
        for i in range(self.num_links - 1):
            pivot = pymunk.constraints.PivotJoint(
                self.bodies[i + 1].body, self.bodies[i].body, (x + self.segment_length / 2, 0), (-x - self.segment_length / 2, 0))
            self.pivots.append(pivot)

    def add_to_space(self, space):
        super().add_to_space(space)
        space.add(*self.pivots)
        space.add(*self.angular_springs)

    @property
    def position(self):
        """Returns position of all segments"""
        return np.array([b.position for b in self.bodies])

    @property
    def velocity(self):
        """Returns vector of velocities of all segments"""
        return np.array([b.velocity for b in self.bodies])

    @velocity.setter
    def velocity(self, vel):
        assert len(vel) == len(self.bodies)
        for i in range(len(self.bodies)):
            self.bodies[i].velocity = vel[i]

    @position.setter
    def position(self, pos):
        assert len(pos) == len(self.bodies)
        prev = pos[0]
        for i in range(len(self.bodies)):
            self.bodies[i].position = pos[i]
            diff = pos[i] - prev
            angle = np.arctan2(diff[1], diff[0])
            self.bodies[i].orientation = angle + np.pi
            prev = pos[i]
            self.bodies[i].velocity = (0, 0)

    @property
    def orientation(self):
        return np.array([b.orientation for b in self.bodies])

    @orientation.setter
    def orientation(self, angle):
        for i in range(len(self.bodies)):
            self.bodies[i].orientation = angle[i]

    def glob2loc(self, vec):
        transposed = False
        # converts global vector to local vector (rotated by angle)
        assert len(vec.shape) == 2, "Input vector to location must be 2D"
        if vec.shape[0] == 2:
            assert vec.shape[1] == len(
                self.bodies), "Input vector must have the same number of elements as cable segments"
            vec = vec.T
            transposed = True

        if vec.shape[0] == len(self.bodies):
            assert vec.shape[1] == 2, "Input vector must have 2 elements"

        rot_matrices = self._generate_rotations(-self.orientation)
        results = rot_matrices @ vec[:, :, None]
        if transposed:
            return results.squeeze().T
        return results.squeeze()

    def loc2glob(self, vec):
        transposed = False
        # converts local vector to global vector (rotated by angle)
        assert len(vec.shape) == 2, "Input vector to location must be 2D"
        if vec.shape[0] == 2:
            assert vec.shape[1] == len(
                self.bodies), "Input vector must have the same number of elements as cable segments"
            vec = vec.T
            transposed = True

        if vec.shape[0] == len(self.bodies):
            assert vec.shape[1] == 2, "Input vector must have 2 elements"
        rot_matrices = self._generate_rotations(self.orientation)
        results = rot_matrices @ vec[:, :, None]
        if transposed:
            return results.squeeze().T
        return results.squeeze()

    def _generate_rotations(self, orient):
        orientations = orient
        # np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        cos = np.cos(orientations)
        sin = np.sin(orientations)
        rot_matrices = np.array([[cos, -sin], [sin, cos]])
        return np.transpose(rot_matrices, (2, 0, 1))

    def angles_between(self):
        # returns angles between segments
        orientations = self.orientation
        diffs = orientations[1:] - orientations[:-1]
        return diffs
