import numpy as np
import warnings
import pymunk
from typing import Tuple

from ..utils.common_utils import rot_matrix
from .multibody import MultiBody
from .rectangle import Rectangle
from .circle import Circle
from .spring_params import STANDARD_StructuralSpringParams, STANDARD_ShearSpringParams


class Foam(MultiBody):

    def __init__(self, position: np.array, dimensions: np.array, masspoint_per_length, mass_radius=5, structStringParams=STANDARD_StructuralSpringParams, shearSpringParams=STANDARD_ShearSpringParams, **kwargs):
        super().__init__(**kwargs)
        self.mass_radius = mass_radius
        self.density = 0.005
        self.structStringParams = structStringParams
        self.shearSpringParams = shearSpringParams
        width, height = dimensions
        self.row_mass_num = int(width * masspoint_per_length)
        self.col_mass_num = int(height * masspoint_per_length)
        self.structStringParams = structStringParams
        self.shearSpringParams = shearSpringParams
        print(f"Foam: {self.row_mass_num} x {self.col_mass_num} mass points")
        self.sep_row = width / self.row_mass_num
        self.sep_col = height / self.col_mass_num

        assert self.sep_row > self.mass_radius, "Mass points are too close to each other in row direction"
        assert self.sep_col > self.mass_radius, "Mass points are too close to each other in column direction"
        assert self.sep_row > 0, "Mass points are too close to each other in row direction"
        assert self.sep_col > 0, "Mass points are too close to each other in column direction"
        assert self.row_mass_num > 1, "Zero mass points in row direction"
        assert self.col_mass_num > 1, "Zero mass points in column direction"

        self.structSprings = []
        self.shearSprings = []
        self._create_foam(position, dimensions)

    def _create_foam(self, position, dimensions):
        self._create_bodies(position, dimensions)
        self._create_springs()
        self._create_shear_springs()

    def _create_bodies(self, position, dimensions):
        width, height = dimensions
        # Create the mass points
        for i in range(self.row_mass_num):
            for j in range(self.col_mass_num):
                x = position[0] + i * self.sep_row
                y = position[1] + j * self.sep_col
                mass_point = Circle(
                    np.array([x, y]), self.mass_radius, pymunk.Body.DYNAMIC)
                self.append(mass_point)

    def _create_springs(self):
        # Create the structural springs
        for i in range(self.row_mass_num):
            for j in range(self.col_mass_num - 1):
                mass_point1 = self.bodies[i * self.col_mass_num + j]
                mass_point2 = self.bodies[i * self.col_mass_num + j + 1]
                spring = pymunk.constraints.DampedSpring(
                    mass_point1.body, mass_point2.body, (0, 0), (0, 0), self.sep_row, self.structStringParams.stiffness, self.structStringParams.damping)
                self.structSprings.append(spring)

        for i in range(self.row_mass_num - 1):
            for j in range(self.col_mass_num):
                mass_point1 = self.bodies[i * self.col_mass_num + j]
                mass_point2 = self.bodies[(i + 1) * self.col_mass_num + j]
                spring = pymunk.constraints.DampedSpring(
                    mass_point1.body, mass_point2.body, (0, 0), (0, 0), self.sep_col, self.structStringParams.stiffness, self.structStringParams.damping)
                self.structSprings.append(spring)

    def _create_shear_springs(self):
        # Create the shear springs
        for i in range(self.row_mass_num - 1):
            for j in range(self.col_mass_num - 1):
                mass_point1 = self.bodies[i * self.col_mass_num + j]
                mass_point2 = self.bodies[(
                    i + 1) * self.col_mass_num + (j + 1)]
                spring = pymunk.constraints.DampedSpring(
                    mass_point1.body, mass_point2.body, (0, 0), (0, 0), np.sqrt(self.sep_row**2 + self.sep_col**2), self.shearSpringParams.stiffness, self.shearSpringParams.damping)
                self.shearSprings.append(spring)

        for i in range(self.row_mass_num - 1):
            for j in range(1, self.col_mass_num):
                mass_point1 = self.bodies[i * self.col_mass_num + j]
                mass_point2 = self.bodies[(
                    i + 1) * self.col_mass_num + (j - 1)]
                spring = pymunk.constraints.DampedSpring(
                    mass_point1.body, mass_point2.body, (0, 0), (0, 0), np.sqrt(self.sep_row**2 + self.sep_col**2), self.shearSpringParams.stiffness, self.shearSpringParams.damping)
                self.shearSprings.append(spring)

    def add_to_space(self, space):
        super().add_to_space(space)
        space.add(*self.structSprings)
        space.add(*self.shearSprings)
