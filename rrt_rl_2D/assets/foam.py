import numpy as np
import warnings
import pymunk


from ..utils.common_utils import rot_matrix
from .multibody import MultiBody
from .rectangle import Rectangle


class Foam(MultiBody):
    def __init__(self, track_colisions=False, ignore_neighbour_collision=True):
        super().__init__(track_colisions, ignore_neighbour_collision)
