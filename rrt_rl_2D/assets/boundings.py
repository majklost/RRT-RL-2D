import pymunk

from .multibody import MultiBody
from .rectangle import Rectangle


class Boundings(MultiBody):
    def __init__(self, width, height, btype=pymunk.Body.STATIC):
        super().__init__()
        self.width = width
        self.height = height
        self._btype = btype
        self._create_boundings()

    def _create_boundings(self):
        self.append(Rectangle([self.width / 2, 0],
                    self.width, 20, self._btype))
        self.append(Rectangle([self.width / 2, self.height],
                    self.width, 20, self._btype))
        self.append(Rectangle([0, self.height / 2],
                    20, self.height, self._btype))
        self.append(Rectangle([self.width, self.height / 2],
                    20, self.height, self._btype))
