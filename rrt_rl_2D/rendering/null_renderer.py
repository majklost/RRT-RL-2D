from typing import TYPE_CHECKING
from .base_renderer import BaseRenderer
if TYPE_CHECKING:
    from ..simulator.simulator import Simulator


class NullRenderer(BaseRenderer):
    def render(self, simulator: 'Simulator'):
        pass

    def register_callback(self, clb):
        pass

    def close(self):
        pass
