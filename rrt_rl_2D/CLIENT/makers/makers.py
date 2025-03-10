from abc import abstractmethod, ABC

from ...RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name
from ...envs import *
from ...utils.seed_manager import init_manager
from ...simulator.standard_config import STANDARD_CONFIG
from ...maps import str2map, RectangleEmpty
from ...node_managers import *
from ...envs.debug_radius import CableRadiusEmpty
from ...envs.blend_env import BlendEnvR, BlendEnvI
from ...envs.cable_env import CableEnvR, CableEnvI
from ..analyzable.analyzable import SimulatorA
"""
Here you can build your environments so they can be easily imported everywhere
(for learning, for rrt planning, for saved path replaying, for replaying learned RL models... )
Each function should return a tuple of (env_maker, name, objects)
Each function should have the same signature as the ones below
def obs_vel_stronger_fast(map_name='AlmostEmpty', cfg=None, render_mode='human', resetable=False, **kwargs):

Function should not be changed after using, some experiments might depend on them


"""


class _Maker(ABC):
    def __init__(self, map_name, cfg, render_mode='human', resetable=False, **kwargs):
        self.map_name = map_name
        self.cfg = cfg
        self.was_cfg_none = cfg is None
        self.render_mode = render_mode
        self.resetable = resetable

    def _manager_helper(self):
        if self.resetable:
            return NodeManager(self.cfg)
        else:
            return VelNodeManager(self.cfg)

    def _cfg_helper(self):
        if self.cfg is None:
            self.cfg = STANDARD_CONFIG.copy()
        return self.cfg

    def _map_helper(self):

        if self.map_name is None:
            self.map_name = 'AlmostEmpty'
        cur_map_cls = str2map[self.map_name]
        return cur_map_cls

    @abstractmethod
    def _resetable_class(self):
        raise NotImplementedError("Depends on the specific maker")

    @abstractmethod
    def _non_resetable_class(self):
        raise NotImplementedError("Depends on the specific maker")

    def _resetable_decision(self):
        if self.resetable:
            return self._resetable_class()
        else:
            return self._non_resetable_class()

    @abstractmethod
    def first_try(self):
        raise NotImplementedError("Depends on the specific maker")


class CableRadiusMaker(_Maker):

    def _resetable_class(self):
        return CableRadiusNearestObsVelR

    def _non_resetable_class(self):
        return CableRadiusNearestObsVelI

    def first_try(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = self._manager_helper()

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            if self.was_cfg_none:
                self.cfg['threshold'] = cur_map.agent.length / 2

            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm, "cfg": self.cfg}

    def analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = self._manager_helper()

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm, "cfg": self.cfg}


class DebugMaker(_Maker):
    def _resetable_class(self):
        return CableRadiusEmpty

    def _non_resetable_class(self):
        return CableRadiusEmpty

    def first_try(self, **kwargs):
        def raw_maker():
            return self._resetable_class()(render_mode='human')
        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), dict()


class BlendMaker(_Maker):
    def _resetable_class(self):
        return BlendEnvR

    def _non_resetable_class(self):
        return BlendEnvI

    def _cfg_helper(self):
        if self.cfg is None:
            self.cfg = STANDARD_CONFIG.copy()
            self.cfg['threshold'] = 20
        return self.cfg

    def first_try(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = ControllableManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, 600, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}

    def analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = ControllableManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}


class StandardCableMaker(_Maker):
    def _resetable_class(self):
        return CableEnvR

    def _non_resetable_class(self):
        return CableEnvI

    def first_try(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = None
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}

    def analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = None
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}


class RectMaker(_Maker):
    def _resetable_class(self):
        return RectEnvR

    def _non_resetable_class(self):
        return RectEnvI

    def first_try(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = VelNodeManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, 1000, nm, render_mode=self.render_mode)
        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}

    def analyzable(self, **kwargs):
        class Rect(RectangleEmpty, self._map_helper()):
            pass

        cur_map_cls = Rect

        self.cfg = self._cfg_helper()
        nm = VelNodeManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 1000, nm, render_mode=self.render_mode)
        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}
