from abc import abstractmethod, ABC

from ...RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name
from ...envs import *
from ...utils.seed_manager import init_manager
from ...simulator.standard_config import STANDARD_CONFIG
from ...maps import str2map, RectangleEmpty
from ...node_managers import *
from ...envs.debug_radius import CableRadiusEmpty
from ...envs.blend_env import BlendEnvR, BlendEnvI, BlendStrengthEnvI, BlendStrengthEnvR
from ...envs.rect import RectPIDR, RectPIDI, RectVelEnvI, RectVelEnvR
from ...envs.cable_env import CableInnerAnglesI, CableInnerAnglesR, CableEnvNaiveR, CableEnvNaiveI
from ...envs.cable_env import CablePIDEnvI, CablePIDEnvR, CableBigTestI, CableBigTestR
from ...envs.cable_env import CableEnvR, CableEnvI
from ...envs.last_env import LastEnvR, LastEnvI
from ..analyzable.analyzable import SimulatorA
from ...simulator.simulator import Simulator
from ...envs.dodge_env import DodgeEnvR, DodgeEnvI, DodgeEnvVelI, DodgeEnvVelR, DodgeEnvPenaltyI, DodgeEnvPenaltyR, DodgeEnvReductionI, DodgeEnvReductionR, DodgeEnvReductionVelR, DodgeEnvReductionVelI, DodgeEnvPenaltyReductionI, DodgeEnvPenaltyReductionR


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
        return maker, get_name(type(self).__name__ + '='), {"nm": nm, "cfg": self.cfg}

    def analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = self._manager_helper()

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm, "cfg": self.cfg}


class DebugMaker(_Maker):
    def _resetable_class(self):
        return CableRadiusEmpty

    def _non_resetable_class(self):
        return CableRadiusEmpty

    def first_try(self, **kwargs):
        def raw_maker():
            return self._resetable_class()(render_mode='human')
        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), dict()


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
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

    def two_controllable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = [0, self.cfg['seg_num'] - 1]
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, 600, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

    def analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = ControllableManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 300, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}


class BlendStrengthMaker(BlendMaker):
    def _resetable_class(self):
        return BlendStrengthEnvR

    def _non_resetable_class(self):
        return BlendStrengthEnvI

    def first_try(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        nm = ControllableManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, 600, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}


class StandardCableMaker(_Maker):
    def _resetable_class(self):
        return CableEnvR

    def _non_resetable_class(self):
        return CableEnvI

    def first_try(self, movement_force=500, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = None
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, movement_force, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

    def one_controllable_analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = [0]
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 2000, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

    def two_controllable_analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = [0, self.cfg['seg_num'] - 1]
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 1000, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

    def analyzable(self, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = None
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=SimulatorA)
            return self._resetable_decision()(cur_map, 600, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}


class DodgeEnvMaker(StandardCableMaker):
    def _resetable_class(self):
        return DodgeEnvR

    def _non_resetable_class(self):
        return DodgeEnvI

    def first_try(self, **kwargs):
        return super().first_try(movement_force=500, **kwargs)

    def analyzable(self, **kwargs):
        return super().analyzable(**kwargs)


class DodgeEnvVelMaker(DodgeEnvMaker):
    def _resetable_class(self):
        return DodgeEnvVelR

    def _non_resetable_class(self):
        return DodgeEnvVelI

    def first_try(self, **kwargs):
        return super().first_try(**kwargs)

    def analyzable(self, **kwargs):
        return super().analyzable(**kwargs)


class DodgeEnvPenaltyMaker(DodgeEnvMaker):
    def _resetable_class(self):
        return DodgeEnvPenaltyR

    def _non_resetable_class(self):
        return DodgeEnvPenaltyI


class DodgeEnvReductionMaker(DodgeEnvMaker):
    def _resetable_class(self):
        return DodgeEnvReductionR

    def _non_resetable_class(self):
        return DodgeEnvReductionI


class DodgeEnvReductionVelMaker(DodgeEnvMaker):
    def _resetable_class(self):
        return DodgeEnvReductionVelR

    def _non_resetable_class(self):
        return DodgeEnvReductionVelI


class DodgeEnvPenaltyReductionMaker(DodgeEnvMaker):
    def _resetable_class(self):
        return DodgeEnvPenaltyReductionR

    def _non_resetable_class(self):
        return DodgeEnvPenaltyReductionI


class CableNaiveMaker(StandardCableMaker):
    def _resetable_class(self):
        return CableEnvNaiveR

    def _non_resetable_class(self):
        return CableEnvNaiveI

    def first_try(self, **kwargs):
        return super().first_try(movement_force=500, **kwargs)


class CableBigTestMaker(StandardCableMaker):
    def _resetable_class(self):
        return CableBigTestR

    def _non_resetable_class(self):
        return CableBigTestI

    def first_try(self, **kwargs):
        return super().first_try(movement_force=300, **kwargs)


class CableInnerAnglesMaker(StandardCableMaker):
    def _resetable_class(self):
        return CableInnerAnglesR

    def _non_resetable_class(self):
        return CableInnerAnglesI


class PIDCableMaker(StandardCableMaker):
    def _resetable_class(self):
        return CablePIDEnvR

    def _non_resetable_class(self):
        return CablePIDEnvI

    def first_try(self, **kwargs):
        return super().first_try(movement_force=80, **kwargs)


class RectMaker(_Maker):
    def _resetable_class(self):
        return RectEnvR

    def _non_resetable_class(self):
        return RectEnvI

    def first_try(self, force_strength=1000, **kwargs):
        class Rect(RectangleEmpty, self._map_helper()):
            pass

        cur_map_cls = Rect

        self.cfg = self._cfg_helper()
        nm = VelNodeManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, force_strength, nm, render_mode=self.render_mode)
        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

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
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}


class RectPIDMaker(_Maker):
    def _resetable_class(self):
        return RectPIDR

    def _non_resetable_class(self):
        return RectPIDI

    def first_try(self, **kwargs):
        class Rect(RectangleEmpty, self._map_helper()):
            pass

        cur_map_cls = Rect

        self.cfg = self._cfg_helper()
        nm = VelNodeManager(self.cfg)

        def raw_maker():
            cur_map = cur_map_cls(self.cfg)
            return self._resetable_decision()(cur_map, 80, nm, render_mode=self.render_mode)
        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}


class RectVelMaker(RectMaker):
    def _resetable_class(self):
        return RectVelEnvR

    def _non_resetable_class(self):
        return RectVelEnvI

    def first_try(self, max_velocity=1000, **kwargs):
        return super().first_try(force_strength=max_velocity, **kwargs)


class LastEnvMaker(_Maker):
    def _resetable_class(self):
        return LastEnvR

    def _non_resetable_class(self):
        return LastEnvI

    def first_try(self, movement_force=450, sim_cls=Simulator, **kwargs):
        cur_map_cls = self._map_helper()
        self.cfg = self._cfg_helper()
        ctrl_idxs = None
        nm = ControllableManager(self.cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = self.cfg['threshold']

        def raw_maker():
            cur_map = cur_map_cls(self.cfg, sim_cls=sim_cls)
            return self._resetable_decision()(cur_map, movement_force, nm, render_mode=self.render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(type(self).__name__ + '='), {"nm": nm}

    def analyzable(self, **kwargs):
        return self.first_try(sim_cls=SimulatorA, **kwargs)
