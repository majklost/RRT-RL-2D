from ..RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name
from ..envs import *
from ..utils.seed_manager import init_manager
from ..simulator.standard_config import STANDARD_CONFIG
from ..maps import str2map
from ..node_managers import *
from ..envs.debug_radius import CableRadiusEmpty
from ..envs.blend_env import BlendEnvR, BlendEnvI
from ..envs.cable_env import CableEnvR, CableEnvI
"""
Here you can build your environments so they can be easily imported everywhere
(for learning, for rrt planning, for saved path replaying, for replaying learned RL models... )
Each function should return a tuple of (env_maker, name, objects)
Each function should have the same signature as the ones below
def obs_vel_stronger_fast(map_name='AlmostEmpty', cfg=None, render_mode='human', resetable=False, **kwargs):

Function should not be changed after using, some experiments might depend on them


"""


def _cfg_map_helper(cfg, map_name):
    if cfg is None:
        cfg = STANDARD_CONFIG.copy()

    if map_name is None:
        map_name = 'AlmostEmpty'
    cur_map_cls = str2map[map_name]
    return cur_map_cls, cfg


def _manager_helper(cfg, resetable):
    if resetable:
        return NodeManager(cfg)
    else:
        return VelNodeManager(cfg)


class CableRadiusMaker:
    @staticmethod
    def first_try(map_name='AlmostEmpty', cfg=None, render_mode='human', resetable=False, **kwargs):
        # kwargs used to be compatible with other makers
        was_none = cfg is None

        map_cls, cfg = _cfg_map_helper(cfg, map_name)
        if was_none:
            cfg['threshold'] = cfg['cable_length'] / 2

        nm = _manager_helper(cfg, resetable)

        def raw_maker():
            cur_map = map_cls(cfg)
            if resetable:
                return CableRadiusNearestObsVelR(cur_map, 300, nm, render_mode=render_mode)
            else:
                return CableRadiusNearestObsVelI(cur_map, 300, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}

    # @staticmethod
    # def obs_vel_stronger_stones(map_name='StandardStones', cfg=None, render_mode=None, resetable=True):
    #     return CableRadiusMaker.obs_vel_stronger_fast(map_name, cfg, render_mode, resetable)


class DebugMaker:
    @staticmethod
    def debug_radius_fast(**kwargs):

        def raw_maker():
            return CableRadiusEmpty(render_mode='human')

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), dict()


class BlendMaker:
    @staticmethod
    def first_try(map_name, cfg, render_mode='human', resetable=False, **kwargs):
        was_none = cfg is None

        map_cls, cfg = _cfg_map_helper(cfg, map_name)
        if was_none:
            cfg['threshold'] = 20

        ctrl_idxs = None
        nm = ControllableManager(cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = cfg['threshold']

        def raw_maker():
            cur_map = map_cls(cfg)
            if resetable:
                return BlendEnvR(cur_map, 600, nm, render_mode=render_mode)
            else:
                return BlendEnvI(cur_map, 600, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}

    # @staticmethod
    # def standard_stones(map_name='StandardStones', cfg=None, render_mode=None, resetable=True):
    #     return BlendMaker.first_try(map_name, cfg, render_mode, resetable)


class StandardCableMaker:
    @staticmethod
    def first_try(map_name, cfg, render_mode='human', resetable=False, **kwargs):
        map_cls, cfg = _cfg_map_helper(cfg, map_name)
        ctrl_idxs = None
        nm = ControllableManager(cfg, ctrl_idxs=ctrl_idxs)
        nm.wanted_threshold = cfg['threshold']

        def raw_maker():
            cur_map = map_cls(cfg)
            if resetable:
                return CableEnvR(cur_map, 600, nm, render_mode=render_mode)
            else:
                return CableEnvI(cur_map, 600, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '='), {"nm": nm}
