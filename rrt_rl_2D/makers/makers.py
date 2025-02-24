from ..RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name
from ..envs import *
from ..utils.seed_manager import init_manager
from ..simulator.standard_config import STANDARD_CONFIG
from ..maps import *
from ..node_managers import NodeManager
from ..envs.debug_radius import CableRadiusEmpty
from ..envs.blend_env import BlendEnvR, BlendEnvI


class CableRadius:
    @staticmethod
    def obs_vel_stronger_fast(cur_map_maker=None, render_mode=None, resetable=True):

        cfg = STANDARD_CONFIG.copy()
        cfg['threshold'] = cfg['cable_length'] / 2
        nm = NodeManager(cfg)

        def raw_maker():
            if cur_map_maker is None:
                cur_map = AlmostEmpty(cfg)
            else:
                cur_map = cur_map_maker()
            if resetable:
                return CableRadiusNearestObsVelR(cur_map, 600, nm, render_mode=render_mode)
            else:
                return CableRadiusNearestObsVelI(cur_map, 600, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '=')

    @staticmethod
    def obs_vel_stronger_stones(cur_map_maker=None, render_mode=None, resetable=True):

        cfg = STANDARD_CONFIG.copy()
        cfg['threshold'] = cfg['cable_length'] / 2
        nm = NodeManager(cfg)

        def raw_maker():
            if cur_map_maker is None:
                cur_map = StandardStones(cfg)
            else:
                cur_map = cur_map_maker()

            if resetable:
                return CableRadiusNearestObsVelR(cur_map, 600, nm, render_mode=render_mode)
            else:
                return CableRadiusNearestObsVelI(cur_map, 600, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '=')


class Debug:
    @staticmethod
    def debug_radius_fast(**kwargs):

        def raw_maker():
            return CableRadiusEmpty(render_mode='human')

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '=')


class Blend:
    @staticmethod
    def first_try(cur_map_maker=None, render_mode=None, resetable=True):

        cfg = STANDARD_CONFIG.copy()
        cfg['threshold'] = 10
        nm = NodeManager(cfg)

        def raw_maker():
            if cur_map_maker is None:
                cur_map = AlmostEmpty(cfg)
            else:
                cur_map = cur_map_maker()

            if resetable:
                return BlendEnvR(cur_map, 600, nm, render_mode=render_mode)
            else:
                return BlendEnvI(cur_map, 600, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '=')

    @staticmethod
    def standard_stones(raw_maker=None, render_mode=None, resetable=True):
        if raw_maker is None:
            cfg = STANDARD_CONFIG.copy()
            cfg['threshold'] = 10
            nm = NodeManager(cfg)

            def raw_maker():
                cur_map = StandardStones(cfg)

                if resetable:
                    return BlendEnvR(cur_map, 600, nm, render_mode=render_mode)
                else:
                    return BlendEnvI(cur_map, 600, nm, render_mode=render_mode)
            raw_maker = raw_maker

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '=')
