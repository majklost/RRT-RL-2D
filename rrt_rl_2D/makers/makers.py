from ..RL.training_utils import standard_wrap, create_multi_env, create_callback_list, get_name
from ..envs import *
from ..utils.seed_manager import init_manager
from ..simulator.standard_config import STANDARD_CONFIG
from ..maps import *
from ..node_managers import NodeManager


class CableRadius:
    @staticmethod
    def obs_vel_stronger_fast(cur_map=None, render_mode=None, cur_cls=CableRadiusNearestObsVelR):

        cfg = STANDARD_CONFIG.copy()
        cfg['threshold'] = cfg['cable_length'] / 2
        if cur_map is None:
            cur_map = AlmostEmpty(cfg)
        nm = NodeManager(cfg)

        def raw_maker():
            return cur_cls(cur_map, 600, nm, render_mode=render_mode)

        maker = standard_wrap(raw_maker, max_episode_steps=1000)
        return maker, get_name(__class__.__name__ + '=')
