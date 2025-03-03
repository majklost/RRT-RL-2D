import numpy as np
from rrt_rl_2D.utils.rect_path import RectPath
from rrt_rl_2D.CLIENT.makers import RectMaker
from rrt_rl_2D import *
from pathlib import Path
MAP_NAME = "ThickStones"


cfg = STANDARD_CONFIG.copy()
cfg['seed_env'] = 50
cfg['threshold'] = 20
factory = RectMaker(MAP_NAME, cfg)
maker, maker_name, stuff = factory.first_try(force_strength=2000)
node_manager = stuff['nm']


def distance_fnc(n1, n2):

    if isinstance(n1, GoalNode):
        if isinstance(n2, GoalNode):
            raise ValueError("Both nodes are goal nodes in comparision.")
        else:
            return np.linalg.norm(n2.agent_pos - n1.goal)

    elif isinstance(n2, GoalNode):
        return np.linalg.norm(n1.agent_pos - n2.goal)
    return np.linalg.norm(n1.agent_pos - n2.agent_pos)


rp = RectPath(maker, node_manager, cfg, distance_fnc)


path = rp.get_path()
filepath = Path(__file__).parent
for n in path.nodes:
    print(n.agent_pos)
VelPathSaver(maker_name, path, cfg, MAP_NAME,
             script_name=__file__, data={}).save(filepath, "rect_path")


spr = SavePathReplayer(filepath / "rect_path.rpath")
spr.replay()
