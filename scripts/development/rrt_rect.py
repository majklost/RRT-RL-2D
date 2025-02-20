import numpy as np
from rrt_rl_2D import *

node_manager = node_managers.VelNodeManager()


def distance_fnc(n1, n2):
    if n1 is GoalNode:
        if n2 is GoalNode:
            raise ValueError("Both nodes are goal nodes in comparision.")
        else:
            pass

    elif n2 is GoalNode:
        pass
    return np.linalg.norm(n1.position - n2.position)


storage = storages.GNAT()
