from .base_planner import BasePlanner
from ..envs.rrt_env import RRTEnv


class ManualPlannerRect(BasePlanner):
    """
    A planner with manual policy to find the path in environment.
    """

    def __init__(self, env: RRTEnv):
        super().__init__(env)

    def check_path(self, start, goal):
        self.env.import_start(start)
        self.env.import_goal(goal)




class ManualPlannerCableStandard(BasePlanner):
    """
    A planner with manual policy to find the path in environment.
    """

    def __init__(self, env):
        super().__init__(env)

    def check_path(self, start, goal):
        pass


class ManualPlannerCableRadius(BasePlanner):
    pass
