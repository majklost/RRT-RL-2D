from .base_planner import BasePlanner


class ManualPlanner(BasePlanner):
    """
    A planner with manual policy to find the path in environment.
    """

    def __init__(self, env):
        super().__init__(env)
        raise NotImplementedError("Manual planner is not implemented yet")
