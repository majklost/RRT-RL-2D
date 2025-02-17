from .base_planner import BasePlanner


class LearnablePlanner(BasePlanner):
    """
    A planner that learns even during the planning process.
    """

    def __init__(self, env):
        raise NotImplementedError("Learnable planner is not implemented yet")
