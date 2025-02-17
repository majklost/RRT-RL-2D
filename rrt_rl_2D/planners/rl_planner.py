from .base_planner import BasePlanner


class RLPlanner(BasePlanner):
    """
    A planner with RL policy to find the path in environment.
    """

    def __init__(self, env, model):
        super().__init__(env)
        self.model = model

    def check_path(self, start, goal):
        """
        Checks if a path exists between the start and goal nodes.
        """
        self.env.import_start(start)
        self.env.import_goal(goal)
