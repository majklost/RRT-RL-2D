from .base_planner import BasePlanner
from ..simulator.standard_config import StandardConfig
from stable_baselines3.common.env_util import VecEnv


class VecEnvPlanner(BasePlanner):
    """
    A planner that uses VecEnv to find the path in environment.
    Assuming that the environment is prepared with norms.
    Assuming that number of environments is 1.
    Assumming that info dict contain booleans fail and reached.
    """

    def __init__(self, env: VecEnv, model, cfg: StandardConfig):
        super().__init__(env)
        self.model = model
        self.cfg = cfg
        assert env.num_envs == 1, "Planner only supports single environment"

    def check_path(self, start, goal):
        self.env.env_method("import_goal", goal)
        self.env.env_method("import_start", start)

        obs = self.env.reset()
        max_steps = self.cfg['max_steps']
        period = self.cfg['checkpoint_period']

        for i in range(max_steps):
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if done:
                break

            # class ManualPlannerCableStandard(BasePlanner):
            #     """
            #     A planner with manual policy to find the path in environment.
            #     """

            #     def __init__(self, env):
            #         super().__init__(env)

            #     def check_path(self, start, goal):
            #         pass

            # class ManualPlannerCableRadius(BasePlanner):
            #     pass
