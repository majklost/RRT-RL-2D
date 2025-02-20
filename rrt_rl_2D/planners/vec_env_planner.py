from stable_baselines3.common.env_util import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from .base_planner import BasePlanner, PlannerResponse
from ..simulator.standard_config import StandardConfig
from ..node_managers.node_manager import NodeManager
from ..manual_models.base_model import BaseManualModel


class VecEnvPlanner(BasePlanner):
    """
    A planner that uses VecEnv to find the path in environment.
    Assuming that the environment is prepared with norms.
    Assuming that number of environments is 1.
    Assumming that info dict contain booleans fail and reached.
    """

    def __init__(self, env: VecEnv, model: OnPolicyAlgorithm | BaseManualModel, cfg: StandardConfig):
        super().__init__(env)
        self.model = model
        self.cfg = cfg
        assert env.num_envs == 1, "Planner only supports single environment"

    def check_path(self, start, goal) -> PlannerResponse:
        self.env.env_method("import_goal", goal)
        self.env.env_method("import_start", start)

        obs = self.env.reset()
        max_steps = self.cfg['max_steps']
        period = self.cfg['checkpoint_period']
        exports = []
        data = dict(reached=False)

        for i in range(max_steps):
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if done:
                if info[0]['fail']:
                    break
                else:
                    exports.append(self.env.env_method("export_state")[0])
                    data['reached'] = True
                    break
            elif i + 10 % period == 0:
                exports.append(self.env.env_method("export_state")[0])
        return PlannerResponse(exports, data)
