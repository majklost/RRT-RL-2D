import time
import numpy as np
# Return a path on map where rectangle reaches from start to end point
from ..simulator.standard_config import StandardConfig
from ..manual_models import BaseManualModel
from ..RL.training_utils import create_multi_env
from ..samplers import NDIMSampler
from ..storages import GNAT
from ..nodes import GoalNode
from ..storage_wrappers import RectEndWrapper
from ..planners import VecEnvPlanner, PlannerResponse


class _LinearModel(BaseManualModel):
    def predict(self, obs, **kwargs):
        return obs, None


class RectPath:
    def __init__(self, maker, node_manager, cfg: StandardConfig, distance_fnc, max_steps=100_000):
        self.maker = maker
        self.cfg = cfg
        self.node_manager = node_manager
        self.distance_fnc = distance_fnc
        self.model = _LinearModel()
        self._time_taken = 0
        self.max_steps = max_steps

    def get_path(self):
        env = create_multi_env(self.maker, 1, normalize=False)
        sampler = NDIMSampler((0, 0), (self.cfg["width"], self.cfg["height"]))
        storage = GNAT(self.distance_fnc)
        overall_goal = self._overall_goal()
        s_wrapper = RectEndWrapper(
            storage, self.distance_fnc, overall_goal, self.cfg)
        planner = VecEnvPlanner(env, self.model, self.cfg)
        start_node = env.env_method("export_state")
        response = PlannerResponse(start_node, {})
        s_wrapper.save_to_storage(response)

        start_time = time.time()
        iter_cnt = 0
        while planner.step_cnt < self.max_steps:
            if not s_wrapper.want_next_iter:
                reachead_goal = True
                print("Heuristics Goal reached")
                break
            qrand_raw = sampler.sample()
            self.node_manager.wanted_position = qrand_raw
            qrand = self.node_manager.create_goal()
            nearest = s_wrapper.get_nearest(qrand)
            response = planner.check_path(nearest, qrand)

            s_wrapper.save_to_storage(response)
            iter_cnt += 1
            if iter_cnt % 100 == 0:
                print("Heur Iterations: ", iter_cnt)
                print("Heur steps sum: ", planner.step_cnt)
                self._time_taken = time.time() - start_time
        self._time_taken = time.time() - start_time
        return s_wrapper.get_path()

    def _overall_goal(self):
        return GoalNode(
            (self.cfg['width'] - 200, self.cfg['height'] // 2), threshold=250)

    @property
    def time_taken(self):
        return self._time_taken
