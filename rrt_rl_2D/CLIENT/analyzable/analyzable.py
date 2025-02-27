import time

from rrt_rl_2D.simulator.simulator import Simulator
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG
from rrt_rl_2D.planners.vec_env_planner import VecEnvPlanner


class SimulatorA(Simulator):
    def __init__(self, movable_objects, fixed_objects, config=STANDARD_CONFIG):
        super().__init__(movable_objects, fixed_objects, config)
        self._step_time = 0

    def step(self):
        st = time.perf_counter()
        super().step()
        et = time.perf_counter()
        self._step_time += et - st

    @property
    def step_time(self):
        return self._step_time


class VecEnvPlannerA(VecEnvPlanner):
    def __init__(self, env, model, cfg):
        super().__init__(env, model, cfg)
        self._env_export_time = 0
        self._env_import_time = 0
        self._env_reset_time = 0
        self._env_render_time = 0
        self._env_step_time = 0
        self._collided_cnt = 0
        self._reached_cnt = 0
        self._timeouts_cnt = 0
        self._node_cnt = 0

    def _env_import(self, start, goal):
        st = time.perf_counter()
        super()._env_import(start, goal)
        et = time.perf_counter()
        self._env_import_time += et - st

    def _env_reset(self):
        st = time.perf_counter()
        ret = super()._env_reset()
        et = time.perf_counter()
        self._env_reset_time += et - st
        return ret

    def _env_export(self):
        st = time.perf_counter()
        ret = super()._env_export()
        et = time.perf_counter()
        self._env_export_time += et - st
        return ret

    def _env_render(self):
        st = time.perf_counter()
        super()._env_render()
        et = time.perf_counter()
        self._env_render_time += et - st

    def _env_step(self, action):
        st = time.perf_counter()
        ret = super()._env_step(action)
        et = time.perf_counter()
        self._env_step_time += et - st
        return ret

    def _update_inner_stats(self, exports, data):
        if data['timeout']:
            self._timeouts_cnt += 1
        if data['fail']:
            self._collided_cnt += 1
        if data['reached']:
            self._reached_cnt += 1
        self._node_cnt += len(exports)

    @property
    def env_export_time(self):
        return self._env_export_time

    @property
    def env_import_time(self):
        return self._env_import_time

    @property
    def env_reset_time(self):
        return self._env_reset_time

    @property
    def env_render_time(self):
        return self._env_render_time

    @property
    def env_step_time(self):
        return self._env_step_time

    @property
    def env_times(self):
        return {
            "export": self.env_export_time,
            "import": self.env_import_time,
            "reset": self.env_reset_time,
            "render": self.env_render_time,
            "step": self.env_step_time
        }

    @property
    def result_cnts(self):
        return {
            "collided": self._collided_cnt,
            "reached": self._reached_cnt,
            "timeouts": self._timeouts_cnt
        }

    @property
    def node_cnt(self):
        return self._node_cnt
