from pathlib import Path
from typing import TypedDict
import json
import datetime


class EnvTimes(TypedDict):
    """
    Times spent in the environment.
    """
    reset_t: float  # time spent in reset
    step_t: float  # time spent in step
    render_t: float  # time spent in render
    export_t: float  # time spent in export
    import_t: float  # time spent in import


class StandardAnalytics(TypedDict):
    """
    Standard analytics.
    """
    finished: bool  # whether the goal was reached
    iterations: int  # number of RRT iterations
    steps_sum: int  # sum of steps of simulator from all iterations
    tot_time: float  # total time of all iterations
    sim_time: float  # total time of simulator from all iterations
    env_times: EnvTimes  # total time spent in gym env
    storage_time: float  # total time spent in storage (NN search...)
    collided_cnt: int  # number of iterations where collision was detected
    reached_cnt: int  # number of iterations where local goal was reached
    timeout_cnt: int  # number of iterations where timeout was reached
    node_cnt: int  # number of nodes in the tree
    path_node_cnt: int  # number of nodes in the path
    additional: dict  # additional data MUST be serializable
    creation_date: str = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")  # creation date


def save(analytics, path, fname: str):
    """
    Saves the analytics to the path.
    """
    with open(Path(path) / f"{fname}.analytics", 'w') as f:
        json.dump(analytics, f)
