from typing import TypedDict, Tuple


class StandardConfig(TypedDict):
    width: int
    height: int
    fps: int
    gravity: Tuple[int, int]
    damping: float
    cable_length: int
    seg_num: int
    seed_env: int | None
    seed_plan: int | None
    check_dimensions: Tuple[int, int] | int
    max_steps: int
    checkpoint_period: int  # Period for saving checkpoints
    threshold: int  # Threshold for reaching the goal


# Standard configuration for the environment
STANDARD_CONFIG: StandardConfig = {
    'width': 1900,
    'height': 800,
    'fps': 60,
    'gravity': (0, 0),
    'damping': 0.15,
    'cable_length': 300,
    'seg_num': 20,
    'seed_env': None,
    'seed_plan': None,
    'check_dimensions': (10, 10),
    'max_steps': 1000,
    'checkpoint_period': 200,
    'threshold': 200
}
