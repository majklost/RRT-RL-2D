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
    'seed_plan': None
}
