from pathlib import Path

from rrt_rl_2D.utils.save_manager import delete_experiment, load_manager

EXPERIMENTS_FOLDER = Path(__file__).parent.parent / "experiments" / 'RL'
EXPERIMENTS_FOLDER.mkdir(parents=True, exist_ok=True)

load_manager(EXPERIMENTS_FOLDER)

# print(list(EXPERIMENTS_FOLDER.iterdir()))
