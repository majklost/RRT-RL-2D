import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from stable_baselines3 import PPO


from rrt_rl_2D.CLIENT import *
from rrt_rl_2D.utils.save_manager import consistency_check, get_run_paths, load_manager
from rrt_rl_2D.utils.seed_manager import init_manager
from rrt_rl_2D.RL.players import play_model
from rrt_rl_2D.rendering import EnvRenderer
from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG


EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "experiments" / 'RL'
parser = ArgumentParser(prog="play_experiment",
                        description="Play a model on the environment.")
parser.add_argument("experiment_name", type=str)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--run', type=int, default=-1)
parser.add_argument('--maker_name', type=str, default=None)
parser.add_argument('--experiments_path', type=str, default=EXPERIMENTS_PATH)
parser.add_argument('--weights', choices=['best', 'last'], default='best')
parser.add_argument('--map', type=str, default=None)

args = parser.parse_args()
load_manager(args.experiments_path)
consistency_check()
experiment = get_run_paths(args.experiment_name, args.run)

if args.maker_name is not None:
    maker_name = args.maker_name
else:
    maker_name = experiment['maker_name']

if args.seed == -1:
    args.seed = np.random.randint(0, 100)
else:
    init_manager(args.seed + 10, args.seed + 12)

print(maker_name)
my_cls = maker_name.split('=')[0]
method_name = maker_name.split('=')[1]

map_name = args.map
if map_name is None:
    map_name = experiment['data'].get('map_name', 'AlmostEmpty')

cfg = experiment['data'].get('cfg', None)

maker_factory = globals()[my_cls](
    render_mode='human', map_name=map_name, cfg=cfg, resetable=True)
print(maker_factory.__class__.__name__)
maker, maker_name, objects = getattr(maker_factory, method_name)()

cfg = objects.get('cfg', STANDARD_CONFIG.copy())

renderer = EnvRenderer(cfg)
print(args.weights)
if args.weights == 'best':
    model_path = experiment['model_best']
else:
    model_path = experiment['model_last']

play_model(model_path, experiment['norm'],
           maker, normalize=True, renderer=renderer)
