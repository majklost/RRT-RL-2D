from argparse import ArgumentParser

from rrt_rl_2D.importable.saved_path_replayer import SavePathReplayer
from rrt_rl_2D.envs.rect import RectEnvI
from rrt_rl_2D.RL.training_utils import standard_wrap

from rrt_rl_2D.analytics.standard_analytics import StandardAnalytics

parser = ArgumentParser(prog="play_experiment",
                        description="Play a model on the environment.")
my_fpath = "./data/Test.rpath"
parser.add_argument("experiment_path", type=str, default=my_fpath)
my_args = parser.parse_args()

replayer = SavePathReplayer(my_args.experiment_path)
replayer.replay()
