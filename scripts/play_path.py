from argparse import ArgumentParser

from rrt_rl_2D.importable.saved_path_replayer import SavePathReplayer

parser = ArgumentParser(prog="play_experiment",
                        description="Play a model on the environment.")
parser.add_argument("experiment_path", type=str)
args = parser.parse_args()

replayer = SavePathReplayer(args.experiment_path)
replayer.replay()
