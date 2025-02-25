from rrt_rl_2D.importable.saved_path_replayer import SavePathReplayer
from rrt_rl_2D.envs.rect import RectEnvI
from rrt_rl_2D.RL.training_utils import standard_wrap


my_fpath = "test.rpath"

replayer = SavePathReplayer(my_fpath)
replayer.replay()
