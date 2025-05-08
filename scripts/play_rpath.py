"""Play *.rpath file"""
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

from rrt_rl_2D.importable.saved_path_replayer import SavePathReplayer
from rrt_rl_2D.rendering import PictureRendererDebug

parser = ArgumentParser(prog="play_experiment",
                        description="Play a rpath file on the environment.")
parser.add_argument("experiment_path", type=str)
parser.add_argument('--video_path', type=str, default=None)
args = parser.parse_args()

if args.video_path is not None:
    print(f"Creating video at {args.video_path}")
    args.video_path = Path(args.video_path)
    tmp_dir = Path("/tmp/rpath/")
    if tmp_dir.exists():
        rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    renderer = PictureRendererDebug(
        tmp_dir, video_path=args.video_path, max_frames=10000)

else:
    renderer = None


replayer = SavePathReplayer(args.experiment_path, renderer=renderer)
replayer.replay()
if renderer is not None:
    renderer.make_video(args.video_path)
    renderer._end()
