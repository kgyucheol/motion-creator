"""Build the sim2sim scene from a MotionCreator bundle's environment.json.

    python scripts/build_scene.py <bundle-dir> <out-dir> [--target "Mupama_Stack_L03_R04"] [--start -1.0 0 0]
"""
import argparse
from pathlib import Path

from ramen_controller.scene import build_scene

parser = argparse.ArgumentParser()
parser.add_argument("bundle")
parser.add_argument("out")
parser.add_argument("--target", default=None, help="name of the object the vision/grasp stages should pick")
parser.add_argument("--start", nargs=3, type=float, default=[-1.0, 0.0, 0.0], metavar=("X", "Y", "YAW"))
parser.add_argument("--assets", default=str(Path(__file__).resolve().parents[1] / "assets" / "g1_tools"))
args = parser.parse_args()
print(build_scene(args.assets, Path(args.bundle) / "environment.json", args.out, target_name=args.target,
                  robot_start_xy_yaw=tuple(args.start)))
