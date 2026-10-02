"""Build the sim2sim scene from a MotionCreator bundle.

    python scripts/build_sim_scene.py <bundle-dir> <out-dir> [--target NAME] [--start X Y YAW] [--tools LEFT:RIGHT]

The robot's tools default to the pair recorded in the bundle's metadata.json (`mounted_tools`).
"""
import argparse
from pathlib import Path

from motioncreator.sim_scene import build_scene, bundle_tools
from motioncreator.tool_model import parse_tools

parser = argparse.ArgumentParser()
parser.add_argument("bundle")
parser.add_argument("out")
parser.add_argument("--target", default=None, help="name of the object the vision/grasp stages should pick")
parser.add_argument("--start", nargs=3, type=float, default=[-1.0, 0.0, 0.0], metavar=("X", "Y", "YAW"))
parser.add_argument("--tools", default=None, help="override the bundle's tools, e.g. scoop:end_support")
args = parser.parse_args()

tools = parse_tools(args.tools) if args.tools else bundle_tools(args.bundle)
scene = build_scene(Path(args.bundle) / "environment.json", args.out, tools=tools, target_name=args.target,
                    robot_start_xy_yaw=tuple(args.start))
print(f"tools {tools[0]}:{tools[1]}")
print(scene)
