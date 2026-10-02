"""Launch decoupled_wbc's control loop with its simulator running this project's scene.

    python -m ramen_controller.sim_stack --scene-dir build/scene [control-loop options...]

With `--interface sim` the control loop creates the simulator in-process (a thread), so one process holds the
MuJoCo sim, the DDS bridge, the whole-body policy and the ROS 2 node; the FSM runs as a separate process and
talks to it over ROS 2 (goals/state) and ZMQ (camera + sim truth), mirroring the real-robot split.
The patches below touch only in-memory objects of the pinned upstream revision.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONTROL_ARGS = ["--interface", "sim", "--enable-waist", "--no-with-hands", "--no-enable-onscreen",
                        "--keyboard-dispatcher-type", "ros", "--no-verbose"]


def patch_config(scene_xml: Path, image_dt: float) -> None:
    from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
    original = ControlLoopConfig.load_wbc_yaml

    def load_wbc_yaml(self):
        config = original(self)
        config["ROBOT_SCENE"] = str(scene_xml)          # absolute path: the sim joins it onto the package root
        config["NUM_HAND_JOINTS"] = 0                   # the tool-equipped model has no hand actuators
        config["IMAGE_DT"] = image_dt
        return config

    ControlLoopConfig.load_wbc_yaml = load_wbc_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", required=True, help="output of scripts/build_scene.py")
    parser.add_argument("--image-dt", type=float, default=0.1, help="camera/truth feed period in seconds")
    args, control_args = parser.parse_known_args(argv)
    scene_xml = Path(args.scene_dir).resolve() / "scene.xml"
    if not scene_xml.is_file():
        sys.exit(f"{scene_xml} not found; run scripts/build_scene.py first")

    from . import sim_hook
    import tyro
    from decoupled_wbc.control.main.teleop import run_g1_control_loop
    from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

    patch_config(scene_xml, args.image_dt)
    sim_hook.install()
    run_g1_control_loop.main(tyro.cli(ControlLoopConfig, args=control_args or DEFAULT_CONTROL_ARGS))


if __name__ == "__main__":
    main()
