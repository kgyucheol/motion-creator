"""Run the task FSM against a running sim stack (python -m ramen_controller.sim.stack ...).

    python scripts/run_sim2sim.py --motions configs/sim_motions.json [--cycles 1] [--gui-port 8765] [--keyboard]

Vision is the MockPoseEstimator fed by the simulator's ground truth, so everything downstream of vision is
exercised before a real estimator is plugged in (see ramen_controller/vision.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ramen_controller.auto_positioner import AutoPositioner
from ramen_controller.contracts import MotionKind
from ramen_controller.debug_gui import DebugServer
from ramen_controller.fsm import TaskFsm, TaskFsmConfig
from ramen_controller.motion_bundle import load_segment, root_pose_at
from ramen_controller.motion_library import FixedGraspPlanner, MotionLibrary, load_entries
from ramen_controller.operator import KeyboardOperator
from ramen_controller.sim.proxy_grasp import ProxyGraspMonitor
from ramen_controller.runner import TaskRunner
from ramen_controller.sim.feed import SimFeed
from ramen_controller.vision import MockPoseEstimator, VisionService
from ramen_controller.adapters.decoupled_wbc import RosWbcLink, bringup_sim

parser = argparse.ArgumentParser()
parser.add_argument("--motions", default="configs/sim_motions.json")
parser.add_argument("--cycles", type=int, default=1)
parser.add_argument("--timeout-s", type=float, default=240.0)
parser.add_argument("--gui-port", type=int, default=0, help="0 disables the debug GUI")
parser.add_argument("--keyboard", action="store_true", help="position the robot by keyboard instead of automatically")
parser.add_argument("--hold-s", type=float, default=0.0, help="keep the debug GUI (and camera) open this long after the run")
parser.add_argument("--no-bringup", action="store_true", help="skip the ] / 9 key sequence (already done)")
parser.add_argument("--grasp-tolerance-m", type=float, default=0.03)
parser.add_argument("--proxy-grasp", action="store_true",
                    help="ASSUME the grasp succeeds (no contact check) to exercise the rest of the cycle")
parser.add_argument("--log", default="logs/events.jsonl")
args = parser.parse_args()

config_path = Path(args.motions).resolve()
settings = json.loads(config_path.read_text(encoding="utf-8"))
entries = load_entries(config_path)

# Where the authored grasp expects the target, in the base frame of the authored root pose.
root_position, root_wxyz = root_pose_at(entries[MotionKind.GRASP].bundle, settings["authored_root_at_s"])
rotation = Rotation.from_quat([root_wxyz[1], root_wxyz[2], root_wxyz[3], root_wxyz[0]])
authored_position = rotation.inv().apply(np.asarray(settings["target_world"]) - root_position)
grasp = entries[MotionKind.GRASP]
planner = FixedGraspPlanner(load_segment(grasp.bundle, MotionKind.GRASP, grasp.start_s, grasp.end_s),
                            authored_position, tolerance_m=args.grasp_tolerance_m)
motion = MotionLibrary(entries, planner)

link = RosWbcLink()
feed = SimFeed()
vision = VisionService(feed, MockPoseEstimator(truth=feed.target_pose, position_sigma_m=0.003, rotation_sigma_deg=1.0,
                                               dropout_probability=0.1, outlier_probability=0.03, seed=0))
operator = KeyboardOperator() if args.keyboard else AutoPositioner(target_xy_yaw=(0.0, 0.0, 0.0))
fsm = TaskFsm(vision, motion, operator, feed, TaskFsmConfig(base_height=0.74))
if args.proxy_grasp:
    fsm.grasp = ProxyGraspMonitor(fsm)
    print("WARNING: --proxy-grasp assumes every grasp succeeds; grasp physics is NOT being validated")
runner = TaskRunner(fsm, link, operator)

Path(args.log).parent.mkdir(parents=True, exist_ok=True)
log = open(args.log, "w", encoding="utf-8")
previous = fsm.on_event
fsm.on_event = lambda e: (log.write(json.dumps({"t": e.stamp, "kind": e.kind.value, "state": e.state, "reason": e.reason,
                                                 "data": e.data}, default=str) + "\n"), log.flush(), previous(e))

print(f"authored grasp expects the target at {np.round(authored_position, 3)} (base frame), tolerance {args.grasp_tolerance_m} m")
print("waiting for robot state and camera feed ...")
deadline = time.monotonic() + 30
while (link.read_state(time.monotonic()) is None or feed.latest() is None) and time.monotonic() < deadline:
    time.sleep(0.1)
if not args.no_bringup:
    print("bring-up: activating the lower-body policy, then releasing the elastic band")
    bringup_sim(link.press_key, time.sleep, settle_s=3.0)
    time.sleep(3.0)
if args.keyboard:
    operator.start()
server = DebugServer(runner, feed, port=args.gui_port).start() if args.gui_port else None
if server:
    print(f"debug GUI: http://127.0.0.1:{server.port}/")

print("running ...")
started = time.monotonic()
try:
    runner.run(duration_s=args.timeout_s, until=lambda: fsm.cycles >= args.cycles or
               (runner.last_state is not None and runner.last_state.fallen))
    if server and args.hold_s > 0:
        print(f"run finished; holding the debug GUI open for {args.hold_s:.0f} s (Ctrl+C to stop)", flush=True)
        time.sleep(args.hold_s)
finally:
    if server:
        server.stop()
    log.close()

transitions = [(round(e.stamp, 1), e.state) for e in fsm.events if e.kind.value == "transition"]
fallen = runner.last_state is not None and runner.last_state.fallen
print(f"\nfinished after {time.monotonic() - started:.0f} s wall, task time {runner.task_time:.1f} s")
print(f"cycles completed: {fsm.cycles} | final state: {fsm.state.value} | fallen: {fallen}")
if fsm.state.value == "RECOVERY":
    print(f"recovery reason: {fsm.recovery_reason}")
print("transitions:", transitions)
link.close()
sys.stdout.flush(); sys.stderr.flush()
# Skip interpreter teardown: rclpy/DDS threads abort() on exit, which would turn a good run into exit code 134.
os._exit(0 if fsm.cycles >= args.cycles and not fallen else 1)
