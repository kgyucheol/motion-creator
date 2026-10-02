"""Check a running sim stack: camera feed, ground truth and robot state.

    python scripts/probe_stack.py [--save-frame out.png] [--seconds 5]
"""
import argparse
import os
import sys
import time

import numpy as np

from ramen_controller.sim.feed import SimFeed
from ramen_controller.adapters.decoupled_wbc import RosWbcLink

parser = argparse.ArgumentParser()
parser.add_argument("--save-frame")
parser.add_argument("--seconds", type=float, default=5.0)
args = parser.parse_args()

feed = SimFeed(require_depth=True)
link = RosWbcLink()
deadline, frames, state, stamps = time.monotonic() + args.seconds, 0, None, []
last = None
while time.monotonic() < deadline:
    frame = feed.latest()
    if frame is not None and frame.stamp != getattr(last, "stamp", None):
        frames, last = frames + 1, frame
        stamps.append(frame.stamp)
    state = link.read_state(time.monotonic()) or state
    time.sleep(0.01)

print(f"camera frames in {args.seconds:.0f}s: {frames}" + (f" (~{(len(stamps) - 1) / (stamps[-1] - stamps[0]):.1f} Hz)" if frames > 1 else ""))
if last is not None:
    d = last.depth
    valid = d[d > 0]
    print(f"rgb {last.rgb.shape} mean {last.rgb.mean():.0f} | depth {d.shape} valid {valid.size / d.size:.0%} range {valid.min():.2f}-{valid.max():.2f} m")
    print("K fx %.1f cx %.1f | camera in base: pos %s" % (last.K[0, 0], last.K[0, 2], np.round(last.T_base_cam[:3, 3], 3)))
    print("target pose (base frame):", feed.target_pose())
    print("held:", feed.held())
    if args.save_frame:
        import cv2
        cv2.imwrite(args.save_frame, last.rgb[..., ::-1])
        depth_vis = np.clip(d / 3.0, 0, 1)
        cv2.imwrite(args.save_frame.replace(".png", "_depth.png"), (255 * (1 - depth_vis)).astype(np.uint8))
else:
    print("NO camera frames received")
if state is None:
    print("NO robot state received")
else:
    print(f"robot: root z {state.root_position[2]:.3f} m, xy {np.round(state.root_position[:2], 3)}, fallen={state.fallen}")
link.close()
sys.stdout.flush()
os._exit(0)   # skip interpreter teardown (rclpy/DDS threads abort on exit)
