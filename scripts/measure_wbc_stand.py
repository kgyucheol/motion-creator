#!/usr/bin/env python
"""Measure the pose the decoupled-WBC holds when it simply stands, and write configs/wbc_stand.json.

The editor's default stand has the legs nearly straight (knee 0.24 rad, pelvis 0.79 m). The whole-body controller
stands with the knees bent (~0.74 rad) and the pelvis ~4 cm lower, so upper-body motion authored on the default stand
lands lower and further from the target at run time. Re-run this after changing the WBC policy or the base height.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motioncreator.decoupled_wbc import DecoupledSimulation  # noqa: E402
from motioncreator.motion import new_project  # noqa: E402
from motioncreator.robot import Robot  # noqa: E402

SETTLE_STEPS, WINDOW_STEPS = 1800, 300
LOWER_BODY = ("hip", "knee", "ankle", "waist")


def main() -> None:
    robot = Robot("g1-tools", ("scoop", "scoop"))
    sim = DecoupledSimulation(robot, new_project(robot), autostart=False)
    sim.set_control_mode("external")
    sim.play()
    samples = []
    for step in range(SETTLE_STEPS):
        sim.step()
        if step >= SETTLE_STEPS - WINDOW_STEPS:
            with sim.lock:
                samples.append(sim.data.qpos[:36].copy())
    samples = np.stack(samples)
    q = samples.mean(axis=0)
    joints = {name: round(float(value), 5) for name, value in zip(robot.names, q[7:])
              if any(part in name for part in LOWER_BODY)}
    out = {
        "format": "motioncreator.wbc_stand.v1",
        "source": "decoupled_wbc standing with the neutral upper-body command, mean over the last "
                  f"{WINDOW_STEPS} of {SETTLE_STEPS} sim steps",
        "pelvis_height_m": round(float(q[2]), 5),
        "pelvis_height_spread_m": round(float(np.ptp(samples[:, 2])), 6),
        "joints": joints,
    }
    path = ROOT / "configs/wbc_stand.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}: pelvis {out['pelvis_height_m']} m, knees "
          f"{joints['left_knee_joint']:.3f}/{joints['right_knee_joint']:.3f} rad")


if __name__ == "__main__":
    main()
