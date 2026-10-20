"""Joint layout and safety limits of the arm_sdk replay (values follow g1-live-studio's config where they exist there)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CYCLONEDDS_HOME = Path(os.environ.get('CYCLONEDDS_HOME', Path.home() / 'cyclonedds' / 'install'))   # the build the Unitree SDK's binding matches
ROBOT_PYTHON = os.environ.get('MOTIONCREATOR_ROBOT_PYTHON')        # interpreter that has unitree_sdk2py (default: this one)
ROBOT_NET = '192.168.123.'

N_JOINTS = 29
ARM_SLICE = slice(15, 29)                      # left 7 + right 7, Unitree order
WAIST_SLICE = slice(12, 15)                    # waist yaw, roll, pitch
UPPER_SLICE = slice(12, 29)                    # waist then arms: what a replay with the waist commands
ARM14 = {'left': slice(0, 7), 'right': slice(7, 14)}

# ---- safety (enforced on the server whatever the browser sends) ------------------------------------------------------------
COMMAND_HZ = 50
STATE_STALE_S = 0.5           # refuse / release if the robot state is older than this
TILT_FAULT_DEG = 35.0         # release if the pelvis tilts more than this
SETTLED_SPEED = 0.15          # rad/s: the arms must be slower than this to start
TRACKING_FAULT_RAD = 0.35     # stop if a joint lags its command by more than this ...
TRACKING_FAULT_S = 0.4        # ... for this long
MIN_MOVE_S = 0.8              # shortest planned approach / return
RETURN_VMAX = 0.5             # rad/s on the way back to the starting pose
REPLAY_VMAX = 1.5             # rad/s: recommended ceiling; above it a replay needs an explicit confirmation
REPLAY_VMAX_HARD = 3.0        # rad/s: nothing above this is allowed
REPLAY_APPROACH_VMAX = 0.6    # rad/s: the move to the first frame, gentle whatever the recording's speed
REPLAY_SPEED_MIN = 0.1
REPLAY_FREQ_MAX = 250
APPROACH_OK_RAD = 0.35        # the first frame this close to the arms needs no warning
JOINT_MARGIN = 0.02           # rad kept from the model's joint limits

# ---- replayability: what the arm_sdk path cannot play -----------------------------------------------------------------------
LOWER_BODY_TOLERANCE_RAD = 0.02    # legs and waist roll/pitch may differ this much from the first frame
ROOT_HEIGHT_TOLERANCE_M = 0.01
ROOT_TILT_TOLERANCE_RAD = 0.02
WAIST_YAW_TOLERANCE_RAD = 0.02     # waist yaw, when the waist is not sent

# ---- arm_sdk writer ---------------------------------------------------------------------------------------------------------
FF_DEFAULT = 0.5              # share of the computed gravity torque sent as feed-forward (untested on the robot: start low)
ARM_SDK_DEFAULTS = {'ff': FF_DEFAULT, 'hz': 250, 'kp_arm': 80.0, 'kd_arm': 3.0, 'kp_wrist': 40.0, 'kd_wrist': 1.5}
ARM_SDK_LIMITS = {'ff': (0.0, 1.0), 'kp_arm': (10.0, 200.0), 'kd_arm': (0.5, 8.0), 'kp_wrist': (5.0, 80.0), 'kd_wrist': (0.2, 4.0)}
ARM_SDK_HZ_CHOICES = (50, 100, 125, 250)
ARM_SDK_ENGAGE_S = 1.5        # arm_sdk weight 0 -> 1 while the arms hold the pose they are in
ARM_SDK_RELEASE_S = 2.0       # weight 1 -> 0 (the robot's own controller takes the arms back)
ARM_SDK_WATCHDOG_S = 0.5      # no target for this long -> the writer releases by itself
ARM_SDK_VLIMIT = REPLAY_VMAX_HARD
ARM_SDK_MAX_LEAD_RAD = 0.25   # the command never gets further than this ahead of the measured joint
