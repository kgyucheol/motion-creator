"""Data contracts shared by the FSM, the vision/motion modules and the whole-body controller.

Conventions: right-handed, +X forward, +Y left, +Z up; metres, radians, seconds.
Joint order is the 29-DoF G1 order used by MotionCreator bundles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
WAIST_SLICE = slice(12, 15)   # yaw, roll, pitch as authored in MotionCreator
ARMS_SLICE = slice(15, 29)    # left 7 + right 7
ARM_DOF = 14


class MotionKind(str, Enum):
    READY = "ready"
    GRASP = "grasp"
    TRANSFER = "transfer"
    PLACE = "place"
    RETURN = "return"


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class RobotState:
    """Measured robot state, as supplied by the sim or the real-robot adapter."""
    stamp: float
    joint_positions: np.ndarray            # (29,) in JOINT_NAMES order
    root_position: np.ndarray              # (3,) world
    root_quaternion_wxyz: np.ndarray       # (4,) world
    fallen: bool = False

    def __post_init__(self):
        object.__setattr__(self, "joint_positions", _array(self.joint_positions, (29,), "joint_positions"))
        object.__setattr__(self, "root_position", _array(self.root_position, (3,), "root_position"))
        object.__setattr__(self, "root_quaternion_wxyz",
                           _array(self.root_quaternion_wxyz, (4,), "root_quaternion_wxyz"))


@dataclass(frozen=True)
class ObjectPose:
    """6D pose of a detected object. `frame_id` names the frame the pose is expressed in."""
    stamp: float
    frame_id: str
    position: np.ndarray                   # (3,)
    quaternion_xyzw: np.ndarray            # (4,)
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self):
        object.__setattr__(self, "position", _array(self.position, (3,), "position"))
        quaternion = _array(self.quaternion_xyzw, (4,), "quaternion_xyzw")
        norm = np.linalg.norm(quaternion)
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"quaternion_xyzw must be normalised, norm={norm:.4f}")
        object.__setattr__(self, "quaternion_xyzw", quaternion / norm)


@dataclass(frozen=True)
class WbcCommand:
    """What the FSM asks of the whole-body controller. The controller owns the final joint commands."""
    stamp: float
    arm_targets: np.ndarray                # (14,) left 7 + right 7
    waist_ypr: np.ndarray                  # (3,) authored waist yaw, roll, pitch; the WBC derives the torso pose
    base_height: float
    nav: np.ndarray = field(default_factory=lambda: np.zeros(3))   # vx, vy, yaw_rate

    def __post_init__(self):
        object.__setattr__(self, "arm_targets", _array(self.arm_targets, (ARM_DOF,), "arm_targets"))
        object.__setattr__(self, "waist_ypr", _array(self.waist_ypr, (3,), "waist_ypr"))
        object.__setattr__(self, "nav", _array(self.nav, (3,), "nav"))

    def torso_rpy(self, limits) -> np.ndarray:
        """Policy-style XYZ torso RPY for backends that take a torso command instead of waist joints."""
        return waist_to_torso_rpy(self.waist_ypr, limits)


def waist_to_torso_rpy(waist_yaw_roll_pitch, limits) -> np.ndarray:
    """Authored waist joints (yaw, roll, pitch) to the policy's XYZ torso RPY, clipped to `limits`.

    Same conversion as MotionCreator's `authored_waist_to_torso_rpy`.
    """
    yaw, roll, pitch = np.asarray(waist_yaw_roll_pitch, dtype=float)
    rpy = Rotation.from_euler("zxy", [yaw, roll, pitch]).as_euler("xyz")
    limits = np.asarray(limits, dtype=float)
    return np.clip(rpy, -limits, limits)


@dataclass(frozen=True)
class MotionSegment:
    """Upper-body reference for one task stage, sampled at `fps`."""
    kind: MotionKind
    fps: float
    arm_targets: np.ndarray                # (T, 14)
    waist_ypr: np.ndarray                  # (T, 3) authored waist yaw, roll, pitch
    name: str = ""
    # Optional closed-loop correction applied while the segment plays: (t, arms, waist, measured state) -> (arms, waist).
    refine: Callable[[float, np.ndarray, np.ndarray, "RobotState"], tuple[np.ndarray, np.ndarray]] | None = \
        field(default=None, compare=False, repr=False)

    def __post_init__(self):
        arms = np.asarray(self.arm_targets, dtype=float)
        waist = np.asarray(self.waist_ypr, dtype=float)
        if arms.ndim != 2 or arms.shape[1] != ARM_DOF or len(arms) < 1:
            raise ValueError(f"arm_targets must be (T>=1, {ARM_DOF}), got {arms.shape}")
        if waist.shape != (len(arms), 3):
            raise ValueError(f"waist_ypr must be ({len(arms)}, 3), got {waist.shape}")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        object.__setattr__(self, "arm_targets", arms)
        object.__setattr__(self, "waist_ypr", waist)

    @property
    def duration(self) -> float:
        return (len(self.arm_targets) - 1) / self.fps

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolated (arm_targets, waist_ypr) at time `t`, clamped to the segment."""
        position = float(np.clip(t * self.fps, 0.0, len(self.arm_targets) - 1))
        low = int(np.floor(position))
        high = min(low + 1, len(self.arm_targets) - 1)
        weight = position - low
        return (self.arm_targets[low] * (1 - weight) + self.arm_targets[high] * weight,
                self.waist_ypr[low] * (1 - weight) + self.waist_ypr[high] * weight)


class EventKind(str, Enum):
    TRANSITION = "transition"
    CYCLE_COMPLETE = "cycle_complete"
    WARNING = "warning"


@dataclass(frozen=True)
class TaskEvent:
    """One logged FSM event. Every state change carries the reason that justified it."""
    stamp: float
    kind: EventKind
    state: str
    reason: str
    data: dict[str, Any] = field(default_factory=dict)
