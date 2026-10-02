"""Adapter between the FSM's contracts and decoupled_wbc's control loop (the same interface sim and real use).

`run_g1_control_loop` subscribes `ControlPolicy/upper_body_pose` (msgpack dict) and publishes
`G1Env/env_state_act`. The conversion logic here is pure so it is testable without ROS; `RosWbcLink`
is the thin transport and imports decoupled_wbc/rclpy lazily.

The upper-body vector order is owned by decoupled_wbc's robot model, so it is mapped by joint NAME,
never by hard-coded position.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import ARMS_SLICE, JOINT_NAMES, WAIST_SLICE, RobotState, WbcCommand

CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
STATE_TOPIC = "G1Env/env_state_act"

_COMMAND_JOINTS = JOINT_NAMES[WAIST_SLICE] + JOINT_NAMES[ARMS_SLICE]


@dataclass(frozen=True)
class UpperBodyLayout:
    """Joint names of `target_upper_body_pose`, in the order the whole-body controller expects."""
    names: tuple[str, ...]

    def __post_init__(self):
        if len(set(self.names)) != len(self.names):
            raise ValueError("upper-body joint names must be unique")
        unknown = [n for n in self.names if n not in _COMMAND_JOINTS and "hand" not in n]
        if unknown:
            raise ValueError(f"upper-body layout has joints this adapter cannot fill: {unknown}")

    @classmethod
    def from_robot_model(cls, robot_model) -> "UpperBodyLayout":
        indices = robot_model.get_joint_group_indices("upper_body")
        return cls(tuple(robot_model.joint_names[i] for i in indices))

    @property
    def has_waist(self) -> bool:
        return any(name in JOINT_NAMES[WAIST_SLICE] for name in self.names)

    def vector(self, command: WbcCommand, hand_positions: Mapping[str, float] | None = None) -> np.ndarray:
        values = dict(zip(JOINT_NAMES[WAIST_SLICE], command.waist_ypr))
        values.update(zip(JOINT_NAMES[ARMS_SLICE], command.arm_targets))
        values.update(hand_positions or {})
        return np.array([values.get(name, 0.0) for name in self.names], dtype=np.float64)


def build_goal(command: WbcCommand, layout: UpperBodyLayout, target_time: float,
               hand_positions: Mapping[str, float] | None = None) -> dict:
    """The dict published on CONTROL_GOAL_TOPIC. `target_time` is on the `time.monotonic()` clock."""
    return {
        "target_upper_body_pose": layout.vector(command, hand_positions),
        "navigate_cmd": np.asarray(command.nav, dtype=np.float64),
        "base_height_command": float(command.base_height),
        "target_time": float(target_time),
    }


def parse_state(message: Mapping, whole_body_names: Sequence[str], stamp: float,
                min_height_m: float = 0.4, max_tilt_deg: float = 60.0) -> RobotState:
    """RobotState from a G1Env/env_state_act message. `whole_body_names` is robot_model.joint_names."""
    q = np.asarray(message["q"], dtype=float)
    if len(q) != len(whole_body_names):
        raise ValueError(f"state has {len(q)} joints but the model has {len(whole_body_names)}")
    index = {name: i for i, name in enumerate(whole_body_names)}
    missing = [name for name in JOINT_NAMES if name not in index]
    if missing:
        raise ValueError(f"model lacks joints {missing}")
    base = np.asarray(message["floating_base_pose"], dtype=float)      # x y z, qw qx qy qz
    quat_wxyz = base[3:7]
    up_z = 1 - 2 * (quat_wxyz[1] ** 2 + quat_wxyz[2] ** 2)               # world z of the pelvis z axis
    tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1)))
    return RobotState(stamp=stamp, joint_positions=np.array([q[index[name]] for name in JOINT_NAMES]),
                      root_position=base[:3], root_quaternion_wxyz=quat_wxyz,
                      fallen=bool(base[2] < min_height_m or tilt > max_tilt_deg))


class RosWbcLink:
    """Publishes goals to / reads state from the running decoupled_wbc control loop over ROS 2."""

    def __init__(self, goal_lead_s: float = 0.05, enable_waist: bool = True, node_name: str = "ramen_controller"):
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig  # noqa: F401  (import check)
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber

        self._ros = ROSManager(node_name=node_name)
        waist = "lower_and_upper_body" if enable_waist else "lower_body"
        model = instantiate_g1_robot_model(waist_location=waist)
        self.layout = UpperBodyLayout.from_robot_model(model)
        self._names = list(model.joint_names)
        self._publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self._subscriber = ROSMsgSubscriber(STATE_TOPIC)
        self.goal_lead_s = goal_lead_s

    def send(self, command: WbcCommand) -> None:
        self._publisher.publish(build_goal(command, self.layout, time.monotonic() + self.goal_lead_s))

    def read_state(self, stamp: float) -> RobotState | None:
        message = self._subscriber.get_msg()
        return None if message is None else parse_state(message, self._names, stamp)

    def close(self) -> None:
        self._ros.shutdown()
