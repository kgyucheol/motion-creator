"""OperatorPort that walks the robot to a target pose from odometry (sim2sim stand-in for keyboard positioning).

Same interface as KeyboardOperator, plus `observe(state)`, which the runner calls every tick. This is the
simulation-only precursor of the planned locomotion-based positioning: it uses the simulator's world pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import RobotState


@dataclass
class AutoPositioner:
    target_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)     # world frame
    position_tolerance_m: float = 0.03
    yaw_tolerance_rad: float = 0.05
    hold_s: float = 1.0                       # must stay inside the tolerance this long to confirm
    kp_linear: float = 0.8
    kp_yaw: float = 1.0
    max_linear: float = 0.3
    max_yaw: float = 0.5
    min_linear: float = 0.12                  # the locomotion policy ignores commands below ~0.05
    min_yaw: float = 0.15
    _nav: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _inside_since: float | None = field(default=None, init=False)
    _confirmed: bool = field(default=False, init=False)
    _reset: bool = field(default=False, init=False)

    def observe(self, state: RobotState) -> None:
        w, x, y, z = state.root_quaternion_wxyz
        yaw = Rotation.from_quat([x, y, z, w]).as_euler("xyz")[2]
        error_world = np.array(self.target_xy_yaw[:2]) - state.root_position[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        error_body = np.array([c * error_world[0] + s * error_world[1], -s * error_world[0] + c * error_world[1]])
        yaw_error = (self.target_xy_yaw[2] - yaw + np.pi) % (2 * np.pi) - np.pi
        distance = float(np.linalg.norm(error_world))

        inside = distance <= self.position_tolerance_m and abs(yaw_error) <= self.yaw_tolerance_rad
        if inside:
            self._inside_since = state.stamp if self._inside_since is None else self._inside_since
            self._confirmed = self._confirmed or state.stamp - self._inside_since >= self.hold_s
            self._nav = np.zeros(3)
            return
        self._inside_since = None
        linear = self.kp_linear * error_body if distance > self.position_tolerance_m else np.zeros(2)
        speed = float(np.linalg.norm(linear))
        if speed > self.max_linear:
            linear *= self.max_linear / speed
        elif 0 < speed < self.min_linear:
            linear *= self.min_linear / speed
        yaw_rate = self.kp_yaw * yaw_error if abs(yaw_error) > self.yaw_tolerance_rad else 0.0
        if abs(yaw_rate) > self.max_yaw:
            yaw_rate = np.sign(yaw_rate) * self.max_yaw
        elif 0 < abs(yaw_rate) < self.min_yaw:
            yaw_rate = np.sign(yaw_rate) * self.min_yaw
        self._nav = np.array([linear[0], linear[1], yaw_rate])

    def nav(self) -> np.ndarray:
        return self._nav

    def position_confirmed(self) -> bool:
        return self._confirmed

    def reset_requested(self) -> bool:
        flag, self._reset = self._reset, False
        return flag

    def request_reset(self) -> None:
        self._reset = True
        self._confirmed = False
        self._inside_since = None
