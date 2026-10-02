"""Task FSM: one executor for sim2sim and the real robot.

The FSM never touches hardware. Each `step(now, state)` returns the WbcCommand to send; vision, motion,
operator input and grasp sensing are injected through the ports below, so the same FSM runs against a
simulator or the robot by swapping adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np

from .contracts import (ARMS_SLICE, WAIST_SLICE, EventKind, MotionKind, MotionSegment, ObjectPose,
                        RobotState, TaskEvent, WbcCommand)
from .pose_average import PoseAveragingConfig, PoseAveragingError, average_poses


class TaskState(str, Enum):
    INITIALIZE = "INITIALIZE"
    MANUAL_POSITION = "MANUAL_POSITION"
    OBSERVE = "OBSERVE"
    PLAN_GRASP = "PLAN_GRASP"
    EXECUTE_GRASP = "EXECUTE_GRASP"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RETURN = "RETURN"
    RECOVERY = "RECOVERY"


class GraspPlanError(RuntimeError):
    """Raised by MotionPort.plan_grasp when no feasible grasp exists for the observed pose."""


class VisionPort(Protocol):
    def detections(self, since: float) -> Sequence[ObjectPose]:
        """Detections with stamp > `since`, oldest first."""


class MotionPort(Protocol):
    def segment(self, kind: MotionKind) -> MotionSegment: ...
    def plan_grasp(self, pose: ObjectPose) -> MotionSegment: ...


class OperatorPort(Protocol):
    def nav(self) -> np.ndarray:
        """(vx, vy, yaw_rate) requested by the operator."""
    def position_confirmed(self) -> bool: ...
    def reset_requested(self) -> bool:
        """True once per operator reset request."""


class GraspMonitor(Protocol):
    def held(self) -> bool:
        """True while the object is judged to be held by both hands."""


@dataclass(frozen=True)
class TaskFsmConfig:
    base_height: float = 0.74
    blend_s: float = 1.0                    # smooth transition into each segment's first frame
    skip_manual_position: bool = False
    nav_zero_tolerance: float = 0.05
    observe_duration_s: float = 1.0         # detections averaged over this window
    observe_timeout_s: float = 5.0
    grasp_confirm_s: float = 0.5            # held() must stay true this long
    grasp_confirm_timeout_s: float = 2.0
    release_timeout_s: float = 2.0
    recovery_hold_s: float = 1.0
    max_auto_retries: int = 3
    averaging: PoseAveragingConfig = field(default_factory=PoseAveragingConfig)


@dataclass
class _Playback:
    segment: MotionSegment
    started: float
    blend_from_arms: np.ndarray
    blend_from_waist: np.ndarray
    blend_s: float

    def sample(self, now: float) -> tuple[np.ndarray, np.ndarray, bool]:
        elapsed = now - self.started
        first_arms, first_waist = self.segment.sample(0.0)
        if elapsed < self.blend_s:
            x = elapsed / self.blend_s
            u = x * x * (3 - 2 * x)
            return (self.blend_from_arms * (1 - u) + first_arms * u,
                    self.blend_from_waist * (1 - u) + first_waist * u, False)
        t = elapsed - self.blend_s
        arms, waist = self.segment.sample(t)
        return arms, waist, t >= self.segment.duration


class TaskFsm:
    def __init__(self, vision: VisionPort, motion: MotionPort, operator: OperatorPort, grasp: GraspMonitor,
                 config: TaskFsmConfig = TaskFsmConfig(),
                 on_event: Callable[[TaskEvent], None] | None = None):
        self.vision, self.motion, self.operator, self.grasp, self.config = vision, motion, operator, grasp, config
        self.on_event = on_event
        self.events: list[TaskEvent] = []
        self.state = TaskState.INITIALIZE
        self.cycles = 0
        self.retries = 0
        self.target_pose: ObjectPose | None = None
        self.recovery_reason = ""
        self._arms: np.ndarray | None = None
        self._waist: np.ndarray | None = None
        self._playback: _Playback | None = None
        self._entered = 0.0
        self._samples: list[ObjectPose] = []
        self._last_detection = float("-inf")
        self._recoverable = False
        self._confirm_since: float | None = None
        self._started = False

    # -- public -----------------------------------------------------------------------------------

    def step(self, now: float, robot: RobotState) -> WbcCommand:
        if self._arms is None:
            self._arms = robot.joint_positions[ARMS_SLICE].copy()
            self._waist = robot.joint_positions[WAIST_SLICE].copy()
        if not self._started:
            self._started = True
            self._enter(self.state, now, "start")
        if robot.fallen and self.state is not TaskState.RECOVERY:
            self._recoverable = False
            self._transition(TaskState.RECOVERY, now, "robot fell")
        nav = getattr(self, f"_tick_{self.state.value.lower()}")(now, robot)
        return self._command(now, np.zeros(3) if nav is None else nav)

    # -- plumbing ---------------------------------------------------------------------------------

    def _emit(self, now: float, kind: EventKind, reason: str, **data) -> None:
        event = TaskEvent(stamp=now, kind=kind, state=self.state.value, reason=reason, data=data)
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def _transition(self, new: TaskState, now: float, reason: str, **data) -> None:
        old = self.state
        self.state = new
        self._emit(now, EventKind.TRANSITION, reason, previous=old.value, **data)
        self._enter(new, now, reason)

    def _enter(self, state: TaskState, now: float, reason: str) -> None:
        self._entered = now
        self._confirm_since = None
        self._playback = None
        if state is TaskState.OBSERVE:
            self._samples, self._last_detection = [], now
        elif state is TaskState.INITIALIZE:
            self._play(self.motion.segment(MotionKind.READY), now)
        elif state is TaskState.TRANSFER:
            self._play(self.motion.segment(MotionKind.TRANSFER), now)
        elif state is TaskState.PLACE:
            self._play(self.motion.segment(MotionKind.PLACE), now)
        elif state is TaskState.RETURN:
            self._play(self.motion.segment(MotionKind.RETURN), now)
        elif state is TaskState.RECOVERY:
            self.recovery_reason = reason

    def _play(self, segment: MotionSegment, now: float) -> None:
        self._playback = _Playback(segment, now, self._arms.copy(), self._waist.copy(), self.config.blend_s)

    def _advance_playback(self, now: float) -> bool:
        arms, waist, finished = self._playback.sample(now)
        self._arms, self._waist = arms, waist
        return finished

    def _command(self, now: float, nav: np.ndarray) -> WbcCommand:
        return WbcCommand(stamp=now, arm_targets=self._arms, waist_ypr=self._waist,
                          base_height=self.config.base_height, nav=nav)

    def _held_for(self, now: float, seconds: float) -> bool:
        """True once held() has been continuously true for `seconds`."""
        if not self.grasp.held():
            self._confirm_since = None
            return False
        if self._confirm_since is None:
            self._confirm_since = now
        return now - self._confirm_since >= seconds

    # -- state ticks (return the nav command, or None for zero) --------------------------------------

    def _tick_initialize(self, now: float, robot: RobotState):
        if self._advance_playback(now):
            nxt = TaskState.OBSERVE if self.config.skip_manual_position else TaskState.MANUAL_POSITION
            self._transition(nxt, now, "ready pose reached")

    def _tick_manual_position(self, now: float, robot: RobotState):
        nav = np.asarray(self.operator.nav(), dtype=float)
        if self.operator.position_confirmed() and np.linalg.norm(nav) <= self.config.nav_zero_tolerance:
            self._transition(TaskState.OBSERVE, now, "operator confirmed position")
            return None
        return nav

    def _tick_observe(self, now: float, robot: RobotState):
        for pose in self.vision.detections(self._last_detection):
            self._samples.append(pose)
            self._last_detection = max(self._last_detection, pose.stamp)
        window_start = now - self.config.observe_duration_s
        self._samples = [pose for pose in self._samples if pose.stamp >= window_start]
        elapsed = now - self._entered
        if elapsed < self.config.observe_duration_s:
            return
        try:
            self.target_pose = average_poses(self._samples, self.config.averaging)
        except PoseAveragingError as error:
            if elapsed >= self.config.observe_timeout_s:
                self._recoverable = True
                self._transition(TaskState.RECOVERY, now, f"observation failed: {error}")
            return
        self._transition(TaskState.PLAN_GRASP, now, "observation stable", pose_source=self.target_pose.source)

    def _tick_plan_grasp(self, now: float, robot: RobotState):
        try:
            segment = self.motion.plan_grasp(self.target_pose)
        except GraspPlanError as error:
            self._recoverable = True
            self._transition(TaskState.RECOVERY, now, f"grasp planning failed: {error}")
            return
        self._transition(TaskState.EXECUTE_GRASP, now, "grasp planned")
        self._play(segment, now)

    def _tick_execute_grasp(self, now: float, robot: RobotState):
        if not self._advance_playback(now):
            return
        if self._held_for(now, self.config.grasp_confirm_s):
            self._transition(TaskState.TRANSFER, now, "grasp confirmed")
        elif now - self._entered > self._playback.segment.duration + self._playback.blend_s \
                + self.config.grasp_confirm_timeout_s:
            self._recoverable = False
            self._transition(TaskState.RECOVERY, now, "grasp not confirmed")

    def _tick_transfer(self, now: float, robot: RobotState):
        if not self.grasp.held():
            self._recoverable = False
            self._transition(TaskState.RECOVERY, now, "object lost during transfer")
        elif self._advance_playback(now):
            self._transition(TaskState.PLACE, now, "transfer pose reached")

    def _tick_place(self, now: float, robot: RobotState):
        if not self._advance_playback(now):
            return
        if not self.grasp.held():
            self._transition(TaskState.RETURN, now, "object released")
        elif now - self._entered > self._playback.segment.duration + self._playback.blend_s \
                + self.config.release_timeout_s:
            self._recoverable = False
            self._transition(TaskState.RECOVERY, now, "object not released")

    def _tick_return(self, now: float, robot: RobotState):
        if not self._advance_playback(now):
            return
        self.cycles += 1
        self.retries = 0
        self._emit(now, EventKind.CYCLE_COMPLETE, "cycle complete", cycles=self.cycles)
        self._transition(TaskState.OBSERVE, now, "returned to ready pose")

    def _tick_recovery(self, now: float, robot: RobotState):
        if self.operator.reset_requested() and not robot.fallen:
            self.retries = 0
            self._transition(TaskState.INITIALIZE, now, "operator reset")
        elif (self._recoverable and not robot.fallen and self.retries < self.config.max_auto_retries
              and now - self._entered >= self.config.recovery_hold_s):
            self.retries += 1
            self._transition(TaskState.OBSERVE, now, f"automatic retry {self.retries}")
