"""TaskRunner: drives the FSM from a WbcLink at a fixed rate and exposes a snapshot for debugging.

The FSM runs on a *task clock* that only advances while fresh robot state arrives. If the state stream drops
out (control loop died, DDS hiccup) the task pauses instead of jumping ahead on resume, and nothing is sent, so
the whole-body controller's own goal timeout takes over.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Protocol

import numpy as np

from .contracts import EventKind, RobotState, TaskEvent, WbcCommand
from .fsm import TaskFsm


class WbcLink(Protocol):
    def read_state(self, stamp: float) -> RobotState | None:
        """The newest robot state, or None if nothing new has arrived since the last call."""
    def send(self, command: WbcCommand) -> None: ...


class TaskRunner:
    def __init__(self, fsm: TaskFsm, link: WbcLink, operator=None, clock: Callable[[], float] = time.monotonic,
                 rate_hz: float = 50.0, max_state_gap_s: float = 0.5, event_history: int = 50):
        self.fsm, self.link, self.operator, self.clock = fsm, link, operator, clock
        self.period = 1.0 / rate_hz
        self.max_state_gap_s = max_state_gap_s
        self.task_time = 0.0
        self.ticks = 0
        self.stalled = False
        self.last_state: RobotState | None = None
        self.last_command: WbcCommand | None = None
        self.recent_events: deque[TaskEvent] = deque(maxlen=event_history)
        self._last_wall: float | None = None
        self._last_state_wall: float | None = None
        previous = fsm.on_event
        fsm.on_event = lambda event: (self.recent_events.append(event), previous and previous(event))

    def tick(self) -> WbcCommand | None:
        wall = self.clock()
        state = self.link.read_state(wall)
        if state is not None:
            if self.stalled:
                self.stalled = False
                self._note(wall, "state stream resumed")
            self._last_state_wall = wall
            self.last_state = state
        elif self._last_state_wall is not None and wall - self._last_state_wall > self.max_state_gap_s \
                and not self.stalled:
            self.stalled = True
            self._note(wall, f"no robot state for {wall - self._last_state_wall:.2f} s; task paused")
        step = 0.0 if self._last_wall is None else min(wall - self._last_wall, 2 * self.period)
        self._last_wall = wall
        if state is None:                                   # also covers a stalled stream
            return None
        self.task_time += step
        if hasattr(self.operator, "observe"):
            self.operator.observe(state)
        command = self.fsm.step(self.task_time, state)
        self.link.send(command)
        self.last_command = command
        self.ticks += 1
        return command

    def run(self, duration_s: float | None = None, until: Callable[[], bool] | None = None,
            sleep: Callable[[float], None] = time.sleep) -> None:
        end = None if duration_s is None else self.clock() + duration_s
        while (end is None or self.clock() < end) and not (until and until()):
            started = self.clock()
            self.tick()
            sleep(max(0.0, self.period - (self.clock() - started)))

    def _note(self, wall: float, reason: str) -> None:
        event = TaskEvent(stamp=self.task_time, kind=EventKind.WARNING, state=self.fsm.state.value, reason=reason)
        self.fsm.events.append(event)
        self.recent_events.append(event)

    def snapshot(self) -> dict:
        """JSON-friendly view for the debug GUI."""
        state, command, fsm = self.last_state, self.last_command, self.fsm
        tilt = None
        if state is not None:
            w, x, y, z = state.root_quaternion_wxyz
            tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        return {
            "state": fsm.state.value, "cycles": fsm.cycles, "retries": fsm.retries,
            "recovery_reason": fsm.recovery_reason if fsm.state.value == "RECOVERY" else "",
            "task_time": round(self.task_time, 2), "ticks": self.ticks, "stalled": self.stalled,
            "robot": None if state is None else {"root_xyz": np.round(state.root_position, 3).tolist(),
                                                "tilt_deg": round(tilt, 1), "fallen": state.fallen},
            "command": None if command is None else {"nav": np.round(command.nav, 3).tolist(),
                                                     "waist_ypr": np.round(command.waist_ypr, 3).tolist(),
                                                     "arms_max_abs": round(float(np.max(np.abs(command.arm_targets))), 3)},
            "target": None if fsm.target_pose is None else {"position": np.round(fsm.target_pose.position, 3).tolist(),
                                                           "source": fsm.target_pose.source},
            "events": [{"t": round(e.stamp, 2), "kind": e.kind.value, "state": e.state, "reason": e.reason}
                       for e in list(self.recent_events)[-20:]],
        }
