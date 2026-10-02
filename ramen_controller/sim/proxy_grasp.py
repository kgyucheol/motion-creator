"""Grasp monitor that ASSUMES the grasp worked. Simulation only, for exercising the FSM without grasp physics.

It reports "held" from the end of the grasp motion until the end of the place motion, which is what a working
grip sensor would report if every grasp succeeded. It says nothing about whether a real grasp would succeed:
use SimFeed.held (contact-based) or a real sensor to judge that.
"""
from __future__ import annotations

from ..fsm import TaskFsm, TaskState


class ProxyGraspMonitor:
    def __init__(self, fsm: TaskFsm):
        self.fsm = fsm

    def held(self) -> bool:
        state = self.fsm.state
        return (state is TaskState.TRANSFER
                or (state is TaskState.EXECUTE_GRASP and self.fsm.playback_finished)
                or (state is TaskState.PLACE and not self.fsm.playback_finished))
