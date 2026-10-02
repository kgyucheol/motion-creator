import numpy as np
import pytest

from ramen_controller.contracts import MotionKind, MotionSegment, ObjectPose, RobotState
from ramen_controller.fsm import GraspPlanError, TaskFsm, TaskFsmConfig


def make_segment(kind, arm_value=0.0, seconds=1.0, fps=10):
    frames = int(seconds * fps) + 1
    return MotionSegment(kind=kind, fps=fps, arm_targets=np.full((frames, 14), arm_value),
                         waist_ypr=np.zeros((frames, 3)))


def robot_state(now=0.0, fallen=False):
    return RobotState(stamp=now, joint_positions=np.zeros(29), root_position=np.array([0, 0, .74]),
                      root_quaternion_wxyz=np.array([1., 0, 0, 0]), fallen=fallen)


class FakeVision:
    def __init__(self):
        self.visible = True
        self.position = np.array([0.5, 0.0, 0.9])
        self.now = 0.0

    def detections(self, since):
        if not self.visible:
            return []
        return [ObjectPose(stamp=self.now, frame_id="base", position=self.position,
                           quaternion_xyzw=[0, 0, 0, 1], confidence=0.9, source="fake")]


class FakeMotion:
    def __init__(self):
        self.fail_plan = False
        self.planned = []

    def segment(self, kind):
        return make_segment(kind, arm_value={MotionKind.READY: 0.0, MotionKind.TRANSFER: 0.5,
                                             MotionKind.PLACE: 0.8, MotionKind.RETURN: 0.0}[kind])

    def plan_grasp(self, pose):
        if self.fail_plan:
            raise GraspPlanError("unreachable")
        self.planned.append(pose)
        return make_segment(MotionKind.GRASP, arm_value=0.3)


class FakeOperator:
    def __init__(self):
        self.velocity = np.zeros(3)
        self.confirmed = False
        self.reset = False

    def nav(self):
        return self.velocity

    def position_confirmed(self):
        return self.confirmed

    def reset_requested(self):
        flag, self.reset = self.reset, False
        return flag


class FakeGrasp:
    """held() follows the FSM stage the way a real sensor would: true from grasp until release."""
    def __init__(self):
        self.value = False

    def held(self):
        return self.value


class Harness:
    def __init__(self, **config):
        self.vision, self.motion, self.operator, self.grasp = FakeVision(), FakeMotion(), FakeOperator(), FakeGrasp()
        self.fsm = TaskFsm(self.vision, self.motion, self.operator, self.grasp,
                           TaskFsmConfig(**{"blend_s": 0.2, **config}))
        self.now = 0.0
        self.fallen = False
        self.commands = []

    def run(self, seconds, dt=0.02, until=None):
        end = self.now + seconds
        while self.now < end:
            self.now += dt
            self.vision.now = self.now
            self.commands.append(self.fsm.step(self.now, robot_state(self.now, self.fallen)))
            if until and self.fsm.state.value == until:
                return True
        return until is None

    def states(self):
        return [event.state for event in self.fsm.events if event.kind.value == "transition"]


@pytest.fixture
def harness():
    return Harness
