import json

import numpy as np

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


from ramen_controller.contracts import JOINT_NAMES  # noqa: E402
from ramen_controller.motion_bundle import BUNDLE_FORMAT  # noqa: E402


def write_bundle(folder, frames=31, fps=30, **overrides):
    qpos = np.zeros((frames, 36))
    qpos[:, 3] = 1.0
    qpos[:, 7 + 15:7 + 29] = np.linspace(0, 1, frames)[:, None]       # arms ramp 0 -> 1
    qpos[:, 7 + 12:7 + 15] = [0.1, 0.0, 0.2]                         # waist yaw, roll, pitch
    metadata = {"format": BUNDLE_FORMAT, "joint_names": list(JOINT_NAMES), "fps": fps, "samples": frames}
    metadata.update(overrides)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(json.dumps(metadata))
    np.savetxt(folder / "motion.csv", qpos, delimiter=",")
    return folder


from ramen_controller.operator import ScriptedOperator  # noqa: E402
from ramen_controller.runner import TaskRunner  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ScriptedLink:
    """Fresh state each call unless `silent`; remembers what was sent."""
    def __init__(self, clock):
        self.clock, self.silent, self.sent = clock, False, []
        self.arms = np.zeros(14)

    def read_state(self, stamp):
        if self.silent:
            return None
        joints = np.zeros(29)
        joints[15:29] = self.arms
        return RobotState(stamp=stamp, joint_positions=joints, root_position=np.array([0, 0, .75]),
                          root_quaternion_wxyz=np.array([1., 0, 0, 0]))

    def send(self, command):
        self.sent.append(command)
        self.arms = command.arm_targets.copy()           # the robot tracks the command perfectly


def Grasp():
    from ramen_controller.sim.proxy_grasp import ProxyGraspMonitor
    return ProxyGraspMonitor(None)


def make(clock, **config):
    op = ScriptedOperator()
    op.confirmed = True
    link = ScriptedLink(clock)
    grasp = Grasp()
    fsm = TaskFsm(vision=None, motion=FakeMotion(), operator=op, grasp=grasp,
                  config=TaskFsmConfig(**{"blend_s": 0.2, "skip_manual_position": True, **config}))
    grasp.fsm = fsm
    return fsm, TaskRunner(fsm, link, op, clock=clock, rate_hz=50), link, op, grasp
