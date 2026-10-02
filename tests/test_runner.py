import json

import numpy as np

from ramen_controller.contracts import RobotState
from ramen_controller.fsm import TaskFsm, TaskFsmConfig, TaskState
from ramen_controller.operator import ScriptedOperator
from ramen_controller.runner import TaskRunner
from ramen_controller.vision import CameraFrame, MockPoseEstimator, VisionService
from tests.conftest import FakeMotion, robot_state


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
    from ramen_controller.proxy_grasp import ProxyGraspMonitor
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


def test_pauses_the_task_clock_and_sends_nothing_while_state_is_missing():
    clock = Clock()
    fsm, runner, link, *_ = make(clock)
    for _ in range(5):
        runner.tick(); clock.sleep(0.02)
    advanced, sent = runner.task_time, len(link.sent)
    link.silent = True
    for _ in range(80):                                    # 1.6 s of silence
        runner.tick(); clock.sleep(0.02)
    assert runner.stalled and len(link.sent) == sent and runner.task_time == advanced
    link.silent = False
    runner.tick()
    assert not runner.stalled
    assert runner.task_time - advanced <= 2 * runner.period + 1e-9      # resumed without a time jump
    reasons = [e.reason for e in runner.recent_events]
    assert sum("task paused" in r for r in reasons) == 1 and any("resumed" in r for r in reasons)


def test_a_long_wall_clock_gap_cannot_jump_the_task_forward():
    clock = Clock()
    fsm, runner, *_ = make(clock)
    runner.tick()
    clock.now += 30.0                                      # process was suspended
    runner.tick()
    assert runner.task_time <= 2 * runner.period + 1e-9


def test_operator_observe_is_called_with_each_state():
    clock = Clock()
    fsm, runner, link, op, _ = make(clock)
    seen = []
    op.observe = seen.append
    runner.tick(); runner.tick()
    assert len(seen) == 2


def test_snapshot_is_json_serialisable_and_reports_the_state():
    clock = Clock()
    fsm, runner, *_ = make(clock)
    runner.tick()
    snapshot = json.loads(json.dumps(runner.snapshot()))
    assert snapshot["state"] == "INITIALIZE" and snapshot["robot"]["fallen"] is False and snapshot["events"] == []


def test_full_cycle_through_vision_service_with_a_noisy_mock_backend():
    clock = Clock()
    fsm, runner, link, op, grasp = make(clock)
    truth = __import__("ramen_controller.contracts", fromlist=["ObjectPose"]).ObjectPose(
        stamp=0, frame_id="base", position=[0.5, 0, 0.9], quaternion_xyzw=[0, 0, 0, 1])

    class Camera:
        def latest(self):
            return CameraFrame(stamp=clock.now, rgb=np.zeros((4, 4, 3), np.uint8), K=np.eye(3), T_base_cam=np.eye(4))

    fsm.vision = VisionService(Camera(), MockPoseEstimator(truth=lambda: truth, dropout_probability=0.2,
                                                         outlier_probability=0.1, seed=1))
    runner.run(duration_s=60, until=lambda: fsm.cycles >= 2, sleep=clock.sleep)
    assert fsm.cycles >= 2, [e.reason for e in fsm.events][-6:]
    order = [e.state for e in fsm.events if e.kind.value == "transition"]
    assert order[:7] == ["OBSERVE", "PLAN_GRASP", "EXECUTE_GRASP", "TRANSFER", "PLACE", "RETURN", "OBSERVE"]
