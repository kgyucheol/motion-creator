import numpy as np

from ramen_controller.fsm import TaskState


def reach_observe(h):
    assert h.run(3, until="MANUAL_POSITION")
    h.operator.confirmed = True
    assert h.run(1, until="OBSERVE")


def test_full_cycle_visits_states_in_order_and_repeats(harness):
    h = harness()
    reach_observe(h)
    assert h.run(5, until="EXECUTE_GRASP")
    h.grasp.value = True                                   # hands close on the object
    assert h.run(10, until="PLACE")
    h.grasp.value = False                                  # released at the target
    assert h.run(10, until="OBSERVE")
    assert h.fsm.cycles == 1
    assert h.states() == ["MANUAL_POSITION", "OBSERVE", "PLAN_GRASP", "EXECUTE_GRASP", "TRANSFER",
                          "PLACE", "RETURN", "OBSERVE"]
    assert h.motion.planned[0].source.startswith("mean[")  # grasp planned from the averaged pose
    # second cycle starts without operator involvement
    assert h.run(5, until="EXECUTE_GRASP")


def test_transitions_carry_reasons(harness):
    h = harness()
    reach_observe(h)
    reasons = {event.state: event.reason for event in h.fsm.events}
    assert reasons["MANUAL_POSITION"] == "ready pose reached"
    assert reasons["OBSERVE"] == "operator confirmed position"


def test_manual_position_passes_operator_nav_until_confirmed_and_stopped(harness):
    h = harness()
    assert h.run(3, until="MANUAL_POSITION")
    h.operator.velocity = np.array([0.3, 0.0, 0.0])
    h.operator.confirmed = True                            # confirmed but still moving: must wait
    h.run(0.5)
    assert h.fsm.state is TaskState.MANUAL_POSITION
    assert np.allclose(h.commands[-1].nav, [0.3, 0, 0])
    h.operator.velocity = np.zeros(3)
    assert h.run(1, until="OBSERVE")
    assert np.allclose(h.commands[-1].nav, 0)


def test_observe_waits_for_full_window_then_averages(harness):
    h = harness(skip_manual_position=True)
    assert h.run(3, until="OBSERVE")
    entered = h.now
    assert h.run(5, until="PLAN_GRASP")
    assert h.now - entered >= 1.0


def test_observe_timeout_recovers_automatically_and_retries(harness):
    h = harness(skip_manual_position=True)
    h.vision.visible = False
    assert h.run(15, until="RECOVERY")
    assert "observation failed" in h.fsm.recovery_reason
    h.vision.visible = True
    assert h.run(5, until="EXECUTE_GRASP")


def test_automatic_retries_are_capped(harness):
    h = harness(skip_manual_position=True, max_auto_retries=1)
    h.vision.visible = False
    h.run(40)
    assert h.fsm.state is TaskState.RECOVERY
    assert h.fsm.retries == 1


def test_grasp_planning_failure_goes_to_recovery(harness):
    h = harness(skip_manual_position=True)
    h.motion.fail_plan = True
    assert h.run(10, until="RECOVERY")
    assert h.fsm.recovery_reason == "grasp planning failed: unreachable"


def test_unconfirmed_grasp_stops_for_operator(harness):
    h = harness(skip_manual_position=True)
    assert h.run(10, until="EXECUTE_GRASP")
    h.run(10, until="RECOVERY")                            # held() never becomes true
    assert h.fsm.recovery_reason == "grasp not confirmed"
    h.run(10)
    assert h.fsm.state is TaskState.RECOVERY               # no automatic retry with arms out of ready pose
    h.operator.reset = True
    assert h.run(1, until="INITIALIZE")


def test_object_lost_during_transfer(harness):
    h = harness(skip_manual_position=True)
    assert h.run(10, until="EXECUTE_GRASP")
    h.grasp.value = True
    assert h.run(10, until="TRANSFER")
    h.grasp.value = False
    assert h.run(1, until="RECOVERY")
    assert h.fsm.recovery_reason == "object lost during transfer"


def test_fall_enters_non_recoverable_recovery(harness):
    h = harness(skip_manual_position=True)
    assert h.run(5, until="OBSERVE")
    h.fallen = True
    h.run(0.1)
    assert h.fsm.state is TaskState.RECOVERY
    h.run(20)
    assert h.fsm.state is TaskState.RECOVERY               # stays until the operator resets after recovery
    h.operator.reset = True
    h.run(0.5)
    assert h.fsm.state is TaskState.RECOVERY               # reset refused while still fallen
    assert "INITIALIZE" not in h.states()                  # ...and never even attempted
    h.fallen = False
    h.operator.reset = True
    assert h.run(1, until="INITIALIZE")


def test_commands_are_continuous_across_segment_boundaries(harness):
    h = harness(skip_manual_position=True)
    assert h.run(5, until="EXECUTE_GRASP")
    h.grasp.value = True
    h.run(8)
    arms = np.stack([c.arm_targets for c in h.commands])
    assert np.abs(np.diff(arms, axis=0)).max() < 0.1       # no jumps between consecutive commands


def test_torso_command_is_clipped_to_limits(harness):
    from ramen_controller.contracts import WbcCommand, waist_to_torso_rpy
    limits = [0.52, 0.52, 2.618]
    assert np.all(np.abs(waist_to_torso_rpy([0.0, 1.5, 0.0], limits)) <= limits)
    assert np.allclose(waist_to_torso_rpy([0.0, 0.1, 0.0], limits), [0.1, 0.0, 0.0], atol=1e-9)


def test_no_automatic_retry_if_robot_falls_while_already_recovering(harness):
    h = harness(skip_manual_position=True)
    h.vision.visible = False
    assert h.run(15, until="RECOVERY")                     # recoverable observation failure
    h.fallen = True
    h.run(10)
    assert h.fsm.state is TaskState.RECOVERY and h.fsm.retries == 0


def test_command_carries_waist_joints_and_derives_torso_rpy(harness):
    from ramen_controller.contracts import WbcCommand
    command = WbcCommand(stamp=0, arm_targets=np.zeros(14), waist_ypr=[0.0, 0.1, 0.0], base_height=0.74)
    assert np.allclose(command.waist_ypr, [0, 0.1, 0])
    assert np.allclose(command.torso_rpy([0.52, 0.52, 2.618]), [0.1, 0, 0], atol=1e-9)


def test_transition_time_scales_with_the_pose_gap(harness):
    from ramen_controller.contracts import MotionKind
    h = harness(skip_manual_position=True, blend_s=3.0, blend_speed_rad_s=1.0)
    h.run(0.02)                                              # first step: INITIALIZE starts, arms were at 0.0
    near = h.fsm._playback                                   # READY segment holds 0.0: no gap, no blend
    assert near.blend_s == 0.0
    from controller_support import make_segment
    h.fsm._arms = np.zeros(14)
    h.fsm._play(make_segment(MotionKind.GRASP, arm_value=0.5), 0.0)
    assert np.isclose(h.fsm._playback.blend_s, 0.5)          # 0.5 rad at 1 rad/s
    h.fsm._play(make_segment(MotionKind.GRASP, arm_value=9.0), 0.0)
    assert h.fsm._playback.blend_s == 3.0                    # capped
