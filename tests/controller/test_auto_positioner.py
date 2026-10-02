import numpy as np
from scipy.spatial.transform import Rotation

from ramen_controller.auto_positioner import AutoPositioner
from ramen_controller.contracts import RobotState
from ramen_controller.adapters.decoupled_wbc import bringup_sim


def state(x, y, yaw=0.0, t=0.0):
    q = Rotation.from_euler("z", yaw).as_quat()
    return RobotState(stamp=t, joint_positions=np.zeros(29), root_position=np.array([x, y, 0.75]),
                      root_quaternion_wxyz=np.array([q[3], q[0], q[1], q[2]]))


def test_walks_forward_when_the_target_is_ahead():
    p = AutoPositioner(target_xy_yaw=(0, 0, 0))
    p.observe(state(-1.0, 0.0))
    assert p.nav()[0] > 0 and abs(p.nav()[1]) < 1e-9 and p.nav()[2] == 0
    assert not p.position_confirmed()


def test_command_is_expressed_in_the_body_frame():
    p = AutoPositioner(target_xy_yaw=(0, 0, 0))
    p.observe(state(-1.0, 0.0, yaw=np.pi / 2))                    # robot faces world +y; the target is on its right
    assert p.nav()[0] < 0.05 and p.nav()[1] < -0.1
    p.observe(state(-1.0, 0.0, yaw=np.pi))                        # facing away: the target is behind
    assert p.nav()[0] < -0.1


def test_speed_is_clipped_and_small_commands_are_lifted_above_the_policy_threshold():
    p = AutoPositioner(target_xy_yaw=(0, 0, 0))
    p.observe(state(-5.0, 0.0))
    assert np.isclose(np.linalg.norm(p.nav()[:2]), p.max_linear)
    p.observe(state(-0.06, 0.0))                                   # kp * 0.06 = 0.05 < min
    assert np.isclose(np.linalg.norm(p.nav()[:2]), p.min_linear)


def test_turns_toward_the_target_yaw_the_short_way():
    p = AutoPositioner(target_xy_yaw=(0, 0, np.radians(170)))
    p.observe(state(0, 0, yaw=np.radians(-170)))                   # 20 deg apart across the +-180 seam
    assert p.nav()[2] < 0


def test_confirmation_needs_a_steady_hold_inside_tolerance():
    p = AutoPositioner(target_xy_yaw=(0, 0, 0), hold_s=1.0)
    p.observe(state(0.01, 0.0, t=0.0))
    assert not p.nav().any() and not p.position_confirmed()
    p.observe(state(0.01, 0.0, t=0.9))
    assert not p.position_confirmed()
    p.observe(state(0.01, 0.0, t=1.1))
    assert p.position_confirmed()


def test_leaving_the_tolerance_restarts_the_hold_timer():
    p = AutoPositioner(target_xy_yaw=(0, 0, 0), hold_s=1.0)
    p.observe(state(0.01, 0, t=0.0))
    p.observe(state(0.20, 0, t=0.8))                               # pushed out
    p.observe(state(0.01, 0, t=1.0))
    p.observe(state(0.01, 0, t=1.9))
    assert not p.position_confirmed()
    p.observe(state(0.01, 0, t=2.1))
    assert p.position_confirmed()


def test_reset_is_one_shot_and_clears_confirmation():
    p = AutoPositioner()
    p.observe(state(0, 0, t=0)); p.observe(state(0, 0, t=2))
    assert p.position_confirmed()
    p.request_reset()
    assert p.reset_requested() and not p.reset_requested() and not p.position_confirmed()


def test_bringup_sends_policy_activation_before_band_release_with_a_wait_between():
    log = []
    bringup_sim(lambda k: log.append(("key", k)), lambda s: log.append(("wait", s)), settle_s=2.5)
    assert log == [("key", "]"), ("wait", 2.5), ("key", "9")]
