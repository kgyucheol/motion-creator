"""Object-relative retargeting of an authored grasp (needs mujoco and the tool robot model)."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from scipy.spatial.transform import Rotation  # noqa: E402

from motioncreator.pose_adaptive_grasp import HANDS, RetargetedGraspPlanner  # noqa: E402
from motioncreator.robot import FEET, Robot  # noqa: E402
from ramen_controller.contracts import ObjectPose, RobotState  # noqa: E402
from ramen_controller.fsm import GraspPlanError  # noqa: E402

AUTHORED_CENTER = np.array([0.45, 0.0, 0.62])
AUTHORED_QUATERNION = Rotation.from_euler("y", -30, degrees=True).as_quat()


@pytest.fixture(scope="module")
def robot():
    return Robot("g1-tools", ("scoop", "scoop"))


def authored_motion(robot):
    """A short two-hand reach built by IK from the home pose, standing in for an authored grasp."""
    home = robot.home.copy()
    data = robot.data(home)
    left, right = (np.asarray(robot.point(data, hand)[0]) for hand in HANDS)
    frames, seed = [home], home
    for lift in (0.0, 0.06, 0.12):
        solved, info = robot.solve(seed, home, pins=[*FEET, "pelvis"], mode="free", resistance=0.,
                                   selected_targets={"left_hand": left + [0.12, 0, 0.1 + lift], "right_hand": right + [0.12, 0, 0.1 + lift]},
                                   max_nfev=200)
        assert info["converged"], info
        frames.append(solved)
        seed = solved
    samples = []
    for a, b in zip(frames, frames[1:]):
        samples.extend(a + (b - a) * w for w in np.linspace(0, 1, 11)[:-1])
    samples.append(frames[-1])
    return np.stack(samples), home


def state_from(qpos):
    return RobotState(stamp=0.0, joint_positions=qpos[7:], root_position=qpos[:3], root_quaternion_wxyz=qpos[3:7])


def base_pose(world_position, world_quaternion, root_qpos):
    """The observation a vision module would give: the object pose expressed in the base frame."""
    w, x, y, z = root_qpos[3:7]
    base = Rotation.from_quat([x, y, z, w])
    position = base.inv().apply(np.asarray(world_position) - root_qpos[:3])
    return ObjectPose(stamp=0.0, frame_id="base", position=position,
                      quaternion_xyzw=(base.inv() * Rotation.from_quat(world_quaternion)).as_quat())


def tcp_world(robot, joints, root_qpos):
    qpos = np.concatenate([root_qpos[:7], joints])
    data = robot.data(qpos)
    return {hand: np.asarray(robot.point(data, hand)[0]) for hand in HANDS}


@pytest.fixture(scope="module")
def planner_and_motion(robot):
    qpos, home = authored_motion(robot)
    planner = RetargetedGraspPlanner(robot, qpos, fps=10, authored_object_position=AUTHORED_CENTER,
                                     authored_object_quaternion_xyzw=AUTHORED_QUATERNION)
    return planner, qpos, home


def segment_joints(segment):
    return segment.arm_targets, segment.waist_ypr


def test_same_object_pose_reproduces_the_authored_tool_paths(robot, planner_and_motion):
    planner, qpos, home = planner_and_motion
    segment = planner.plan(base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, home), state_from(home))
    full = np.tile(home, (len(segment.arm_targets), 1))
    full[:, 22:36], full[:, 19:22] = segment.arm_targets, segment.waist_ypr
    for index in (0, len(full) // 2, len(full) - 1):
        authored = tcp_world(robot, qpos[min(int(index / (len(full) - 1) * (len(qpos) - 1)), len(qpos) - 1)][7:], home)
        planned = tcp_world(robot, full[index][7:], home)
        for hand in HANDS:
            assert np.linalg.norm(planned[hand] - authored[hand]) < 0.012, (index, hand)


def test_tool_paths_follow_the_object_when_it_is_somewhere_else(robot, planner_and_motion):
    planner, qpos, home = planner_and_motion
    shift = np.array([0.03, 0.02, 0.01])
    segment = planner.plan(base_pose(AUTHORED_CENTER + shift, AUTHORED_QUATERNION, home), state_from(home))
    same = planner.plan(base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, home), state_from(home))
    assert not np.allclose(segment.arm_targets, same.arm_targets, atol=1e-3)            # the plan really adapts
    last = len(segment.arm_targets) - 1
    for hand in HANDS:
        moved = tcp_world(robot, _joints(home, segment, last), home)[hand]
        reference = tcp_world(robot, _joints(home, same, last), home)[hand]
        assert np.linalg.norm((moved - reference) - shift) < 0.012, hand                 # shifted by the object's shift


def _joints(home, segment, index):
    joints = home[7:].copy()
    joints[12:15], joints[15:29] = segment.waist_ypr[index], segment.arm_targets[index]
    return joints


def test_robot_pose_matters_the_observation_is_converted_through_the_current_root(robot, planner_and_motion):
    planner, _, home = planner_and_motion
    stepped_back = home.copy()
    stepped_back[0] -= 0.04                                                            # the robot drifted back 4 cm
    # the object has not moved in the world, so the planned world-frame TCP paths must not change
    segment_home = planner.plan(base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, home), state_from(home))
    segment_back = planner.plan(base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, stepped_back), state_from(stepped_back))
    last = len(segment_home.arm_targets) - 1
    world_home = tcp_world(robot, _joints(home, segment_home, last), home)
    world_back = tcp_world(robot, _joints(home, segment_back, last), stepped_back)
    for hand in HANDS:
        assert np.linalg.norm(world_home[hand] - world_back[hand]) < 0.012, hand


def test_needs_the_robot_state_and_refuses_unreachable_or_foreign_poses(robot, planner_and_motion):
    planner, _, home = planner_and_motion
    observed = base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, home)
    with pytest.raises(GraspPlanError, match="current state"):
        planner.plan(observed, None)
    with pytest.raises(GraspPlanError, match="base frame"):
        planner.plan(ObjectPose(stamp=0, frame_id="camera", position=[0, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), state_from(home))
    far = base_pose(AUTHORED_CENTER + [0.4, 0, 0], AUTHORED_QUATERNION, home)
    with pytest.raises(GraspPlanError, match="authored"):
        planner.plan(far, state_from(home))
    unreachable = base_pose(AUTHORED_CENTER + [0.9, 0.0, 0.0], AUTHORED_QUATERNION, home)       # 1.35 m in front of the pelvis
    planner.max_object_shift_m = 1.5
    with pytest.raises(GraspPlanError, match="unreachable"):
        planner.plan(unreachable, state_from(home))


def test_authored_motion_shape_is_validated(robot):
    with pytest.raises(ValueError, match="authored_qpos"):
        RetargetedGraspPlanner(robot, np.zeros((1, 36)), 30, AUTHORED_CENTER, AUTHORED_QUATERNION)


def test_locking_the_waist_never_returns_a_plan_that_moves_it(robot, planner_and_motion):
    _, qpos, home = planner_and_motion
    locked = RetargetedGraspPlanner(robot, qpos, fps=10, authored_object_position=AUTHORED_CENTER,
                                    authored_object_quaternion_xyzw=AUTHORED_QUATERNION, lock_waist=True)
    assert locked.angle_pins == ["pelvis", "waist"]
    observed = base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, home)
    try:
        segment = locked.plan(observed, state_from(home))
    except GraspPlanError as error:                        # arms alone may not reach: that is an acceptable, explained outcome
        assert "unreachable" in str(error)
    else:
        assert np.ptp(segment.waist_ypr, axis=0).max() < 5e-3
    assert RetargetedGraspPlanner(robot, qpos, 10, AUTHORED_CENTER, AUTHORED_QUATERNION).angle_pins == ["pelvis"]


def test_a_joint_resting_a_hair_beyond_its_limit_does_not_stop_the_planner(robot, planner_and_motion):
    planner, _, home = planner_and_motion
    observed = base_pose(AUTHORED_CENTER, AUTHORED_QUATERNION, home)
    q = home.copy()
    knee = robot.model.joint("left_knee_joint")
    q[knee.qposadr[0]] = robot.model.jnt_range[knee.id, 1] + 1e-3          # measurement noise past the stop
    q[3:7] *= 1.0005                                                       # and a root quaternion off unit length
    segment = planner.plan(observed, state_from(q))
    assert segment.arm_targets.shape[1] == 14
