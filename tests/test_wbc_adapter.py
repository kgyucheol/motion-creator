import numpy as np
import pytest

from ramen_controller.contracts import JOINT_NAMES, WbcCommand
from ramen_controller.wbc_adapter import UpperBodyLayout, build_goal, parse_state

WAIST = JOINT_NAMES[12:15]
ARMS = JOINT_NAMES[15:29]
HANDS = ("left_hand_index_0_joint", "right_hand_thumb_2_joint")


def command(**kw):
    base = dict(stamp=0.0, arm_targets=np.arange(14) / 10, waist_ypr=[0.1, 0.2, 0.3], base_height=0.7,
                nav=[0.2, 0.0, 0.1])
    base.update(kw)
    return WbcCommand(**base)


def test_vector_follows_layout_names_not_fixed_positions():
    layout = UpperBodyLayout((*ARMS[7:], *WAIST, *ARMS[:7]))                 # deliberately shuffled
    vector = layout.vector(command())
    assert np.allclose(vector[:7], np.arange(7, 14) / 10)                      # right arm first
    assert np.allclose(vector[7:10], [0.1, 0.2, 0.3])                          # then waist yaw, roll, pitch
    assert np.allclose(vector[10:], np.arange(7) / 10)


def test_hands_default_to_zero_and_accept_overrides():
    layout = UpperBodyLayout((*WAIST, *ARMS, *HANDS))
    assert np.allclose(layout.vector(command())[-2:], 0)
    assert np.allclose(layout.vector(command(), {HANDS[1]: 0.5})[-2:], [0, 0.5])


def test_layout_without_waist_drops_waist_values():
    layout = UpperBodyLayout(ARMS)
    assert not layout.has_waist and layout.vector(command()).shape == (14,)
    assert UpperBodyLayout((*WAIST, *ARMS)).has_waist


@pytest.mark.parametrize("names", [(*ARMS, ARMS[0]), (*ARMS, "left_knee_joint")])
def test_layout_rejects_duplicates_and_joints_it_cannot_fill(names):
    with pytest.raises(ValueError):
        UpperBodyLayout(names)


def test_goal_has_what_the_control_loop_requires():
    goal = build_goal(command(), UpperBodyLayout((*WAIST, *ARMS)), target_time=12.5)
    assert set(goal) == {"target_upper_body_pose", "navigate_cmd", "base_height_command", "target_time"}
    assert goal["target_time"] == 12.5 and goal["base_height_command"] == 0.7
    assert goal["navigate_cmd"].shape == (3,) and goal["target_upper_body_pose"].dtype == np.float64


def model_names():
    return [*JOINT_NAMES[:15], *ARMS, *(f"{s}_hand_{j}_joint" for s in ("left", "right") for j in range(7))]


def message(z=0.74, quat=(1, 0, 0, 0)):
    q = np.zeros(len(model_names()))
    q[model_names().index("left_elbow_joint")] = 1.25
    return {"q": q, "floating_base_pose": np.array([0.1, 0.2, z, *quat])}


def test_state_maps_joints_by_name_and_reads_base_pose():
    state = parse_state(message(), model_names(), stamp=3.0)
    assert state.joint_positions[JOINT_NAMES.index("left_elbow_joint")] == 1.25
    assert np.allclose(state.root_position, [0.1, 0.2, 0.74]) and not state.fallen and state.stamp == 3.0


def test_state_extra_hand_joints_are_ignored_but_wrong_length_is_rejected():
    assert parse_state(message(), model_names(), 0).joint_positions.shape == (29,)
    with pytest.raises(ValueError, match="joints"):
        parse_state({**message(), "q": np.zeros(5)}, model_names(), 0)


@pytest.mark.parametrize("z,quat,fallen", [
    (0.74, (1, 0, 0, 0), False),
    (0.30, (1, 0, 0, 0), True),                                   # too low
    (0.74, (np.cos(np.radians(40)), np.sin(np.radians(40)), 0, 0), True),   # pitched 80 deg about x
    (0.74, (np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0), False),  # 20 deg lean is not a fall
])
def test_fallen_detection(z, quat, fallen):
    assert parse_state(message(z, quat), model_names(), 0).fallen is fallen
