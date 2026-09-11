import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motioncreator.robot import Robot, FEET


@pytest.mark.parametrize('yaw', [0, np.pi/2])
@pytest.mark.parametrize('outward', [.025, -.025])
def test_symmetric_hand_targets_change_spacing_while_feet_stay_fixed(yaw, outward):
    robot = Robot()
    q = robot.home.copy()
    q[3:7] = Rotation.from_euler('z', yaw).as_quat()[[3,0,1,2]]
    before = robot.data(q)
    normal = Rotation.from_euler('z', yaw).apply([0,1,0])
    left = robot.point(before, 'left_hand')[0]
    right = robot.point(before, 'right_hand')[0]
    answer, info = robot.solve(q, q, selected_targets={
        'left_hand': left + outward*normal, 'right_hand': right - outward*normal,
    })
    assert info['converged'], info
    after = robot.data(answer)
    separation = np.dot(robot.point(after, 'left_hand')[0] - robot.point(after, 'right_hand')[0], normal)
    assert abs(separation - np.dot(left-right, normal) - 2*outward) < .003
    for foot in FEET:
        assert np.linalg.norm(robot.point(after, foot)[0] - robot.point(before, foot)[0]) < .003
