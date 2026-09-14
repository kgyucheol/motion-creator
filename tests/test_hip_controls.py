import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motioncreator.robot import Robot, FEET
from motioncreator.motion import new_project, save_bundle
from motioncreator.reference import load_reference
from motioncreator.presets import GroupStore


@pytest.mark.parametrize('side', ['left', 'right'])
def test_combined_hip_uses_roll_anchor_and_complete_chain_orientation(side):
    robot = Robot()
    q = robot.home.copy()
    for axis, value in [('pitch', -.25), ('roll', .12), ('yaw', .18)]:
        q[robot.model.joint(f'{side}_hip_{axis}_joint').qposadr[0]] = value
    d = robot.data(q)
    position, rotation = robot.point(d, f'{side}_hip')
    assert np.allclose(position, d.xanchor[robot.model.joint(f'{side}_hip_roll_joint').id])
    assert np.allclose(rotation, d.xmat[robot.model.body(f'{side}_hip_yaw_link').id].reshape(3, 3))
    assert not np.allclose(rotation, d.xmat[robot.model.body(f'{side}_hip_roll_link').id].reshape(3, 3))


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_combined_hip_rotates_all_three_axes_with_support_foot_pinned(side, axis):
    robot = Robot()
    key = f'{side}_hip'
    before = robot.data(robot.home)
    _, rotation = robot.point(before, key)
    support = ('right' if side == 'left' else 'left') + '_foot'
    delta = np.zeros(3); delta[axis] = .04
    target = (Rotation.from_rotvec(delta) * Rotation.from_matrix(rotation)).as_quat()
    q, info = robot.solve(robot.home, robot.home, orientation_targets={key: target}, pins=[support, key])
    assert info['converged'], info
    assert info['angle_error_deg'] < .5
    after = robot.data(q)
    for pin in [support, key]:
        assert np.linalg.norm(robot.point(after, pin)[0] - robot.point(before, pin)[0]) < .003


def test_conflicting_hip_rotation_does_not_silently_release_pins():
    robot = Robot()
    before = robot.data(robot.home)
    target = (Rotation.from_rotvec([0, 0, .04]) * Rotation.from_matrix(robot.point(before, 'left_hip')[1])).as_quat()
    pins = [*FEET, 'left_hip']
    q, info = robot.solve(robot.home, robot.home, orientation_targets={'left_hip': target}, pins=pins)
    assert not info['converged']
    assert info['angle_error_deg'] > 2
    after = robot.data(q)
    for pin in pins:
        assert np.linalg.norm(robot.point(after, pin)[0] - robot.point(before, pin)[0]) < .003


def test_combined_hip_groups_and_motion_export(tmp_path):
    robot = Robot()
    hips = ['left_hip', 'right_hip']
    groups = GroupStore(tmp_path / 'groups.json')
    groups.save('양 고관절', hips)
    assert GroupStore(groups.path).list()[0]['members'] == hips
    project = new_project(robot)
    project['keyframes'][0]['pins'] += hips
    project['keyframes'].append({**project['keyframes'][0], 'duration': .2, 'name': 'Hold'})
    bundle = save_bundle(robot, project, fps=10, directory=tmp_path)
    with np.load(tmp_path / bundle['files'][1], allow_pickle=False) as data:
        assert data['posed_joints'].shape == (3, 34, 3)
    reference, _ = load_reference(tmp_path / bundle['files'][1])
    assert reference['qpos'].shape == (3, 36)
