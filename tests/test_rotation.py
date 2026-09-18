import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from fastapi.testclient import TestClient
from motioncreator.robot import Robot, FEET, quat_matrix
from motioncreator.motion import new_project, compile_motion, save_bundle
from motioncreator.reference import load_reference
from motioncreator.server import app


@pytest.fixture(scope='module')
def robot():
    return Robot()


def target_rotation(robot, q, key, vector):
    return (Rotation.from_rotvec(vector) * Rotation.from_matrix(robot.point(robot.data(q), key)[1])).as_quat()


def assert_feet(robot, before, after):
    a, b = robot.data(before), robot.data(after)
    for key in FEET:
        p0, r0 = robot.point(a, key)
        p1, r1 = robot.point(b, key)
        assert np.linalg.norm(p1-p0) < .001
        assert Rotation.from_matrix(r1 @ r0.T).magnitude() < .001


def test_pinned_hand_rotates_without_releasing_position(robot):
    target = target_rotation(robot, robot.home, 'left_hand', [.2, 0, 0])
    q, info = robot.solve(robot.home, robot.home, pins=[*FEET, 'left_hand'], orientation_targets={'left_hand': target})
    assert info['converged'], info
    assert info['angle_error_deg'] < .05
    assert_feet(robot, robot.home, q)
    assert np.linalg.norm(robot.point(robot.data(q), 'left_hand')[0] - robot.point(robot.data(robot.home), 'left_hand')[0]) < .001


def test_pelvis_rotation_moves_free_root_and_preserves_feet(robot):
    target = target_rotation(robot, robot.home, 'pelvis', [0, .1, .03])
    q, info = robot.solve(robot.home, robot.home, orientation_targets={'pelvis': target})
    assert info['converged'], info
    assert info['angle_error_deg'] < .05
    assert not np.allclose(q[3:7], robot.home[3:7])
    assert_feet(robot, robot.home, q)
    robot.validate_q(q)


@pytest.mark.parametrize('key,end', [('left_elbow', 'left_hand'), ('right_elbow', 'right_hand'), ('left_knee', 'left_foot'), ('right_knee', 'right_foot')])
def test_hinge_angle_keeps_explicit_pins(robot, key, end):
    data = robot.data(robot.home)
    hinge = robot.state(robot.home)['hinges'][key]
    assert np.isclose(np.linalg.norm(hinge['axis_world']), 1)
    assert np.allclose(hinge['position'], robot.point(data, key)[0])
    target = hinge['angle'] + .06
    q, info = robot.solve(robot.home, robot.home, joint_targets={hinge['joint_name']: target}, pins=[*FEET, end] if end not in FEET else FEET)
    assert info['converged'], info
    assert info['angle_error_deg'] < .1
    assert abs(q[7+robot.names.index(hinge['joint_name'])]-target) < .002
    assert np.linalg.norm(robot.point(robot.data(q), end)[0]-robot.point(data, end)[0]) < .001
    assert_feet(robot, robot.home, q)


def test_motor_angle_precision_respects_pins(robot):
    targets = {'left_shoulder_yaw_joint': .25, 'right_shoulder_yaw_joint': -.2}
    q, info = robot.solve(robot.home, robot.home, joint_targets=targets)
    assert info['converged'], info
    for key, value in targets.items():
        assert abs(q[7+robot.names.index(key)]-value) < np.deg2rad(.05)
    assert_feet(robot, robot.home, q)


def test_angle_only_pins_allow_translation_but_hold_hand_orientation_and_joint_angle(robot):
    data = robot.data(robot.home)
    hand_position, hand_rotation = robot.point(data, 'left_hand')
    target = hand_position + np.array([.025, 0., .015])
    q, info = robot.solve(robot.home, robot.home, selected_targets={'left_hand': target},
                          angle_pins=['left_hand', 'right_elbow_joint'])
    assert info['converged'], info
    moved_position, moved_rotation = robot.point(robot.data(q), 'left_hand')
    assert np.linalg.norm(moved_position - hand_position) > .01
    assert Rotation.from_matrix(moved_rotation @ hand_rotation.T).magnitude() < np.deg2rad(.5)
    elbow = robot.model.joint('right_elbow_joint').qposadr[0]
    assert abs(q[elbow] - robot.home[elbow]) < np.deg2rad(.5)
    assert info['angle_pin_error_deg'] < .5


def test_angle_only_pin_is_preserved_through_keyframe_interpolation(robot):
    start_position, start_rotation = robot.point(robot.data(robot.home), 'left_hand')
    q, info = robot.solve(robot.home, robot.home,
                          selected_targets={'left_hand': start_position + [.025, 0., .015]},
                          angle_pins=['left_hand'])
    assert info['converged'], info
    project = new_project(robot)
    project['keyframes'][0]['angle_pins'] = ['left_hand']
    project['keyframes'].append({'name': 'Offset with fixed hand angle', 'duration': .5,
                                 'qpos': q.tolist(), 'pins': list(FEET),
                                 'angle_pins': ['left_hand']})
    motion = compile_motion(robot, project, fps=20)
    positions = []
    for pose in motion['qpos']:
        position, rotation = robot.point(robot.data(pose), 'left_hand')
        positions.append(position)
        assert Rotation.from_matrix(rotation @ start_rotation.T).magnitude() < np.deg2rad(.5)
    assert np.linalg.norm(positions[-1] - positions[0]) > .01


def test_rotation_validation(robot):
    for goals, match in [
        ({'orientation_targets': {'left_foot': [0, 0, 0, 1]}}, '고정'),
        ({'orientation_targets': {'left_elbow': [0, 0, 0, 1]}}, '회전'),
        ({'orientation_targets': {'left_hand': [0, 0, 0, 2]}}, 'normalized'),
        ({'joint_targets': {'left_elbow_joint': 100}}, 'limits'),
    ]:
        with pytest.raises(ValueError, match=match):
            robot.solve(robot.home, robot.home, **goals)
    q = robot.home.copy(); q[3] = 2
    with pytest.raises(ValueError, match='normalized'):
        robot.validate_q(q)


def test_rotation_export_interpolates_quaternions_and_keeps_contacts(robot, tmp_path):
    goals = {key: target_rotation(robot, robot.home, key, vector) for key, vector in
             [('pelvis', [0, .08, 0]), ('left_hand', [.18, 0, 0])]}
    q, info = robot.solve(robot.home, robot.home, pins=[*FEET, 'left_hand'], orientation_targets=goals)
    assert info['converged'], info
    project = new_project(robot, 'rotation test')
    project['keyframes'][0]['pins'].append('left_hand')
    project['keyframes'].append({'name': 'Rotate', 'duration': 2., 'qpos': q.tolist(), 'pins': [*FEET, 'left_hand']})
    bundle = save_bundle(robot, project, fps=15, directory=tmp_path)
    data, metadata = load_reference(tmp_path / bundle['npz_file'])
    assert data['qpos'].shape == (31, 36)
    assert np.allclose(np.linalg.norm(data['root_quat_wxyz'], axis=1), 1)
    assert np.max(np.abs(np.diff(data['qpos'][:, 7:], axis=0))) < .1
    assert np.max(np.abs(data['qvel'][[0, -1]])) < .03
    assert np.allclose(data['qpos'][-1], q, atol=2e-6)
    for frame in data['qpos']:
        assert_feet(robot, robot.home, frame)
    hand_positions = np.array([robot.point(robot.data(frame), 'left_hand')[0] for frame in data['qpos']])
    assert np.max(np.linalg.norm(hand_positions-hand_positions[0], axis=1)) < .001
    assert not metadata['validation']['dynamic_balance_checked']


def test_antipodal_quaternions_do_not_generate_spurious_rotation(robot):
    project = new_project(robot)
    q = robot.home.copy(); q[3:7] *= -1
    project['keyframes'].append({'name': 'Same orientation', 'duration': .5, 'qpos': q.tolist(), 'pins': []})
    motion = compile_motion(robot, project, fps=10)
    assert np.allclose(motion['qvel'], 0, atol=1e-9)
    assert np.allclose(motion['qpos'][:, 3:7], robot.home[3:7])
    assert np.allclose(quat_matrix(motion['qpos'][-1, 3:7]), quat_matrix(q[3:7]))


def test_orientation_only_api_request(robot):
    with TestClient(app) as client:
        state = client.get('/api/init').json()['state']
        assert len(state['hinges']) == 33 and 'poles' not in state
        payload = {'qpos': robot.home.tolist(), 'anchor': robot.home.tolist(),
                   'orientations': {'left_hand': target_rotation(robot, robot.home, 'left_hand', [.15, 0, 0]).tolist()}}
        response = client.post('/api/solve-group', json=payload)
        assert response.status_code == 200, response.text
        assert response.json()['solver']['converged']
        payload.pop('orientations')
        assert client.post('/api/solve-group', json=payload).status_code == 422
