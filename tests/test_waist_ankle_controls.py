import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from fastapi.testclient import TestClient

from motioncreator.robot import Robot, FEET
from motioncreator.motion import new_project, save_bundle
from motioncreator.reference import load_reference
from motioncreator.presets import GroupStore
from motioncreator.server import app


@pytest.mark.parametrize('key', ['waist', 'left_ankle', 'right_ankle'])
def test_combined_control_uses_roll_anchor(key):
    robot = Robot()
    q = robot.home.copy()
    for axis in ('roll', 'pitch'):
        q[robot.model.joint(key + '_' + axis + '_joint').qposadr[0]] += .1
    data = robot.data(q)
    position, orientation = robot.point(data, key)
    assert np.allclose(position, data.xanchor[robot.model.joint(key + '_roll_joint').id])
    orientation_body = 'torso_link' if key == 'waist' else key + '_roll_link'
    assert np.allclose(orientation, data.xmat[robot.model.body(orientation_body).id].reshape(3, 3))


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_waist_full_orientation_keeps_feet_fixed(axis):
    robot = Robot()
    before = robot.data(robot.home)
    delta = np.zeros(3); delta[axis] = .12
    target = (Rotation.from_rotvec(delta) * Rotation.from_matrix(robot.point(before, 'waist')[1])).as_quat()
    q, info = robot.solve(robot.home, robot.home, orientation_targets={'waist': target})
    assert info['converged'], info
    assert info['angle_error_deg'] < .5
    after = robot.data(q)
    for foot in FEET:
        p, r = robot.point(after, foot); bp, br = robot.point(before, foot)
        assert np.linalg.norm(p-bp) < .003
        assert Rotation.from_matrix(r @ br.T).magnitude() < .015


@pytest.mark.parametrize('side', ['left', 'right'])
def test_ankle_api_accepts_motor_angles_but_rejects_free_orientation(side):
    robot = Robot()
    start = robot.home.tolist()
    names = [side+'_ankle_roll_joint', side+'_ankle_pitch_joint']
    targets = {name: start[robot.model.joint(name).qposadr[0]] + .035 for name in names}
    with TestClient(app) as client:
        payload = dict(qpos=start, anchor=start, joints=targets, pins=list(FEET))
        response = client.post('/api/solve-group', json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['solver']['converged'], result['solver']
        for name in names:
            assert abs(result['state']['hinges'][name]['angle'] - targets[name]) < .01
        payload.pop('joints')
        payload['orientations'] = {side+'_ankle': [0, 0, 0, 1]}
        assert client.post('/api/solve-group', json=payload).status_code == 422


def test_waist_ankle_groups_and_export(tmp_path):
    robot = Robot()
    keys = ['waist', 'left_ankle', 'right_ankle']
    groups = GroupStore(tmp_path/'groups.json')
    groups.save('허리와 발목', keys)
    assert GroupStore(groups.path).list()[0]['members'] == keys
    project = new_project(robot)
    project['keyframes'][0]['pins'] += keys
    project['keyframes'].append({**project['keyframes'][0], 'duration': .2, 'name': 'Hold'})
    result = save_bundle(robot, project, fps=10, directory=tmp_path)
    with np.load(tmp_path/result['files'][1], allow_pickle=False) as data:
        assert data['posed_joints'].shape == (3, 34, 3)
        assert data['foot_contacts'].shape == (3, 4)
    reference, _ = load_reference(tmp_path/result['files'][1])
    assert reference['qpos'].shape == (3, 36)
    assert len(reference['joint_names']) == 29
