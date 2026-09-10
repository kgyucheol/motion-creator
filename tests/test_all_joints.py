"""All 29 actuated axes are selectable and use the model's real hinge geometry."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from motioncreator.robot import Robot, JOINT_HANDLES, FEET
from motioncreator.server import app
from motioncreator.presets import GroupStore
from motioncreator.motion import new_project, save_bundle, validate_project


@pytest.fixture(scope='module')
def robot():
    return Robot()


@pytest.mark.parametrize('name', list(JOINT_HANDLES))
def test_each_joint_anchor_axis_and_angle_control(robot, name):
    before = robot.data(robot.home)
    state = robot.state(robot.home)
    joint = robot.model.joint(name)
    handle = state['handles'][name]
    hinge = state['hinges'][name]
    assert np.allclose(handle['position'], before.xanchor[joint.id])
    # Check the published world axis independently through finite rotation of the joint body.
    moved = robot.home.copy()
    moved[joint.qposadr[0]] += .0001
    after = robot.data(moved)
    body = robot.ids[name]
    delta = Rotation.from_matrix(after.xmat[body].reshape(3, 3) @ before.xmat[body].reshape(3, 3).T).as_rotvec()
    assert np.allclose(delta / .0001, hinge['axis_world'], atol=1e-7)
    target = hinge['angle'] + .035
    q, info = robot.solve(robot.home, robot.home, joint_targets={name: target})
    assert info['converged'], (name, info)
    assert abs(q[joint.qposadr[0]] - target) < np.deg2rad(.5)
    for foot in FEET:
        assert np.linalg.norm(robot.point(robot.data(q), foot)[0] - robot.point(before, foot)[0]) < .003


def test_new_joint_groups_translation_pins_and_export(robot, tmp_path):
    names = ['left_wrist_pitch_joint', 'right_wrist_pitch_joint']
    store = GroupStore(tmp_path / 'groups.json')
    store.save('양 손목 피치', names)
    assert GroupStore(store.path).list()[0]['members'] == names
    initial = robot.state(robot.home)
    targets = {name: (np.array(initial['handles'][name]['position']) + [.015, 0, .015]).tolist() for name in names}
    with TestClient(app) as client:
        payload = dict(qpos=robot.home.tolist(), anchor=robot.home.tolist(), targets=targets, pins=list(FEET))
        result = client.post('/api/solve-group', json=payload)
        assert result.status_code == 200, result.text
        assert result.json()['solver']['converged']
        # A 29-anchor group fits the API contract, and spatial pin conflicts are still rejected.
        payload['targets'] = {name: initial['handles'][name]['position'] for name in robot.names}
        assert client.post('/api/solve-group', json=payload).status_code == 200
        payload['pins'] = [names[0]]
        assert client.post('/api/solve-group', json=payload).status_code == 422
    project = new_project(robot)
    project['pins'] = [*FEET, names[0]]
    project['keyframes'][0]['pins'] = project['pins']
    project['keyframes'].append({**project['keyframes'][0], 'name': 'Hold', 'duration': .2})
    validate_project(robot, project)
    bundle = save_bundle(robot, project, fps=10, directory=tmp_path)
    with np.load(tmp_path / bundle['files'][1], allow_pickle=False) as data:
        assert data['qpos'].shape == (3, 36)
        assert data['joint_names'].tolist() == robot.names
        assert set(robot.names) <= set(data['handle_names'].tolist())
        assert data['handle_pos'].shape[1] == data['handle_quat_wxyz'].shape[1] == 40
