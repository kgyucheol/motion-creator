import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from motioncreator.robot import Robot, FEET
from motioncreator.demo import crouch_demo
from motioncreator.motion import compile_motion, save_bundle, validate_project
from motioncreator.server import app


@pytest.fixture(scope='module')
def robot():
    return Robot()


def test_grounded_real_29_dof_model(robot):
    assert robot.model.nq == 36 and robot.model.nv == 35
    assert len(robot.names) == 29
    assert abs(robot.state(robot.home)['floor_min_mm']) < .01


def test_crouch_keeps_feet_and_distant_hands(robot):
    target = robot.home[:3].copy()
    target[2] -= .15
    q, info = robot.solve(robot.home, robot.home, 'pelvis', target, FEET, max_nfev=60)
    assert info['converged'] and info['target_error_mm'] < 2
    a, b = robot.data(robot.home), robot.data(q)
    for key in FEET:
        p0, r0 = robot.point(a, key)
        p1, r1 = robot.point(b, key)
        assert np.linalg.norm(p1-p0) < .0001
        assert np.linalg.norm(r1-r0) < .001
    for key in ('left_hand', 'right_hand'):
        assert np.linalg.norm(robot.point(a, key)[0] - robot.point(b, key)[0]) < .015
    assert q[robot.model.joint('left_knee_joint').qposadr[0]] > 1.
    robot.validate_q(q)


def test_explicit_hand_pin_and_unreachable_target(robot):
    target = robot.home[:3].copy()
    target[2] -= .10
    q, info = robot.solve(robot.home, robot.home, 'pelvis', target, [*FEET, 'left_hand'], max_nfev=60)
    assert not info['rejected']
    assert np.linalg.norm(robot.point(robot.data(q), 'left_hand')[0] - robot.point(robot.data(robot.home), 'left_hand')[0]) < .003
    unreachable = robot.point(robot.data(q), 'right_hand')[0] + [2, 0, 0]
    out, info = robot.solve(q, q, 'right_hand', unreachable, FEET)
    assert not info['converged']
    robot.validate_q(out)
    for key in FEET:
        assert np.linalg.norm(robot.point(robot.data(q), key)[0] - robot.point(robot.data(out), key)[0]) < .003


def test_motion_endpoints_contacts_and_smoothness(robot):
    project = crouch_demo(robot)
    motion = compile_motion(robot, project, fps=30)
    assert motion['qpos'].shape == (121, 36)
    assert motion['qvel'].shape == (121, 35)
    assert np.allclose(np.diff(motion['time']), 1/30)
    assert motion['contacts'].all()
    assert np.array_equal(motion['qpos'][0], project['keyframes'][0]['qpos'])
    assert np.array_equal(motion['qpos'][60], project['keyframes'][1]['qpos'])
    assert np.array_equal(motion['qpos'][-1], project['keyframes'][2]['qpos'])
    # Catch branch changes and discontinuities at the endpoints after IK projection.
    assert np.max(np.abs(np.diff(motion['qpos'][:, 7:], axis=0))) < .10
    assert np.max(np.abs(motion['qvel'][[0, -1]])) < .05
    for key in FEET:
        initial = robot.point(robot.data(robot.home), key)[0]
        assert max(np.linalg.norm(robot.point(robot.data(q), key)[0] - initial) for q in motion['qpos']) < .001
    assert all(np.isfinite(motion[k]).all() for k in ('qpos', 'qvel', 'qacc'))


def test_project_and_npz_roundtrip(robot, tmp_path):
    project = crouch_demo(robot)
    project['name'] = '../../outside/한글 모션'
    bundle = save_bundle(robot, project, fps=15, directory=tmp_path)
    for filename in bundle['files']:
        assert (tmp_path / filename).parent == tmp_path
    editable = json.loads((tmp_path / bundle['files'][0]).read_text())
    validate_project(robot, editable)
    with np.load(tmp_path / bundle['files'][1], allow_pickle=False) as data:
        assert data['dof_pos'].shape == (61, 29)
        assert data['joint_names'].tolist() == robot.names
        assert data['root_quat_wxyz'].shape == (61, 4)
        assert np.allclose(np.linalg.norm(data['root_quat_wxyz'], axis=1), 1)
        assert json.loads(str(data['metadata_json']))['quaternion_order'] == 'wxyz'
        assert data['handle_pos'].shape == (61, 11, 3)


def test_invalid_projects_and_pin_conflicts(robot):
    p = crouch_demo(robot)
    p['model_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='fingerprint'):
        validate_project(robot, p)
    p = crouch_demo(robot)
    p['keyframes'][1]['qpos'][0] += .1
    with pytest.raises(ValueError, match='pinned'):
        compile_motion(robot, p)
    p = crouch_demo(robot)
    p['keyframes'][1]['qpos'][10] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        validate_project(robot, p)


def test_api_init_solve_and_bad_inputs(robot):
    with TestClient(app) as client:
        assert client.get('/api/health').json()['nq'] == 36
        init = client.get('/api/init').json()
        assert len(init['state']['handles']) == 11
        target = robot.home[:3].copy(); target[2] -= .1
        payload = {'qpos': robot.home.tolist(), 'anchor': robot.home.tolist(), 'focus': 'pelvis', 'target': target.tolist(), 'pins': list(FEET)}
        response = client.post('/api/solve', json=payload)
        assert response.status_code == 200, response.text
        assert response.json()['solver']['converged']
        payload['focus'] = 'left_foot'
        assert client.post('/api/solve', json=payload).status_code == 422
        assert client.post('/api/pose', json={'qpos': [0]}).status_code == 422
        assert client.get('/api/files/no_such_file.npz').status_code == 404
