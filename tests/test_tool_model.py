import mujoco
import numpy as np
import pytest

from motioncreator.robot import Robot
from motioncreator.motion import new_project, validate_project
from motioncreator.policy_preview import build_model, simulate
from motioncreator.decoupled_wbc import DecoupledSimulation, build_environment_model, verify_assets


def test_tool_urdf_joint_mapping_and_tcp_ik(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1')
    standard = Robot()
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    assert robot.names == standard.names
    assert robot.model.nq == 36 and robot.model.nu == 29
    assert [robot.model.joint(int(i)).name for i in robot.model.actuator_trnid[:, 0]] == robot.names
    assert 'scoop' in robot.handles['left_hand'][2]
    assert 'end_support' in robot.handles['right_hand'][2]
    np.testing.assert_allclose(robot.handles['left_hand'][1], [.180305, 0, -.0380589785])
    np.testing.assert_allclose(robot.handles['right_hand'][1], [.180305, 0, -.0380589785])
    assert robot.model.body('right_scoop_link').id == robot.ids['left_hand']
    assert robot.model.body('left_end_support_link').id == robot.ids['right_hand']
    home_data = robot.data(robot.home)
    np.testing.assert_allclose(robot.point(home_data, 'left_hand')[1][:, 2], [0, -1, 0], atol=1e-3)
    np.testing.assert_allclose(robot.point(home_data, 'right_hand')[1][:, 2], [0, 1, 0], atol=1e-3)
    left = robot.point(home_data, 'left_hand')[0]
    right = robot.point(home_data, 'right_hand')[0]
    np.testing.assert_allclose(right, left * [1, -1, 1], atol=2e-5, rtol=0)
    assert robot.fingerprint != standard.fingerprint
    camera = robot.state(robot.home)['cameras']['head']
    assert camera['label'] == 'Head Cam · D435'
    assert camera['calibrated_projection'] is False
    assert len(camera['position']) == 3 and len(camera['quaternion']) == 4
    for side in ('left', 'right'):
        target = robot.point(robot.data(robot.home), f'{side}_hand')[0] + [.01, 0, 0]
        q, result = robot.solve(robot.home, robot.home, selected_targets={f'{side}_hand': target})
        assert result['target_error_mm'] < 1
        assert np.isfinite(q).all()


def test_gripper_fingerprint_is_stable_and_pd_physics_runs(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    editor_robot = Robot()
    project = new_project(editor_robot)
    project['keyframes'][0]['duration'] = .1
    worker_robot = Robot()
    assert worker_robot.fingerprint == editor_robot.fingerprint
    assert len(editor_robot.compatible_fingerprints) == 6
    for old_hash in editor_robot.compatible_fingerprints - {editor_robot.fingerprint}:
        legacy_project = new_project(editor_robot)
        legacy_project['model_sha256'] = old_hash
        assert validate_project(worker_robot, legacy_project)['model_sha256'] == worker_robot.fingerprint
    base_project = new_project(Robot('g1'))
    assert validate_project(worker_robot, base_project)['model_sha256'] == worker_robot.fingerprint
    result = simulate(project, controller='pd')
    assert result['physics'] is True
    assert result['controller'] == 'pd'
    assert result['states'][0]['model_id'] == 'g1-tools'


@pytest.mark.parametrize('backend', ['preview', 'wbc'])
def test_tool_collision_and_fixed_carton_physics(monkeypatch, backend):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    project = new_project(robot)
    base = dict(quaternion_xyzw=[0, 0, 0, 1], mass_kg=1., friction=.7,
                color='#ffffff', opacity=1., visible=True)
    project['scene_objects'] = [
        dict(base, id='floor', name='Floor panel', shape='box', size=[.5, .5, .02], position=[2, 0, .4], fixed=True),
        dict(base, id='bundle', name='Bundle', shape='cylinder', size=[.1, .1, .23], position=[2, 0, .55],
             quaternion_xyzw=[0, 2**-.5, 0, 2**-.5]),
    ]
    model = build_model(robot, project) if backend == 'preview' else build_environment_model(project)
    prefix = 'preview' if backend == 'preview' else 'wbc'
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.neq == 1
    floor = model.body(f'{prefix}_object_0').id
    bundle = model.body(f'{prefix}_object_1').id
    model.opt.timestep = .002
    for _ in range(400):
        mujoco.mj_step(model, data)
    assert data.xpos[floor, 2] == pytest.approx(.4, abs=.002)
    assert data.xpos[bundle, 2] == pytest.approx(.46, abs=.01)
    # Both tools participate in object contact, including the scoop convex pieces.
    assert model.npair > 100


@pytest.mark.parametrize('backend', ['preview', 'wbc'])
def test_body_links_collide_with_scene_objects(monkeypatch, backend):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    project = new_project(robot)
    torso = robot.data(robot.home).xpos[robot.model.body('torso_link').id]
    base = dict(quaternion_xyzw=[0, 0, 0, 1], mass_kg=1., friction=.7,
                color='#ffffff', opacity=1., visible=True, fixed=False)
    project['scene_objects'] = [dict(base, id='chest', name='Chest box', shape='box', size=[.1, .1, .1],
                                     position=torso.tolist())]
    model = build_model(robot, project) if backend == 'preview' else build_environment_model(project)
    data = mujoco.MjData(model)
    data.qpos[:36] = robot.home
    mujoco.mj_forward(model, data)
    hands = {int(model.body(f'{side}_wrist_yaw_link').id) for side in ('left', 'right')}

    def is_hand(body):
        while body:
            if body in hands:
                return True
            body = int(model.body_parentid[body])
        return False

    prefix = 'preview' if backend == 'preview' else 'wbc'
    object_body = model.body(f'{prefix}_object_0').id
    touching = {model.body(int(model.geom_bodyid[g])).name
                for c in data.contact[:data.ncon] for g in (c.geom1, c.geom2)
                if int(model.geom_bodyid[g]) not in (0, object_body) and not is_hand(int(model.geom_bodyid[g]))}
    assert touching
    # A far-away object must not touch anything.
    data.qpos[36:39] = [3, 3, .5]
    mujoco.mj_forward(model, data)
    assert not any(int(model.geom_bodyid[g]) == object_body and int(model.geom_bodyid[h]) != 0
                   for c in data.contact[:data.ncon] for g, h in ((c.geom1, c.geom2), (c.geom2, c.geom1)))


@pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")
def test_tool_wbc_session_accepts_open_box_multi_geom_object(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    project = new_project(robot)
    project['keyframes'][0]['duration'] = .1
    project['scene_objects'] = [{
        'id': 'carton', 'name': 'Carton', 'shape': 'open_box',
        'position': [1.5, 0, .2], 'quaternion_xyzw': [0, 0, 0, 1],
        'size': [.72, .55, .4], 'wall_thickness_m': .02,
        'mass_kg': 2., 'friction': .7, 'color': '#9b6b3c',
        'opacity': 1., 'visible': True, 'fixed': True,
    }]

    simulation = DecoupledSimulation(robot, project, autostart=False)
    try:
        assert len(simulation.object_geoms['carton']) == 5
        assert simulation.snapshot()['state']['model_id'] == 'g1-tools'
    finally:
        simulation.close()


def test_attention_pose_is_mirror_symmetric_and_collision_free(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    data = robot.data(robot.attention)
    for left, right in (('left_hand', 'right_hand'), ('left_elbow', 'right_elbow'), ('left_foot', 'right_foot')):
        np.testing.assert_allclose(robot.point(data, right)[0], robot.point(data, left)[0] * [1, -1, 1], atol=2e-5, rtol=0)
    assert abs(min(robot.point(data, key)[0][2] for key in ('left_foot', 'right_foot'))) < 1e-9
    lower_body = [i for i, name in enumerate(robot.names) if name.split('_')[0] in ('left', 'right') and any(
        part in name for part in ('hip', 'knee', 'ankle')) or name.startswith('waist')]
    address = [robot.model.joint(robot.names[i]).qposadr[0] for i in lower_body]
    np.testing.assert_array_equal(robot.attention[:7], robot.home[:7])
    np.testing.assert_array_equal(robot.attention[address], robot.home[address])
    home_data = robot.data(robot.home)
    for key in ('left_foot', 'right_foot'):
        np.testing.assert_allclose(robot.point(data, key)[0], robot.point(home_data, key)[0], atol=1e-6)
    bodies = {frozenset((robot.model.body(int(robot.model.geom_bodyid[c.geom1])).name,
                         robot.model.body(int(robot.model.geom_bodyid[c.geom2])).name)) for c in data.contact[:data.ncon]}
    assert all('world' in pair for pair in bodies)
