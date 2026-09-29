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
    np.testing.assert_allclose(robot.handles['left_hand'][1], [.221805, 0, -.0380589785])
    np.testing.assert_allclose(robot.handles['right_hand'][1], [.221805, 0, -.0380589785])
    home_data = robot.data(robot.home)
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
    assert len(editor_robot.compatible_fingerprints) == 5
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
