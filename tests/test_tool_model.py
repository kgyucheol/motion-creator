import mujoco
import numpy as np
import pytest

from motioncreator.robot import Robot
from motioncreator.motion import new_project
from motioncreator.policy_preview import build_model
from motioncreator.decoupled_wbc import build_environment_model


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
    assert robot.handles['right_hand'][1][1] < 0
    assert robot.fingerprint != standard.fingerprint
    for side in ('left', 'right'):
        target = robot.point(robot.data(robot.home), f'{side}_hand')[0] + [.01, 0, 0]
        q, result = robot.solve(robot.home, robot.home, selected_targets={f'{side}_hand': target})
        assert result['target_error_mm'] < 1
        assert np.isfinite(q).all()


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
