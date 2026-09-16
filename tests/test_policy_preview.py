"""Timeline physics contracts; actual ONNX integration is exercised separately."""
import copy
import json

import mujoco
import numpy as np
import pytest

from motioncreator import policy_preview as preview
from motioncreator.motion import new_project, validate_project
from motioncreator.robot import Robot


def test_free_physics_model_uses_named_torque_motors():
    robot = Robot()
    model = preview.build_model(robot)
    assert model.nq == 36 and model.nu == 29 and model.neq == 0
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    assert model.opt.timestep == .002
    assert model.joint('floating_base_joint').dofadr == 0
    assert np.all(model.dof_armature[6:] > 0)


def test_primitive_scene_objects_are_free_bodies_and_replay_from_authored_pose():
    robot = Robot()
    project = new_project(robot)
    project['keyframes'][0]['duration'] = .12
    project['scene_objects'] = [
        {'id': 'crate', 'name': 'Crate', 'shape': 'box', 'position': [2., 0., .8],
         'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.2, .3, .4], 'mass_kg': 2.,
         'friction': .6, 'color': '#336699', 'opacity': .5, 'visible': True},
        {'id': 'ball', 'name': 'Ball', 'shape': 'sphere', 'position': [2.5, 0., .8],
         'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.2, .2, .2], 'mass_kg': 1.,
         'friction': .4, 'color': '#993366', 'opacity': 1., 'visible': True},
        {'id': 'can', 'name': 'Can', 'shape': 'cylinder', 'position': [3., 0., .8],
         'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.2, .2, .4], 'mass_kg': 1.,
         'friction': .8, 'color': '#669933', 'opacity': .8, 'visible': True},
    ]
    validate_project(robot, project)
    model = preview.build_model(robot, project)
    assert model.nq == 57
    for index in range(3):
        assert model.joint(f'preview_object_joint_{index}').type == mujoco.mjtJoint.mjJNT_FREE
    result = preview.simulate(project, controller='pd')
    assert len(result['object_states']) == len(result['states']) == len(result['time'])
    np.testing.assert_allclose(result['object_states'][0]['crate']['position'], [2., 0., .8])
    assert result['object_states'][-1]['crate']['position'][2] < .8
    assert project['scene_objects'][0]['position'] == [2., 0., .8]


def test_scene_object_start_pose_is_raised_above_floor_for_its_rotation():
    robot = Robot()
    project = new_project(robot)
    project['scene_objects'] = [{
        'id': 'low-box', 'name': 'Low box', 'shape': 'box', 'position': [2., 0., -.5],
        'quaternion_xyzw': [0., np.sqrt(.5), 0., np.sqrt(.5)], 'size': [.2, .4, .6],
        'mass_kg': 1., 'friction': .7, 'color': '#336699', 'opacity': 1., 'visible': True,
    }]
    model = preview.build_model(robot, project)
    address = model.joint('preview_object_joint_0').qposadr[0]
    assert model.qpos0[address + 2] == pytest.approx(.1)


def test_simulated_replay_preserves_authoring_and_reports_tracking(monkeypatch):
    robot = Robot()
    project = new_project(robot)
    project['keyframes'][0]['duration'] = .12
    original = copy.deepcopy(project)

    class FixedTarget:
        def action(self, q, qvel, reference, time, stop_at):
            return robot.home[7:]

    monkeypatch.setattr(preview, 'SonicCPU', FixedTarget)
    updates = []
    result = preview.simulate(project, updates.append)
    assert project == original
    assert result['physics'] and result['policy'] == 'gear-sonic'
    assert result['summary']['reason'] == 'completed'
    assert result['time'][-1] == pytest.approx(.12)
    assert len(result['time']) == len(result['states']) == 4
    assert result['summary']['joint_rmse_rad'] > 0
    assert updates
    assert np.all(np.diff(result['time']) > 0)
    json.dumps(result, allow_nan=False)


def test_physics_preview_starts_at_selected_keyframe():
    robot = Robot()
    project = new_project(robot)
    project['keyframes'][0]['pins'] = []
    middle = copy.deepcopy(project['keyframes'][0])
    middle.update(name='Middle', duration=.1)
    middle['qpos'][7] += .1
    finish = copy.deepcopy(middle)
    finish.update(name='Finish', duration=.14)
    finish['qpos'][7] += .1
    project['keyframes'] += [middle, finish]
    original = copy.deepcopy(project)

    selected = preview.project_from_keyframe(project, 1)
    assert [frame['name'] for frame in selected['keyframes']] == ['Middle', 'Finish']
    assert preview.preview_duration(selected) == pytest.approx(.14)
    assert preview.preview_duration(preview.project_from_keyframe(project, 2)) == pytest.approx(.14)
    result = preview.simulate(project, controller='pd', start_frame_index=1)
    np.testing.assert_allclose(result['states'][0]['qpos'], middle['qpos'])
    assert result['summary']['reference_seconds'] == pytest.approx(.14)
    assert result['summary']['start_frame_index'] == 1
    assert result['summary']['start_frame_name'] == 'Middle'
    assert project == original
    with pytest.raises(ValueError, match='out of range'):
        preview.project_from_keyframe(project, 3)


def test_preview_detects_fall_and_rejects_long_timeline(monkeypatch):
    robot = Robot()
    project = new_project(robot)
    project['keyframes'][0]['qpos'][2] = .24
    project['keyframes'][0]['pins'] = []
    class FixedTarget:
        def action(self, *args):
            return robot.home[7:]
    monkeypatch.setattr(preview, 'SonicCPU', FixedTarget)
    result = preview.simulate(project)
    assert result['summary']['reason'] == 'fallen'
    assert result['summary']['sim_seconds'] < result['summary']['reference_seconds']
    project = new_project(robot)
    project['keyframes'] *= 3
    for frame in project['keyframes']:
        frame['duration'] = 40
    jobs = preview.PreviewJobs()
    with pytest.raises(ValueError, match='60초'):
        jobs.start(project)
    assert not jobs.jobs


def test_job_status_reports_completion_and_missing_jobs(tmp_path):
    jobs = preview.PreviewJobs()
    preview.atomic_json(tmp_path / 'progress.json', {'progress': .91})
    jobs.jobs['done'] = {'status': 'completed', 'folder': tmp_path}
    assert jobs.status('done')['progress'] == 1.
    with pytest.raises(KeyError):
        jobs.status('missing')


def test_worker_start_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(preview, 'runtime', lambda: {'available': True})
    def fail(*args, **kwargs):
        raise OSError('test interpreter unavailable')
    monkeypatch.setattr(preview.subprocess, 'Popen', fail)
    jobs = preview.PreviewJobs()
    with pytest.raises(ValueError, match='물리 CPU 작업을 시작할 수 없습니다'):
        jobs.start(new_project(Robot()))
    assert not jobs.jobs


def test_pd_preview_runs_real_physics_without_loading_sonic(monkeypatch):
    def forbidden():
        pytest.fail('PD-only physics must not load SONIC models')
    monkeypatch.setattr(preview, 'SonicCPU', forbidden)
    robot = Robot()
    project = new_project(robot)
    project['keyframes'][0]['duration'] = .12
    original = copy.deepcopy(project)
    result = preview.simulate(project, controller='pd')
    assert result['physics'] and result['policy'] is None
    assert result['controller'] == 'pd'
    assert result['summary']['policy_hz'] == 0
    assert result['time'][-1] == pytest.approx(.12)
    assert not np.allclose(result['states'][-1]['qpos'], robot.home)
    assert result['summary']['joint_rmse_rad'] > 0
    assert original == project
    json.dumps(result, allow_nan=False)


def test_pd_worker_uses_server_environment_without_sonic_assets(monkeypatch):
    monkeypatch.setattr(preview, 'runtime', lambda: {'available': False})
    calls = []
    def capture(args, **kwargs):
        calls.append(args)
        raise OSError('launch intercepted')
    monkeypatch.setattr(preview.subprocess, 'Popen', capture)
    jobs = preview.PreviewJobs()
    project = new_project(Robot())
    with pytest.raises(ValueError, match='launch intercepted'):
        jobs.start(project, controller='pd')
    assert calls[0][0] == preview.sys.executable
    assert calls[0][-2:] == ['pd', '0']
    with pytest.raises(ValueError, match='SONIC CPU 실행 환경'):
        jobs.start(project, controller='gear-sonic')
    with pytest.raises(ValueError, match='Unknown physics controller'):
        jobs.start(project, controller='unknown')
    assert len(calls) == 1


def test_preview_api_validates_controller_and_preserves_default():
    from pydantic import ValidationError
    from motioncreator.server import PhysicsPreviewInput
    assert PhysicsPreviewInput(project={}).controller == 'gear-sonic'
    assert PhysicsPreviewInput(project={}).start_frame_index == 0
    assert PhysicsPreviewInput(project={}, controller='pd').controller == 'pd'
    assert PhysicsPreviewInput(project={}, start_frame_index=3).start_frame_index == 3
    with pytest.raises(ValidationError):
        PhysicsPreviewInput(project={}, controller='unknown')
    with pytest.raises(ValidationError):
        PhysicsPreviewInput(project={}, start_frame_index=-1)
