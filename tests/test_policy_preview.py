"""Timeline physics contracts; actual ONNX integration is exercised separately."""
import copy
import json

import numpy as np
import pytest

from motioncreator import policy_preview as preview
from motioncreator.motion import new_project
from motioncreator.robot import Robot


def test_free_physics_model_uses_named_torque_motors():
    robot = Robot()
    model = preview.build_model(robot)
    assert model.nq == 36 and model.nu == 29 and model.neq == 0
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    assert model.opt.timestep == .002
    assert model.joint('floating_base_joint').dofadr == 0
    assert np.all(model.dof_armature[6:] > 0)


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
    with pytest.raises(ValueError, match='SONIC CPU 작업을 시작할 수 없습니다'):
        jobs.start(new_project(Robot()))
    assert not jobs.jobs
