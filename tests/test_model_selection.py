import pytest
from fastapi import HTTPException
from motioncreator import server
from motioncreator.decoupled_wbc import sessions
from motioncreator.policy_preview import jobs
from motioncreator.robot import Robot


def test_model_selection_updates_editor_and_wbc_cache(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1')
    monkeypatch.setattr(server, 'robot', Robot('g1'))
    monkeypatch.setattr(sessions, 'robot', server.robot)
    monkeypatch.setattr(sessions, 'sessions', {})
    monkeypatch.setattr(jobs, 'jobs', {})
    monkeypatch.setattr(jobs, 'sonic_process', None)
    monkeypatch.setattr(Robot, 'export_visual', lambda self, path: None)
    assert server.select_model(server.ModelInput(model_id='g1-tools'))['model_id'] == 'g1-tools'
    assert server.initialize()['model_id'] == 'g1-tools'
    assert sessions.robot is None
    assert Robot().model_id == 'g1-tools'
    assert server.select_model(server.ModelInput(model_id='g1'))['model_id'] == 'g1'


def test_model_change_rejects_existing_simulation(monkeypatch):
    original = server.robot
    monkeypatch.setattr(sessions, 'sessions', {'active': object()})
    with pytest.raises(HTTPException) as error:
        server.select_model(server.ModelInput(model_id='g1-tools'))
    assert error.value.status_code == 409
    assert server.robot is original


def test_visual_route_is_model_specific_and_not_cached(monkeypatch):
    selected = Robot('g1')
    monkeypatch.setattr(server, 'robot', selected)
    response = server.model_visual('g1')
    assert str(response.path).endswith('robot-g1.glb')
    assert response.headers['cache-control'] == 'no-store, max-age=0'
    assert server.initialize()['state']['model_id'] == 'g1'
    assert server.initialize()['visual_revision'] == selected.fingerprint[:16]
    with pytest.raises(HTTPException) as error:
        server.model_visual('g1-tools')
    assert error.value.status_code == 409
