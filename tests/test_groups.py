import numpy as np
import pytest
from fastapi.testclient import TestClient
from motioncreator.robot import Robot, FEET
from motioncreator.presets import GroupStore
from motioncreator import server


def test_two_hands_move_together_with_grounded_feet():
    robot = Robot()
    q = robot.home
    data = robot.data(q)
    hands = ['left_hand', 'right_hand']
    delta = np.array([.035, 0., .05])
    targets = {k: robot.point(data, k)[0] + delta for k in hands}
    result, info = robot.solve(q, q, pins=FEET, selected_targets=targets, max_nfev=60)
    actual = robot.data(result)
    assert info['converged'], info
    for k in hands:
        assert np.linalg.norm(robot.point(actual, k)[0] - targets[k]) < .004
    before = robot.point(data, hands[0])[0] - robot.point(data, hands[1])[0]
    after = robot.point(actual, hands[0])[0] - robot.point(actual, hands[1])[0]
    assert np.linalg.norm(before-after) < .004
    for k in FEET:
        assert np.linalg.norm(robot.point(actual, k)[0] - robot.point(data, k)[0]) < .001
    assert set(info['target_errors_mm']) == set(hands)


def test_group_target_validations():
    robot = Robot()
    with pytest.raises(ValueError, match='고정'):
        robot.solve(robot.home, robot.home, selected_targets={'left_hand': [0, 0, 1], 'left_foot': [0, 0, 0]})
    with pytest.raises(ValueError, match='Unknown'):
        robot.solve(robot.home, robot.home, selected_targets={'no_such_joint': [0, 0, 1]})
    with pytest.raises(ValueError, match='finite'):
        robot.solve(robot.home, robot.home, selected_targets={'left_hand': [float('nan'), 0, 1]})


def test_presets_survive_restart_and_edits(tmp_path):
    path = tmp_path / 'presets/groups.json'
    first = GroupStore(path)
    hands = first.save('양손', ['left_hand', 'right_hand'])
    restarted = GroupStore(path)
    assert restarted.list() == [hands]
    changed = restarted.save('양손과 골반', [*hands['members'], 'pelvis'], hands['id'])
    assert first.list() == [changed]
    with pytest.raises(ValueError, match='같은 이름'):
        first.save('양손과 골반', ['pelvis'])
    with pytest.raises(ValueError, match='중복'):
        first.save('잘못된 그룹', ['pelvis', 'pelvis'])
    restarted.delete(hands['id'])
    assert GroupStore(path).list() == []


def test_group_api_and_disk_persistence(tmp_path, monkeypatch):
    path = tmp_path / 'groups.json'
    monkeypatch.setattr(server, 'groups', GroupStore(path))
    with TestClient(server.app) as client:
        response = client.post('/api/groups', json={'name': '양손', 'members': ['left_hand', 'right_hand']})
        assert response.status_code == 200, response.text
        group = response.json()
        monkeypatch.setattr(server, 'groups', GroupStore(path))
        assert client.get('/api/groups').json() == [group]
        robot = server.robot
        targets = {k: (robot.point(robot.data(robot.home), k)[0] + [.02, 0, .03]).tolist() for k in group['members']}
        response = client.post('/api/solve-group', json={'qpos': robot.home.tolist(), 'anchor': robot.home.tolist(), 'targets': targets})
        assert response.status_code == 200, response.text
        assert response.json()['solver']['converged']
        assert client.delete('/api/groups/' + group['id']).status_code == 200
        assert client.get('/api/groups').json() == []
