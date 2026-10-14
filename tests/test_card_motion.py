import copy
import json
from pathlib import Path

import numpy as np
import pytest

from motioncreator.card_motion import resolve_project_motions, validate_motion
from motioncreator.grasp_template import HANDS, _hand_poses
from motioncreator.motion import compile_motion

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'pick_cards_modes.project.json'


@pytest.fixture(scope='module')
def robot():
    from motioncreator import server
    return server.robot            # the example project was made for the model the server starts with


@pytest.fixture()
def project():
    return json.loads(EXAMPLE.read_text())


def move_cube(project, delta):
    cube = project['scene_objects'][0]
    cube['position'] = (np.asarray(cube['position']) + delta).tolist()
    return project


def mid(robot, q):
    poses = _hand_poses(robot, np.asarray(q))
    return (poses['left_hand'][0] + poses['right_hand'][0]) / 2


def test_an_unmoved_object_changes_nothing(robot, project):
    still = compile_motion(robot, copy.deepcopy(project), 30)
    plain = copy.deepcopy(project)
    for frame in plain['keyframes']:
        frame.pop('motion')
    assert np.allclose(still['qpos'], compile_motion(robot, plain, 30)['qpos'], atol=1e-9)
    assert {report['status'] for report in still['motion_reports']} <= {'ok', 'unchanged'}


def test_moving_the_object_retargets_the_card_and_everything_after_it(robot, project):
    resolved, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), [.05, 0., 0.]))
    approach = reports[2]
    assert approach['status'] in ('ok', 'approx') and approach['max_position_error_mm'] < 15
    original, moved = project['keyframes'][2]['qpos'], resolved['keyframes'][2]['qpos']
    assert 0.04 < (mid(robot, moved) - mid(robot, original))[0] < 0.06          # the hands moved with the object
    grasp = resolved['keyframes'][3]['samples']
    assert np.allclose(grasp[0], moved, atol=1e-9)                              # the delta clip starts where the approach ended
    recorded = np.array(project['keyframes'][3]['samples'])
    assert np.allclose(np.array(grasp)[:, 7 + 12:7 + 29] - np.array(moved)[7 + 12:7 + 29],
                       recorded[:, 7 + 12:7 + 29] - recorded[0, 7 + 12:7 + 29], atol=1e-9)       # and keeps the recorded change


def test_the_target_card_pulls_the_following_delta_cards_and_the_last_card_returns_to_the_recording(robot, project):
    resolved, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), [.05, 0., 0.]))
    assert reports[3]['offset_deg'] > 1 and reports[4]['offset_deg'] > 1
    assert np.allclose(resolved['keyframes'][5]['qpos'], project['keyframes'][5]['qpos'], atol=1e-6)   # the carry target did not move
    assert np.allclose(resolved['keyframes'][-1]['samples'][-1], project['keyframes'][-1]['samples'][-1], atol=1e-9)


def test_the_total_time_does_not_change(robot, project):
    before = compile_motion(robot, copy.deepcopy(project), 30)
    after = compile_motion(robot, move_cube(copy.deepcopy(project), [.04, 0., .03]), 30)
    assert len(before['qpos']) == len(after['qpos'])
    assert np.allclose(before['keyframe_times'], after['keyframe_times'])


def test_a_point_target_moves_the_hands_to_the_coordinate(robot, project):
    project['keyframes'][5]['motion']['target']['point'] = [.35, 0., 1.0]
    resolved, reports = resolve_project_motions(robot, project)
    assert reports[5]['status'] in ('ok', 'approx')
    assert np.linalg.norm(mid(robot, resolved['keyframes'][5]['qpos']) - [.35, 0., 1.0]) < .015
    assert reports[6]['offset_deg'] > 1                                         # 팔 벌리기 is a delta card: it follows


def test_unreachable_far_and_missing_objects_are_reported_not_raised(robot, project):
    _, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), [.12, 0., 0.]))
    assert reports[2]['status'] == 'unreachable'
    assert reports[3].get('after_unreachable') == 2
    _, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), [.5, 0., 0.]))
    assert reports[2]['status'] == 'too_far'
    project['scene_objects'] = []
    _, reports = resolve_project_motions(robot, project)
    assert reports[2]['status'] == 'missing_object'


def test_the_original_motion_is_kept_for_comparison(robot, project):
    moved = compile_motion(robot, move_cube(copy.deepcopy(project), [.05, 0., 0.]), 30)
    still = compile_motion(robot, project, 30)
    assert moved['original_qpos'].shape == moved['qpos'].shape
    assert np.allclose(moved['original_qpos'], still['qpos'])


@pytest.mark.parametrize('spec', [
    {'mode': 'sideways'}, {'mode': 'cartesian'}, {'mode': 'cartesian', 'target': {'kind': 'point', 'point': [0, 0]}},
    {'mode': 'cartesian', 'target': {'kind': 'object', 'object_id': ''}},
    {'mode': 'cartesian', 'target': {'kind': 'point', 'point': [0, 0, 1]}, 'hands': ['middle']},
    {'mode': 'cartesian', 'target': {'kind': 'point', 'point': [0, 0, 1]}, 'offset': {'position': [2, 0, 0]}},
])
def test_invalid_motions_are_rejected(spec):
    with pytest.raises(ValueError):
        validate_motion(spec)


def test_cards_become_a_clip_and_back(robot, project):
    from fastapi.testclient import TestClient
    from motioncreator.server import app
    client = TestClient(app)
    response = client.post('/api/keyframes-to-clip', json={'project': project, 'start': 1, 'end': 2, 'fps': 30})
    assert response.status_code == 200
    clip = response.json()
    plain = copy.deepcopy(project)
    assert abs(clip['duration'] - float(plain['keyframes'][2]['duration'])) < 1 / 30 + 1e-9
    assert np.allclose(clip['samples'][0], project['keyframes'][1]['qpos'], atol=1e-6)
    assert np.allclose(clip['samples'][-1], project['keyframes'][2]['qpos'], atol=1e-6)
    project['keyframes'][1:3] = [clip]
    assert len(compile_motion(robot, project, 30)['qpos']) > 400
    assert client.post('/api/keyframes-to-clip', json={'project': project, 'start': 2, 'end': 2}).status_code == 422
