import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
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
    assert reports[2]['status'] == 'partial'
    assert reports[3].get('after_unreachable') == 2
    _, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), [.5, 0., 0.]))
    assert reports[2]['status'] == 'partial' and reports[2]['reached_shift_mm'] <= reports[2]['max_shift_mm'] + 1e-6
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


def test_joining_cards_uses_the_recording_not_the_retargeted_result(robot, project):
    from fastapi.testclient import TestClient
    from motioncreator.server import app
    moved = move_cube(copy.deepcopy(project), [.05, 0., 0.])
    clip = TestClient(app).post('/api/keyframes-to-clip', json={'project': moved, 'start': 1, 'end': 2, 'fps': 30}).json()
    assert np.allclose(clip['samples'][-1], project['keyframes'][2]['qpos'], atol=1e-6)      # the recorded end, not the moved one


def test_the_right_hand_example_is_exactly_left_right_symmetric_and_stays_so_on_the_centre_line(robot):
    from motioncreator.grasp_template import mirrored
    mirror = json.loads((EXAMPLE.parent / 'pick_cards_right_mirrored.project.json').read_text())

    def asymmetry(frames):
        return max(np.degrees(np.abs(np.asarray(q)[7:] - mirrored(robot, np.asarray(q))[7:]).max())
                   for frame in frames for q in (frame['samples'][::5] if frame.get('samples') else [frame['qpos']]))
    assert asymmetry(mirror['keyframes']) < 1e-6
    assert abs(mirror['scene_objects'][0]['position'][1]) < 1e-4                  # on the body's centre line (the model is symmetric to ~10 µm)
    moved = move_cube(copy.deepcopy(mirror), [.05, 0., .03])
    resolved, reports = resolve_project_motions(robot, moved)
    assert reports[2]['status'] in ('ok', 'approx')
    assert asymmetry(resolved['keyframes']) < 1                                    # the retargeted motion is still symmetric


def test_a_flagged_card_reports_how_far_the_target_can_move_the_same_way(robot):
    mirror = json.loads((EXAMPLE.parent / 'pick_cards_right_mirrored.project.json').read_text())
    moved = move_cube(copy.deepcopy(mirror), [-.05, 0., 0.])
    _, reports = resolve_project_motions(robot, moved)
    flagged = reports[4]                                                   # 몸쪽으로 이동 hits a joint limit
    assert flagged['status'] == 'limit' and flagged['caused_by'] == 2
    assert reports[2]['blocks'] == [4] and reports[2]['safe_shift_mm'] == flagged['safe_shift_mm']      # the card to move is told too
    safe = np.array(flagged['safe_shift_mm'])
    assert 0 < flagged['safe_fraction'] < 1 and safe[0] < -20 and abs(safe[1]) < 1 and abs(safe[2]) < 1     # backwards only, less than asked
    within, beyond = (move_cube(copy.deepcopy(mirror), safe * f / 1000) for f in (.98, 1.15))
    assert not [r for r in resolve_project_motions(robot, within)[1].values() if r['status'] in ('limit', 'unreachable', 'partial')]
    assert [r for r in resolve_project_motions(robot, beyond)[1].values() if r['status'] in ('limit', 'unreachable', 'partial')]


def test_a_move_that_is_fine_gets_no_advice_and_costs_no_search(robot):
    mirror = json.loads((EXAMPLE.parent / 'pick_cards_right_mirrored.project.json').read_text())
    _, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(mirror), [.03, 0., .03]))
    assert all('safe_shift_mm' not in report for report in reports.values())


def test_a_point_target_gets_advice_too(robot):
    mirror = json.loads((EXAMPLE.parent / 'pick_cards_right_mirrored.project.json').read_text())
    mirror['keyframes'][5]['motion']['target']['point'] = [.9, 0., 1.0]        # far beyond the arms' reach
    _, reports = resolve_project_motions(robot, mirror)
    assert reports[5]['status'] in ('unreachable', 'partial')
    assert reports[5]['caused_by'] == 5 and reports[5]['safe_fraction'] < 1


@pytest.mark.parametrize('name', ['pick_all_keyframes', 'pick_all_keyframes_smooth'])
def test_the_all_keyframe_examples_have_no_clips_play_smoothly_and_keep_the_length(robot, name):
    project = json.loads((EXAMPLE.parent / f'{name}.project.json').read_text())
    assert not any(frame.get('samples') for frame in project['keyframes'])
    assert len(project['keyframes']) > 8
    assert any(frame.get('generated_transition') for frame in project['keyframes'])     # joined with continuous velocity, not stop and go
    motion = compile_motion(robot, project, 30)
    assert abs(motion['time'][-1] - sum(float(frame['duration']) for frame in project['keyframes'][1:])) < 1 / 30
    joints = np.asarray(motion['qpos'])[:, 7:]
    acceleration = np.linalg.norm(np.diff(np.diff(joints, axis=0) * 30, axis=0) * 30, axis=1)
    assert np.percentile(acceleration, 99) < 45                      # stop-and-go keyframes gave 112; the saved clips 31
    modes = [(frame.get('motion') or {}).get('mode') for frame in project['keyframes']]
    assert 'cartesian' in modes and 'delta' in modes
    resolved, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), [.03, 0., .03]))
    assert not [report for report in reports.values() if report['status'] in ('limit', 'unreachable', 'partial')]


def test_a_cube_beyond_reach_is_followed_as_far_as_the_arms_go(robot, project):
    """The hands end as close to the grasp point as they can reach, in the direction of the cube, not left at the recording."""
    shift = np.array([.12, 0., 0.])
    resolved, reports = resolve_project_motions(robot, move_cube(copy.deepcopy(project), shift))
    report = reports[2]
    assert report['status'] == 'partial' and 0 < report['reached_fraction'] < 1
    assert report['max_position_error_mm'] < 10                                          # what it reaches, it reaches properly
    recorded, moved = mid(robot, project['keyframes'][2]['qpos']), mid(robot, resolved['keyframes'][2]['qpos'])
    forward = (moved - recorded) @ shift / np.linalg.norm(shift)
    assert forward > .5 * report['reached_fraction'] * np.linalg.norm(shift)             # it did move toward the cube
    assert abs(forward - report['reached_shift_mm'] / 1000) < .02                         # and by the reported amount
    assert reports[2]['reached_shift_mm'] < reports[2]['shift_mm']


def test_a_position_delta_pose_card_keeps_the_recorded_hand_move_from_a_moved_previous_card(robot):
    """Joint deltas keep the joint change, which moves the hands differently once the arm pose changed; the position basis
    keeps the recorded TCP move and turn from where the previous card really ended."""
    project = json.loads((EXAMPLE.parent / 'pick_all_keyframes.project.json').read_text())
    frames = project['keyframes']
    linked = next(i for i, frame in enumerate(frames) if (frame.get('motion') or {}).get('mode') == 'cartesian')
    after = linked + 1
    assert (frames[after].get('motion') or {}).get('mode') == 'delta' and frames[after].get('samples') is None

    def hand_moves(basis):
        moved = move_cube(copy.deepcopy(project), [.05, 0., .05])
        moved['keyframes'][after]['motion'] = {'mode': 'delta', 'basis': basis}
        resolved, reports = resolve_project_motions(robot, moved, advise=False)
        before, now = (_hand_poses(robot, np.asarray(resolved['keyframes'][i]['qpos'])) for i in (linked, after))
        return {hand: now[hand][0] - before[hand][0] for hand in HANDS}, reports[after]
    recorded_before, recorded_now = (_hand_poses(robot, np.asarray(frames[i]['qpos'])) for i in (linked, after))
    recorded = {hand: recorded_now[hand][0] - recorded_before[hand][0] for hand in HANDS}
    position, report = hand_moves('position')
    joint, _ = hand_moves('joint')
    assert report['basis'] == 'position' and report['status'] in ('ok', 'approx')
    assert max(np.linalg.norm(position[hand] - recorded[hand]) for hand in HANDS) < .005
    assert max(np.linalg.norm(joint[hand] - recorded[hand]) for hand in HANDS) > max(np.linalg.norm(position[hand] - recorded[hand]) for hand in HANDS)
    # with the cube where it was recorded nothing changes, whichever basis
    still = copy.deepcopy(project)
    still['keyframes'][after]['motion'] = {'mode': 'delta', 'basis': 'position'}
    assert np.allclose(resolve_project_motions(robot, still, advise=False)[0]['keyframes'][after]['qpos'], frames[after]['qpos'], atol=1e-9)


def test_the_position_basis_is_for_pose_cards_only(robot, project):
    clip = next(i for i, frame in enumerate(project['keyframes']) if frame.get('samples') is not None and i > 0)
    project['keyframes'][clip]['motion'] = {'mode': 'delta', 'basis': 'position'}
    with pytest.raises(ValueError):
        resolve_project_motions(robot, project)
    with pytest.raises(ValueError):
        validate_motion({'mode': 'delta', 'basis': 'tcp'})
    assert validate_motion({'mode': 'delta'}) == {'mode': 'delta'} == validate_motion({'mode': 'delta', 'basis': 'joint'})


@pytest.mark.parametrize('basis', ['joint', 'position'])
def test_editing_a_card_carries_the_linked_cards_after_it_from_its_new_end(robot, basis):
    """Moving the approach pose moves the linked grasp cards with it; each keeps its recorded change from the card before."""
    from motioncreator.card_motion import carry_edit
    project = json.loads((EXAMPLE.parent / 'pick_all_keyframes.project.json').read_text())
    frames = project['keyframes']
    linked = next(i for i, frame in enumerate(frames) if (frame.get('motion') or {}).get('mode') == 'cartesian')
    chain = []
    for j in range(linked + 1, len(frames)):
        if (frames[j].get('motion') or {}).get('mode') != 'delta':
            break
        chain.append(j)
    assert len(chain) >= 2
    frames[chain[0]]['motion'] = {'mode': 'delta', 'basis': basis}
    edited = np.array(frames[linked]['qpos'], dtype=float)
    edited[7 + 15] += .1                    # left shoulder pitch
    edited[7 + 22] += .1                    # right shoulder pitch
    keyframes, reports = carry_edit(robot, project, linked, edited.tolist())
    assert np.allclose(keyframes[linked]['qpos'], edited)
    first = chain[0]
    if basis == 'joint':
        assert np.allclose(np.subtract(keyframes[first]['qpos'], keyframes[linked]['qpos'])[7 + 12:],
                           np.subtract(frames[first]['qpos'], frames[linked]['qpos'])[7 + 12:], atol=1e-9)
    else:
        assert reports[first]['status'] in ('ok', 'approx')
        before, now = (_hand_poses(robot, np.asarray(keyframes[i]['qpos'])) for i in (linked, first))
        old_before, old_now = (_hand_poses(robot, np.asarray(frames[i]['qpos'])) for i in (linked, first))
        for hand in HANDS:
            assert np.linalg.norm((now[hand][0] - before[hand][0]) - (old_now[hand][0] - old_before[hand][0])) < .005
    after = chain[-1] + 1
    if after < len(frames):
        assert keyframes[after]['qpos'] == frames[after]['qpos']           # the chain ends at the first card that is not linked
    assert not np.allclose(keyframes[chain[-1]]['qpos'], frames[chain[-1]]['qpos'])


def test_hands_are_read_and_placed_relative_to_the_ends_of_a_row(robot):
    from motioncreator.card_motion import hands_on_row
    project = json.loads((EXAMPLE.parent / 'pick_all_keyframes.project.json').read_text())
    linked = next(i for i, frame in enumerate(project['keyframes']) if (frame.get('motion') or {}).get('mode') == 'cartesian')
    q = project['keyframes'][linked]['qpos']
    across = {'shape': 'cylinder', 'size': [.13, .13, .6], 'position': [.42, .02, .88], 'quaternion_xyzw': [.7071068, 0, 0, .7071068]}
    turned = {**across, 'quaternion_xyzw': (Rotation.from_euler('z', 20, degrees=True) * Rotation.from_quat(across['quaternion_xyzw'])).as_quat().tolist()}
    placement = {'left_hand': [-.02, -.01, .08], 'right_hand': [-.03, .01, .09]}
    for row in (across, turned):
        for frame in ('object', 'robot'):
            placed = hands_on_row(robot, q, row, placement, frame)
            assert placed['status'] == 'ok' and placed['max_position_error_mm'] < 3
            for hand, value in placement.items():
                assert np.allclose(placed['placement'][hand], value, atol=.003)
    # straight across the robot the two frames agree; turned they do not
    assert np.allclose(hands_on_row(robot, q, across)['placement']['left_hand'], hands_on_row(robot, q, across, frame='robot')['placement']['left_hand'], atol=1e-6)
    assert not np.allclose(hands_on_row(robot, q, turned)['placement']['left_hand'], hands_on_row(robot, q, turned, frame='robot')['placement']['left_hand'], atol=1e-3)
    with pytest.raises(ValueError):
        hands_on_row(robot, q, {**across, 'shape': 'sphere'})
