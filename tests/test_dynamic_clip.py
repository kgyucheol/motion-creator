import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motioncreator.dynamic_clip import (ARM_WAIST_COLUMNS, resolve_dynamic_clip, resolve_project_dynamics,
                                        validate_dynamic, window_weights)
from motioncreator.motion import compile_motion, project_from_motion_bytes, validate_project
from motioncreator.robot import Robot

FRAMES = 61
WINDOW = {'ramp_in_start': 10, 'hold_start': 20, 'hold_end': 40, 'ramp_out_end': 50}
REFERENCE = {'position': [.4, 0., .8], 'quaternion_xyzw': [0., 0., 0., 1.]}


def episode():
    """Both arms reach forward and back; every pose is well inside the joint limits."""
    base = np.array([-.5, .2, 0., .9, 0., 0., 0.])
    frames = []
    for i in range(FRAMES):
        swing = .25 * np.sin(np.pi * i / (FRAMES - 1))
        left = (base + [-swing, 0, 0, swing, 0, 0, 0]).tolist()
        right = (base * [1, -1, 1, 1, 1, 1, 1] + [-swing, 0, 0, swing, 0, 0, 0]).tolist()
        arm = lambda q: {'qpos': q, 'qvel': [], 'torque': []}
        frames.append({'actions': {'left_arm': arm(left), 'right_arm': arm(right), 'body': {'qpos': []}},
                       'states': {'left_arm': arm(left), 'right_arm': arm(right), 'body': {'qpos': []}}})
    return json.dumps({'info': {'image': {'fps': 30.0}}, 'text': {'goal': 'g'}, 'data': frames}).encode()


@pytest.fixture(scope='module')
def robot():
    return Robot()


@pytest.fixture()
def project(robot):
    project = project_from_motion_bytes(robot, episode(), 'x.json', 30)
    project['keyframes'][0]['dynamic'] = {'object_id': 'cube', 'reference_pose': copy.deepcopy(REFERENCE),
                                          'hands': ['left', 'right'], 'window': dict(WINDOW)}
    project['scene_objects'] = [{'id': 'cube', 'name': 'Cube', 'shape': 'box', 'position': [.4, 0., .8],
                                 'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.1, .1, .1], 'mass_kg': 1.,
                                 'friction': .7, 'color': '#b98853', 'opacity': 1., 'visible': True}]
    return project


def move(project, position, quaternion=(0., 0., 0., 1.)):
    project['scene_objects'][0]['position'] = list(position)
    project['scene_objects'][0]['quaternion_xyzw'] = list(quaternion)
    return project


def hands(robot, sample):
    data = robot.data(sample)
    return {side: robot.point(data, f'{side}_hand') for side in ('left', 'right')}


def test_window_weights_ramp_hold_ramp():
    weights = window_weights(WINDOW, FRAMES)
    assert np.all(weights[:10] == 0) and np.all(weights[51:] == 0)
    assert weights[10] == 0 and np.all(weights[20:41] == 1) and weights[50] == 0
    assert np.all(np.diff(weights[10:21]) >= 0) and np.all(np.diff(weights[40:51]) <= 0)


def test_unmoved_object_leaves_the_clip_untouched(robot, project):
    resolved, reports = resolve_project_dynamics(robot, project)
    assert reports[0]['status'] == 'unchanged'
    assert resolved['keyframes'][0]['samples'] == project['keyframes'][0]['samples']


def test_moved_object_carries_the_hands_during_the_hold(robot, project):
    clip = project['keyframes'][0]
    shift = np.array([.04, -.03, .02])
    samples, report = resolve_dynamic_clip(robot, clip, np.array(REFERENCE['position']) + shift, REFERENCE['quaternion_xyzw'])
    original = np.asarray(clip['samples'])
    assert report['status'] == 'ok' and report['max_position_error_mm'] < 3
    for index in (20, 30, 40):
        before, after = hands(robot, original[index]), hands(robot, samples[index])
        for side in before:
            assert np.linalg.norm(after[side][0] - (before[side][0] + shift)) < .003
    # outside the window nothing moves, and only arms and waist ever change
    assert np.array_equal(samples[:10], original[:10]) and np.array_equal(samples[50:], original[50:])
    unchanged = np.ones(original.shape[1], dtype=bool)
    unchanged[ARM_WAIST_COLUMNS] = False
    assert np.array_equal(samples[:, unchanged], original[:, unchanged])


def test_yaw_of_the_object_turns_the_hands_around_it(robot, project):
    clip = project['keyframes'][0]
    yaw = Rotation.from_euler('z', 10, degrees=True)
    samples, report = resolve_dynamic_clip(robot, clip, REFERENCE['position'], yaw.as_quat())
    original = np.asarray(clip['samples'])
    assert report['status'] in ('ok', 'approx') and report['rotation_deg'] == pytest.approx(10)
    centre = np.array(REFERENCE['position'])
    before, after = hands(robot, original[30])['left'][0], hands(robot, samples[30])['left'][0]
    assert np.linalg.norm(after - (centre + yaw.apply(before - centre))) < .01


def test_tilt_is_ignored_unless_full_freedom_is_asked_for(robot, project):
    tilt = Rotation.from_euler('x', 15, degrees=True).as_quat()
    flat = validate_dynamic(project['keyframes'][0]['dynamic'], FRAMES)
    _, report = resolve_dynamic_clip(robot, project['keyframes'][0], REFERENCE['position'], tilt, spec=flat)
    assert report['status'] == 'unchanged'
    _, report = resolve_dynamic_clip(robot, project['keyframes'][0], REFERENCE['position'], tilt,
                                     spec={**flat, 'freedom': 'full6d'})
    assert report['rotation_deg'] == pytest.approx(15)


def test_far_objects_and_missing_objects_are_reported_not_raised(robot, project):
    resolved, reports = resolve_project_dynamics(robot, move(project, [.4 + .5, 0., .8]))
    assert reports[0]['status'] == 'too_far'
    assert resolved['keyframes'][0]['samples'] == project['keyframes'][0]['samples']
    project['scene_objects'] = []
    assert resolve_project_dynamics(robot, project)[1][0]['status'] == 'missing_object'


def test_unreachable_target_is_flagged(robot, project):
    project['keyframes'][0]['dynamic']['max_shift_m'] = 2.
    _, reports = resolve_project_dynamics(robot, move(project, [.4 + 1.2, 0., .8]))
    assert reports[0]['status'] == 'unreachable'


def test_compile_follows_the_object_and_reports(robot, project):
    still = compile_motion(robot, copy.deepcopy(project), 30)
    moved = compile_motion(robot, move(copy.deepcopy(project), [.45, .02, .8]), 30)
    assert still['dynamic_reports'][0]['status'] == 'unchanged'
    assert moved['dynamic_reports'][0]['status'] == 'ok' and moved['dynamic_reports'][0]['index'] == 0
    assert len(still['qpos']) == len(moved['qpos']) and abs(still['time'][-1] - moved['time'][-1]) < 1e-9
    assert not np.allclose(still['qpos'][30], moved['qpos'][30])
    assert np.allclose(still['qpos'][0], moved['qpos'][0]) and np.allclose(still['qpos'][-1], moved['qpos'][-1])


def test_compile_keeps_the_uncorrected_motion_for_comparison(robot, project):
    still = compile_motion(robot, copy.deepcopy(project), 30)
    moved = compile_motion(robot, move(copy.deepcopy(project), [.45, .02, .8]), 30)
    original = moved['original_qpos']
    assert original.shape == moved['qpos'].shape
    assert np.allclose(original, still['qpos'])            # what the clip plays without the correction
    assert not np.allclose(original[30], moved['qpos'][30])
    plain = copy.deepcopy(project)
    del plain['keyframes'][0]['dynamic']
    assert 'original_qpos' not in compile_motion(robot, plain, 30)


@pytest.mark.parametrize('change', [
    lambda d: d.update(window={**WINDOW, 'hold_end': 15}),
    lambda d: d.update(window={'ramp_in_start': FRAMES, 'hold_start': FRAMES + 2, 'hold_end': FRAMES + 3, 'ramp_out_end': FRAMES + 4}),
    lambda d: d.update(window={'ramp_in_start': -9, 'hold_start': -8, 'hold_end': -7, 'ramp_out_end': -1}),
    lambda d: d.update(hands=['middle']),
    lambda d: d.update(freedom='sideways'),
    lambda d: d.update(reference_pose={'position': [0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}),
    lambda d: d.update(offset={'position': [2., 0., 0.]}),
    lambda d: d.update(offset={'position': [0., 0., 0.], 'yaw_deg': 270}),
    lambda d: d.update(object_id=7),
])
def test_invalid_bindings_are_rejected(robot, project, change):
    change(project['keyframes'][0]['dynamic'])
    with pytest.raises(ValueError):
        validate_project(robot, project)


def test_only_clips_take_a_binding(robot, project):
    from motioncreator.motion import new_project
    pose = new_project(robot)['keyframes'][0]
    pose['dynamic'] = copy.deepcopy(project['keyframes'][0]['dynamic'])
    project['keyframes'].append(pose)
    with pytest.raises(ValueError):
        validate_project(robot, project)


def split_at(project, at):
    """Cut the clip into two that share sample `at`, the way the editor does, shifting the window numbers."""
    clip = project['keyframes'][0]
    samples, duration, count = clip['samples'], clip['duration'], len(clip['samples'])
    parts = []
    for first, offset in ((samples[:at + 1], 0), (samples[at:], at)):
        window = {key: value - offset for key, value in clip['dynamic']['window'].items()}
        parts.append({**clip, 'samples': first, 'qpos': first[0], 'duration': duration * (len(first) - 1) / (count - 1),
                      'dynamic': {**clip['dynamic'], 'window': window}})
    return {**project, 'keyframes': parts}


@pytest.mark.parametrize('at', [15, 30, 45])
def test_a_split_clip_resolves_like_the_whole_one(robot, project, at):
    move(project, [.45, .02, .8])
    whole = compile_motion(robot, copy.deepcopy(project), 30)
    split = compile_motion(robot, split_at(project, at), 30)
    assert len(whole['qpos']) == len(split['qpos']) and abs(whole['time'][-1] - split['time'][-1]) < 1e-9
    assert np.abs(whole['qpos'] - split['qpos']).max() < 1e-2        # grids differ inside each part, so only about half a degree apart; the same length shows no entry transition was added
    assert [report['index'] for report in split['dynamic_reports']] == [0, 1]


def test_resolve_endpoint_returns_samples_and_report(project):
    from fastapi import HTTPException
    from motioncreator import server
    live = project_from_motion_bytes(server.robot, episode(), 'x.json', 30)
    live['keyframes'][0]['dynamic'] = project['keyframes'][0]['dynamic']
    live['scene_objects'] = move(project, [.45, .02, .8])['scene_objects']
    result = server.dynamic_resolve(server.DynamicResolveInput(project=live, index=0))
    assert result['report']['status'] == 'ok' and len(result['samples']) == FRAMES
    assert not np.allclose(result['samples'][30], live['keyframes'][0]['samples'][30])
    del live['keyframes'][0]['dynamic']
    with pytest.raises(HTTPException):
        server.dynamic_resolve(server.DynamicResolveInput(project=live, index=0))


def test_a_tcp_offset_moves_the_grip_without_any_object(robot, project):
    clip = project['keyframes'][0]
    clip['dynamic'].update(object_id=None, offset={'position': [.03, -.02, .01], 'yaw_deg': 0.})
    project['scene_objects'] = []                              # no object needed
    resolved, reports = resolve_project_dynamics(robot, project)
    assert reports[0]['status'] == 'ok' and reports[0]['object_id'] is None
    original, moved = np.asarray(clip['samples']), np.asarray(resolved['keyframes'][0]['samples'])
    for index in (20, 30, 40):
        before, after = hands(robot, original[index]), hands(robot, moved[index])
        for side in before:
            assert np.linalg.norm(after[side][0] - (before[side][0] + [.03, -.02, .01])) < .003
    assert np.array_equal(moved[:10], original[:10]) and np.array_equal(moved[50:], original[50:])


def test_the_offset_adds_to_the_object_move(robot, project):
    clip = project['keyframes'][0]
    clip['dynamic']['offset'] = {'position': [0., 0., .02], 'yaw_deg': 0.}
    shift = np.array([.03, 0., 0.])
    samples, report = resolve_dynamic_clip(robot, clip, np.array(REFERENCE['position']) + shift, REFERENCE['quaternion_xyzw'])
    before, after = hands(robot, np.asarray(clip['samples'])[30]), hands(robot, samples[30])
    assert report['status'] == 'ok'
    for side in before:
        assert np.linalg.norm(after[side][0] - (before[side][0] + shift + [0, 0, .02])) < .003


def test_a_yaw_offset_turns_the_grip_about_the_reference(robot, project):
    clip = project['keyframes'][0]
    clip['dynamic']['offset'] = {'position': [0., 0., 0.], 'yaw_deg': 8.}
    samples, report = resolve_dynamic_clip(robot, clip, REFERENCE['position'], REFERENCE['quaternion_xyzw'])
    centre, turn = np.array(REFERENCE['position']), Rotation.from_euler('z', 8, degrees=True)
    before, after = hands(robot, np.asarray(clip['samples'])[30])['left'][0], hands(robot, samples[30])['left'][0]
    assert report['status'] in ('ok', 'approx') and np.linalg.norm(after - (centre + turn.apply(before - centre))) < .01


def test_capture_reads_the_offset_off_an_edited_pose(robot, project):
    from motioncreator.dynamic_clip import capture_tcp_offset
    project['keyframes'][0]['dynamic'].update(object_id=None)
    project['scene_objects'] = []
    clip = project['keyframes'][0]
    sample = np.asarray(clip['samples'][30])
    target = robot.point(robot.data(sample), 'left_hand')[0] + [.02, .01, -.015]
    edited, info = robot.solve(sample, sample, pins=('left_foot', 'right_foot', 'pelvis'), angle_pins=['pelvis'], mode='free', resistance=0.,
                               selected_targets={'left_hand': target}, posture_reference=sample, posture_weight=.3, max_nfev=200)
    captured = capture_tcp_offset(robot, project, 0, 30, edited)
    assert captured['hand'] == 'left' and captured['moved_mm'] == pytest.approx(np.linalg.norm([20, 10, -15]), abs=3)
    np.testing.assert_allclose(captured['offset']['position'], [.02, .01, -.015], atol=.003)
    with pytest.raises(ValueError):
        capture_tcp_offset(robot, project, 0, 5, edited)         # outside the hold: the weight would hide the edit
    clip['dynamic']['offset'] = captured['offset']
    resolved, reports = resolve_project_dynamics(robot, project)
    assert reports[0]['status'] == 'ok'
    assert np.linalg.norm(robot.point(robot.data(np.asarray(resolved['keyframes'][0]['samples'][30])), 'left_hand')[0] - target) < .004
