import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motioncreator.card_motion import resolve_project_motions, validate_motion
from motioncreator.grasp_template import _hand_poses
from motioncreator.row_ends import row_goals, row_of

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'pick_cards_modes.project.json'


def box(position, size=(.08, .5, .08), quaternion=(0, 0, 0, 1)):
    return {'shape': 'box', 'size': list(size), 'position': list(position), 'quaternion_xyzw': list(quaternion)}


def cylinder(position, length, yaw_deg=90., diameter=.13):
    """Cylinder lying horizontally; yaw 90 puts its axis along world y."""
    turn = Rotation.from_euler('z', yaw_deg - 90, degrees=True) * Rotation.from_euler('x', -90, degrees=True)
    return {'shape': 'cylinder', 'size': [diameter, diameter, length], 'position': list(position), 'quaternion_xyzw': turn.as_quat().tolist()}


def test_the_long_axis_of_a_box_and_a_cylinder_point_toward_the_robot_left():
    center, axis, length = row_of(box([.3, 0, .9]))
    assert np.allclose(axis, [0, 1, 0]) and length == .5
    _, axis, length = row_of(cylinder([.3, 0, .9], .6, yaw_deg=-80))          # drawn pointing right, reported pointing left
    assert axis[1] > 0 and np.allclose(axis, [np.cos(np.radians(100)), np.sin(np.radians(100)), 0], atol=1e-9) and length == .6
    assert row_of({**box([0, 0, 0]), 'shape': 'sphere'}) is None
    assert row_of(box([0, 0, 0], size=(.08, .08, .5))) is None                # upright


def test_each_hand_keeps_its_distance_from_its_own_end_when_the_row_is_longer_and_turned():
    hands = {'left_hand': (np.array([.3, .27, .9]), np.eye(3)), 'right_hand': (np.array([.3, -.27, .9]), np.eye(3))}
    goals, relations = row_goals(row_of(box([.3, 0, .9])), row_of(cylinder([.5, .1, 1.0], .7, yaw_deg=60)), hands)
    assert relations['left_hand'][0] == 1 and np.isclose(relations['left_hand'][1][0], .02)       # 2 cm outside the +y end
    axis = np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0.])
    center = np.array([.5, .1, 1.0])
    for hand, sign in (('left_hand', 1), ('right_hand', -1)):
        assert np.allclose(goals[hand][0], center + sign * axis * (.35 + .02))
    assert np.isclose(np.degrees(goals['left_hand'][1].as_euler('zyx')[0]), -30)               # turned by heading difference


@pytest.fixture(scope='module')
def robot():
    from motioncreator import server
    return server.robot


def with_ends(project, row):
    cube = project['scene_objects'][0]
    reference = {key: cube[key] for key in ('shape', 'size', 'position', 'quaternion_xyzw')}
    project['keyframes'][2]['motion']['target']['ends'] = {'reference': reference}
    project['keyframes'][2]['motion']['target']['object_id'] = 'row'
    project['scene_objects'].append({**cube, 'id': 'row', 'name': 'row', **row})
    return project


def hands(robot, q):
    poses = _hand_poses(robot, np.asarray(q))
    return poses['left_hand'][0], poses['right_hand'][0]


def test_the_approach_card_puts_the_hands_at_the_ends_of_a_longer_shifted_row(robot):
    project = json.loads(EXAMPLE.read_text())
    cube = project['scene_objects'][0]
    row = cylinder(np.add(cube['position'], [.02, 0, 0]), .62)
    resolved, reports = resolve_project_motions(robot, with_ends(copy.deepcopy(project), row))
    assert reports[2]['status'] in ('ok', 'approx', 'partial') and reports[2]['max_position_error_mm'] < 25
    (left_before, right_before), (left, right) = hands(robot, project['keyframes'][2]['qpos']), hands(robot, resolved['keyframes'][2]['qpos'])
    assert np.isclose(np.linalg.norm(left - right) - np.linalg.norm(left_before - right_before), .12, atol=.03)
    assert 0 < (left - left_before)[1] and (right - right_before)[1] < 0             # the hands moved apart, outward


def test_a_row_that_is_not_lying_is_reported_and_a_bad_reference_is_rejected(robot):
    project = json.loads(EXAMPLE.read_text())
    upright = box(project['scene_objects'][0]['position'], size=(.08, .08, .5))
    _, reports = resolve_project_motions(robot, with_ends(project, upright))
    assert reports[2]['status'] == 'not_a_row'
    spec = copy.deepcopy(project['keyframes'][2]['motion'])
    spec['target']['ends'] = {'reference': {**box([0, 0, 1]), 'shape': 'sphere'}}
    with pytest.raises(ValueError):
        validate_motion(spec)


def test_tcp_placement_ignores_the_recorded_distance_and_uses_inward_lateral_and_up_from_each_end():
    hands = {'left_hand': (np.array([.3, .27, .9]), np.eye(3)), 'right_hand': (np.array([.3, -.27, .9]), np.eye(3))}
    row = row_of(cylinder([.5, .1, 1.0], .7, yaw_deg=60))
    goals, _ = row_goals(row_of(box([.3, 0, .9])), row, hands, tcp=(.03, .01, .04))
    axis, center = row[1], row[0]
    lateral = np.cross([0, 0, 1.], axis)
    for hand, sign in (('left_hand', 1), ('right_hand', -1)):
        expected = center + sign * axis * (.35 - .03) + lateral * .01 + [0, 0, .04]
        assert np.allclose(goals[hand][0], expected, atol=1e-9)
    with pytest.raises(ValueError):
        validate_ends_tcp = copy.deepcopy(json.loads(EXAMPLE.read_text())['keyframes'][2]['motion'])
        validate_ends_tcp['target']['ends'] = {'reference': box([0, 0, 1]), 'tcp': {'inward_m': 2, 'lateral_m': 0, 'up_m': 0}}
        validate_motion(validate_ends_tcp)


def waist_pitch_deg(robot, q):
    return float(np.degrees(q[7 + robot.names.index('waist_pitch_joint')]))


def test_the_waist_stays_put_when_the_arms_reach_and_bends_forward_only_when_they_do_not(robot):
    project = json.loads(EXAMPLE.read_text())
    cube = project['scene_objects'][0]
    recorded = waist_pitch_deg(robot, project['keyframes'][2]['qpos'])
    near, near_reports = resolve_project_motions(robot, with_ends(copy.deepcopy(project), cylinder(np.add(cube['position'], [.03, 0, 0]), .5)))
    assert near_reports[2]['status'] in ('ok', 'approx') and abs(waist_pitch_deg(robot, near['keyframes'][2]['qpos']) - recorded) < .5
    far, far_reports = resolve_project_motions(robot, with_ends(copy.deepcopy(project), cylinder(np.add(cube['position'], [.2, 0, 0]), .5)))
    bend = waist_pitch_deg(robot, far['keyframes'][2]['qpos']) - recorded
    assert far_reports[2]['status'] in ('ok', 'approx') and far_reports[2]['max_position_error_mm'] < 20
    assert 3 < bend <= 30                                              # leaned forward, within the joint limit
