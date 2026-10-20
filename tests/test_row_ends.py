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


def test_objects_that_are_not_lying_rows_still_have_two_ends():
    _, axis, length = row_of(box([0, 0, 0], size=(.08, .12, .5)))             # upright: its longest horizontal side
    assert np.allclose(axis, [0, 1, 0]) and np.isclose(length, .12)
    _, axis, length = row_of(box([0, 0, 0], size=(.1, .1, .1), quaternion=Rotation.from_euler('z', 30, degrees=True).as_quat()))
    assert np.isclose(length, .1) and np.isclose(np.degrees(np.arctan2(axis[1], axis[0])), 120)   # a cube: the side closest to world y
    upright = {**cylinder([0, 0, 0], .3, diameter=.07), 'quaternion_xyzw': [0, 0, 0, 1]}
    for item in (upright, {'shape': 'sphere', 'size': [.09, .09, .09], 'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}):
        _, axis, length = row_of(item)                                         # across the robot, as wide as the diameter
        assert np.allclose(axis, [0, 1, 0]) and np.isclose(length, item['size'][0])


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


def test_a_card_saved_without_ends_takes_the_tracked_object_as_its_reference(robot):
    project = json.loads(EXAMPLE.read_text())
    cube = project['scene_objects'][0]
    target = project['keyframes'][2]['motion']['target']
    assert 'ends' not in target and target['object_id'] == cube['id']
    explicit = copy.deepcopy(project)
    explicit['keyframes'][2]['motion']['target']['ends'] = {'reference': {'shape': cube['shape'], 'size': cube['size'], **target['reference_pose']}}
    moved = np.add(cube['position'], [.03, .02, 0]).tolist()
    for item in (project, explicit):
        item['scene_objects'][0]['position'] = moved
    (old, old_reports), (new, _) = resolve_project_motions(robot, project), resolve_project_motions(robot, explicit)
    assert old_reports[2]['status'] in ('ok', 'approx') and np.allclose(old['keyframes'][2]['qpos'], new['keyframes'][2]['qpos'])
    spec = copy.deepcopy(project['keyframes'][2]['motion'])
    spec['target']['ends'] = {'reference': {**box([0, 0, 1]), 'shape': 'cone'}}
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
    errors = far_reports[2]['hand_errors_mm']                          # reached - goal per TCP, world x y z in mm
    assert set(errors) == {'left_hand', 'right_hand'} and all(len(v) == 3 for v in errors.values())
    assert max(np.linalg.norm(v) for v in errors.values()) <= far_reports[2]['max_position_error_mm'] + 1e-6
    assert 3 < bend <= 30                                              # leaned forward, within the joint limit


def test_a_waist_bend_from_a_pose_already_leaning_forward_stops_at_the_joint_limit(robot):
    from motioncreator.card_motion import _solve_once, ROW_RELAXATIONS
    project = json.loads(EXAMPLE.read_text())
    end = np.array(project['keyframes'][2]['qpos'], dtype=float)
    i = robot.names.index('waist_pitch_joint')
    end[7 + i] = robot.upper[3 + i] - np.radians(10)                  # 10° from the stop: a 29° bend would pass it
    poses = _hand_poses(robot, end)
    goals = {hand: (poses[hand][0] + [.05, 0, 0], Rotation.from_matrix(poses[hand][1])) for hand in ('left_hand', 'right_hand')}
    solved, _, _ = _solve_once(robot, end, {'waist': 'lock'}, goals, ROW_RELAXATIONS, 29.)
    assert solved[7 + i] <= robot.upper[3 + i]


def test_an_angle_tolerance_turns_the_hands_only_as_far_as_needed_instead_of_bending_the_waist(robot):
    project = json.loads(EXAMPLE.read_text())
    cube = project['scene_objects'][0]
    recorded = waist_pitch_deg(robot, project['keyframes'][2]['qpos'])
    row = cylinder(np.add(cube['position'], [.1, 0, 0]), .5)
    exact, exact_reports = resolve_project_motions(robot, with_ends(copy.deepcopy(project), row), advise=False)
    tolerant = with_ends(copy.deepcopy(project), row)
    tolerant['keyframes'][2]['motion']['angle_tolerance_deg'] = 30
    turned, reports = resolve_project_motions(robot, tolerant, advise=False)
    assert waist_pitch_deg(robot, exact['keyframes'][2]['qpos']) - recorded > 5 and 'hand_turn_deg' not in exact_reports[2]
    assert abs(waist_pitch_deg(robot, turned['keyframes'][2]['qpos']) - recorded) < .5        # no bend
    assert reports[2]['status'] in ('ok', 'approx') and 5 < reports[2]['hand_turn_deg'] < 25   # turned, but not to the full 30°
    spec = copy.deepcopy(tolerant['keyframes'][2]['motion'])
    spec['angle_tolerance_deg'] = 60
    with pytest.raises(ValueError):
        validate_motion(spec)


def test_camera_aim_points_each_hand_along_the_line_from_between_the_shoulders_at_elbow_height_to_its_grasp_point(robot):
    project = json.loads(EXAMPLE.read_text())
    cube = project['scene_objects'][0]
    aimed = with_ends(copy.deepcopy(project), cylinder(np.add(cube['position'], [.02, 0, 0]), .5))
    aimed['keyframes'][2]['motion']['camera_aim'] = True
    resolved, reports = resolve_project_motions(robot, aimed, advise=False)
    assert reports[2]['status'] in ('ok', 'approx')
    recorded_end, end = np.asarray(project['keyframes'][2]['qpos']), np.asarray(resolved['keyframes'][2]['qpos'])
    before, after = robot.data(recorded_end), robot.data(end)
    camera = np.mean([before.xpos[robot.model.body(f'{side}_shoulder_pitch_link').id] for side in ('left', 'right')], axis=0)
    camera[2] = np.mean([before.xpos[robot.model.body(f'{side}_elbow_link').id][2] for side in ('left', 'right')])
    for hand in ('left_hand', 'right_hand'):
        position, matrix = robot.point(after, hand)
        pointing = position - after.xpos[robot.orientation_ids[hand]]
        sight = position - camera
        assert np.degrees(np.arccos(pointing @ sight / np.linalg.norm(pointing) / np.linalg.norm(sight))) < 6
        old_position, old_matrix = robot.point(before, hand)
        local = old_matrix.T @ (old_position - before.xpos[robot.orientation_ids[hand]])
        side = np.cross(local, [0, 0, 1.])                                    # a hand axis across the pointing one
        swing = Rotation.align_vectors([matrix @ local], [old_matrix @ local])[0]
        assert np.degrees(np.arccos(np.clip((matrix @ side) @ swing.apply(old_matrix @ side) / np.linalg.norm(side) ** 2, -1, 1))) < 6
    point = copy.deepcopy(project['keyframes'][5]['motion'])
    assert 'camera_aim' not in validate_motion({**point, 'camera_aim': True})     # a point target ignores it


def test_a_position_delta_card_whose_carried_offset_passes_a_joint_limit_still_solves(robot):
    from motioncreator.card_motion import COLUMNS, _position_delta
    project = json.loads(EXAMPLE.read_text())
    previous, end = np.asarray(project['keyframes'][2]['qpos'], dtype=float), np.asarray(project['keyframes'][3]['qpos'], dtype=float)
    offset = np.zeros(COLUMNS.stop - COLUMNS.start)
    offset[robot.names.index('left_elbow_joint') - 12] = 4.          # far past the elbow's range
    resolved_previous = previous.copy()
    solved, _, _ = _position_delta(robot, previous, resolved_previous, end, offset)
    assert np.all(solved[7:] >= robot.lower[3:] - 1e-6) and np.all(solved[7:] <= robot.upper[3:] + 1e-6)
