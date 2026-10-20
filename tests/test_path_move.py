import copy
import json
from pathlib import Path

import numpy as np
import pytest

from motioncreator.grasp_template import _hand_poses
from motioncreator.motion import compile_motion, new_project, validate_project
from motioncreator.path_move import Arc, validate_path

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'pick_cards_right_mirrored.project.json'


@pytest.fixture(scope='module')
def robot():
    from motioncreator import server
    return server.robot


@pytest.fixture(scope='module')
def cards():
    return json.loads(EXAMPLE.read_text())['keyframes']


def two_poses(robot, qa, qb, path=None, pins=('left_foot', 'right_foot')):
    project = new_project(robot, 'path')
    first = {'name': 'a', 'duration': .1, 'qpos': list(qa), 'pins': list(pins), 'angle_pins': []}
    second = {'name': 'b', 'duration': 2., 'qpos': list(qb), 'pins': list(pins), 'angle_pins': []}
    if path:
        second['path'] = path
    project['keyframes'] = [first, second]
    return project


def hand(robot, q, side='left_hand'):
    return _hand_poses(robot, np.asarray(q))[side][0]


def test_an_arc_goes_through_its_via_point_and_a_collinear_one_is_a_line():
    start, via, end = np.array([0., 0., 0.]), np.array([.1, .05, 0.]), np.array([.2, 0., 0.])
    arc = Arc(start, via, end)
    assert np.allclose(arc.point(0.), start) and np.allclose(arc.point(1.), end)
    assert min(np.linalg.norm(arc.point(s) - via) for s in np.linspace(0, 1, 2001)) < 1e-3
    radii = [np.linalg.norm(arc.point(s) - arc.center) for s in np.linspace(0, 1, 11)]
    assert max(radii) - min(radii) < 1e-9
    line = Arc(start, (start + end) / 2, end)
    assert line.straight and np.allclose(line.point(.25), [.05, 0., 0.])


def test_an_arc_longer_than_a_half_circle_still_runs_through_the_via_point():
    start, via, end = np.array([1., 0., 0.]), np.array([-1., 0., 0.]), np.array([0., -1., 0.])      # via first, then end: 270 degrees
    arc = Arc(start, via, end)
    assert min(np.linalg.norm(arc.point(s) - via) for s in np.linspace(0, 1, 4001)) < 1e-3
    assert abs(arc.sweep) > np.pi


@pytest.mark.parametrize('spec', [{'kind': 'spline'}, {'kind': 'circular', 'via_offset': [0, 0]}, {'kind': 'circular', 'via_offset': [2, 0, 0]}])
def test_invalid_paths_are_rejected(spec):
    with pytest.raises(ValueError):
        validate_path(spec)


def test_linear_moves_the_hands_on_straight_lines_where_the_joint_blend_curves(robot, cards):
    qa, qb = cards[7]['samples'][0], cards[7]['samples'][50]

    def deviation(path):
        motion = compile_motion(robot, two_poses(robot, qa, qb, path), 30)
        worst = 0.
        for side in ('left_hand', 'right_hand'):
            points = np.array([hand(robot, q, side) for q in motion['qpos']])
            line = (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0])
            worst = max(worst, max(np.linalg.norm((p - points[0]) - ((p - points[0]) @ line) * line) for p in points))
        return worst, motion
    straight, motion = deviation({'kind': 'linear'})
    assert straight < .002
    assert np.allclose(motion['qpos'][0], qa, atol=1e-9) and np.allclose(motion['qpos'][-1], qb, atol=1e-9)
    assert abs(motion['time'][-1] - 2.) < 1e-9


def test_circular_goes_through_the_offset_via_point_on_both_hands(robot, cards):
    qa, qb, offset = cards[1]['qpos'], cards[2]['qpos'], np.array([.02, 0., .07])
    motion = compile_motion(robot, two_poses(robot, qa, qb, {'kind': 'circular', 'via_offset': offset.tolist()}), 30)
    for side in ('left_hand', 'right_hand'):
        points = np.array([hand(robot, q, side) for q in motion['qpos']])
        assert np.linalg.norm(points - ((points[0] + points[-1]) / 2 + offset), axis=1).min() < .004


def test_the_feet_stay_where_they_were_during_a_move(robot, cards):
    qa, qb = cards[7]['samples'][0], cards[7]['samples'][50]
    motion = compile_motion(robot, two_poses(robot, qa, qb, {'kind': 'linear'}), 30)
    feet = np.array([[np.asarray(robot.point(robot.data(q), side)[0]) for side in ('left_foot', 'right_foot')] for q in motion['qpos'][::10]])
    assert np.abs(feet - feet[0]).max() < .004


def test_joint_is_the_default_and_clips_take_no_path(robot, cards):
    qa, qb = cards[1]['qpos'], cards[2]['qpos']
    plain = compile_motion(robot, two_poses(robot, qa, qb), 30)['qpos']
    joint = compile_motion(robot, two_poses(robot, qa, qb, {'kind': 'joint'}), 30)['qpos']
    assert np.allclose(plain, joint)
    project = json.loads(EXAMPLE.read_text())
    project['keyframes'][3]['path'] = {'kind': 'linear'}                      # card 4 is a clip
    with pytest.raises(ValueError):
        validate_project(robot, project)


def test_a_path_survives_the_card_motion_modes(robot):
    project = json.loads(EXAMPLE.read_text())
    project['keyframes'][2]['path'] = {'kind': 'linear'}                      # approach: a straight hand move
    moved = copy.deepcopy(project)
    moved['scene_objects'][0]['position'][0] += .03
    motion = compile_motion(robot, moved, 30)
    assert abs(motion['time'][-1] - compile_motion(robot, project, 30)['time'][-1]) < 1e-9


def test_a_smooth_connection_that_overshoots_a_joint_limit_still_compiles_with_a_straight_move(robot):
    """The Hermite blend can leave the joint range next to a keyframe at the limit; that must not reject the whole motion."""
    joint = robot.model.joint('left_elbow_joint').qposadr[0]
    index = list(robot.q_indices).index(joint)
    base = robot.home.copy()
    low, high = base.copy(), base.copy()
    low[joint] = robot.upper[index] - .6
    high[joint] = robot.upper[index]
    project = new_project(robot, 'overshoot')
    pins = ['left_foot', 'right_foot']
    cards = [{'name': name, 'duration': 1., 'qpos': list(q), 'pins': pins, 'angle_pins': []} for name, q in (('a', low), ('b', high), ('c', high))]
    cards[1]['generated_transition'] = True
    cards[1]['path'] = {'kind': 'linear'}
    project['keyframes'] = cards
    compiled = compile_motion(robot, project, 30)
    assert np.all(np.asarray(compiled['qpos'])[:, joint] <= robot.upper[index] + 1e-7)
