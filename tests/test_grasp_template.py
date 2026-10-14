import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motioncreator.dynamic_clip import ARM_WAIST_COLUMNS
from motioncreator.grasp_template import (HANDS, _hand_poses, extract_template, generate_motion, load_template,
                                          mirrored, symmetrized)
from motioncreator.motion import compile_motion
from motioncreator.robot import Robot


@pytest.fixture(scope='module')
def robot():
    return Robot()


@pytest.fixture(scope='module')
def template():
    return load_template()


@pytest.fixture(scope='module')
def centre(template):
    return np.array(template['contact']['mid'])


def test_mirror_is_an_involution_and_symmetrizing_makes_a_symmetric_pose(robot):
    q = robot.home.copy()
    q[7 + 15:7 + 29] = np.linspace(-.4, .5, 14)
    assert np.allclose(mirrored(robot, mirrored(robot, q))[7:], q[7:])
    sym = symmetrized(robot, q)
    assert np.allclose(mirrored(robot, sym)[7:], sym[7:])
    left, right = _hand_poses(robot, sym)['left_hand'][0], _hand_poses(robot, sym)['right_hand'][0]
    assert np.allclose(left * [1, -1, 1], right, atol=1e-4)      # the model itself is symmetric to ~10 µm


def test_the_object_at_the_demonstrated_place_reproduces_the_demonstration(robot, template, centre):
    clips, reports = generate_motion(robot, template, centre)
    assert [clip['name'] for clip in clips] == [phase['name'] for phase in template['phases']]
    assert {report['method'] for report in reports} == {'delta', 'cartesian'}
    assert all(report['status'] in ('ok', 'approx') for report in reports)
    for clip, phase in zip(clips[:4], template['phases'][:4]):          # up to and including the grasp
        a, b = _hand_poses(robot, np.array(phase['end'])), _hand_poses(robot, np.array(clip['samples'][-1]))
        assert max(np.linalg.norm(a[h][0] - b[h][0]) for h in HANDS) < 2e-3
    assert np.allclose(clips[-1]['samples'][-1][ARM_WAIST_COLUMNS], np.array(template['phases'][-1]['end'])[ARM_WAIST_COLUMNS], atol=1e-6)


def test_every_clip_is_left_right_symmetric(robot, template, centre):
    clips, _ = generate_motion(robot, template, centre)
    for clip in clips:
        samples = np.array(clip['samples'])
        asymmetry = max(np.abs(q[7:] - mirrored(robot, q)[7:]).max() for q in samples[::5])
        assert np.degrees(asymmetry) < 4


def test_clips_chain_without_jumps_and_stay_inside_the_limits(robot, template, centre):
    clips, _ = generate_motion(robot, template, centre)
    for before, after in zip(clips, clips[1:]):
        assert np.allclose(before['samples'][-1], after['samples'][0], atol=1e-9)
    for clip in clips:
        for sample in clip['samples'][::10]:
            robot.validate_q(sample)


@pytest.mark.parametrize('shift', [(.0, .0, .04), (-.05, 0., 0.), (.05, 0., 0.), (0., .04, 0.)])
def test_nearby_objects_are_reached_by_the_approach(robot, template, centre, shift):
    clips, reports = generate_motion(robot, template, centre + shift)
    assert reports[2]['final_position_error_mm'] < 15
    reached = _hand_poses(robot, np.array(clips[2]['samples'][-1]))
    middle = (reached['left_hand'][0] + reached['right_hand'][0]) / 2
    assert np.linalg.norm(middle - (centre + shift)) < .02


def test_out_of_reach_objects_are_reported_not_raised(robot, template, centre):
    _, reports = generate_motion(robot, template, centre + [.2, 0., 0.])
    assert reports[2]['status'] == 'unreachable'


def test_a_joint_phase_that_would_break_a_limit_falls_back_to_the_hands(robot, template, centre):
    _, reports = generate_motion(robot, template, centre + [0., 0., .08])
    grasp = reports[3]
    assert grasp['method'] == 'cartesian' and grasp['fallback'] == 'joint limit' and grasp['status'] != 'unreachable'


def test_carry_goes_to_a_coordinate_not_an_offset(robot, template, centre):
    target = np.array([.34, 0., 1.0])
    clips, reports = generate_motion(robot, template, centre, carry_target=target)
    reached = _hand_poses(robot, np.array(clips[4]['samples'][-1]))
    assert np.linalg.norm((reached['left_hand'][0] + reached['right_hand'][0]) / 2 - target) < .015
    far = generate_motion(robot, template, centre + [.04, 0., 0.])[0]
    default = np.array(template['carry']['target'])
    got = _hand_poses(robot, np.array(far[4]['samples'][-1]))
    assert np.linalg.norm((got['left_hand'][0] + got['right_hand'][0]) / 2 - default) < .03


def test_extraction_round_trips_a_generated_motion(robot, template, centre):
    clips, _ = generate_motion(robot, template, centre)
    again = extract_template(robot, clips)
    assert np.allclose(again['contact']['mid'], template['contact']['mid'], atol=3e-3)
    assert max(again['recording_asymmetry_deg_rms']) < 5       # nothing left to symmetrize


def test_generated_clips_compile_as_a_project(robot, template, centre):
    from motioncreator.motion import new_project
    clips, _ = generate_motion(robot, template, centre)
    project = new_project(robot, 'generated')
    project['keyframes'] = clips
    motion = compile_motion(robot, project, 30)
    assert len(motion['qpos']) > 400


def test_endpoint_generates_the_clips(template, centre):
    from fastapi.testclient import TestClient
    from motioncreator.server import app
    response = TestClient(app).post('/api/grasp-template/generate', json={'object_position': centre.tolist()})
    assert response.status_code == 200
    body = response.json()
    assert len(body['keyframes']) == 7 and len(body['reports']) == 7
