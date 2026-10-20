import json
import numpy as np
import pytest
from motioncreator.motion import project_from_motion_bytes
from motioncreator.robot import Robot


def episode(frames=3, body=()):
    arm = lambda v: {'qpos': [v] * 7, 'qvel': [], 'torque': []}
    frame = lambda i: {'actions': {'left_arm': arm(0.1 * i), 'right_arm': arm(-0.1 * i), 'body': {'qpos': list(body)}},
                       'states': {'left_arm': arm(0.0), 'right_arm': arm(0.0), 'body': {'qpos': []}}}
    return json.dumps({'info': {'image': {'fps': 30.0}}, 'text': {'goal': 'g'},
                       'data': [frame(i) for i in range(frames)]}).encode()


@pytest.fixture(scope='module')
def robot():
    return Robot()


def test_arms_map_to_unitree_order(robot):
    project = project_from_motion_bytes(robot, episode(), 'x.json', 30)
    samples = np.array(project['keyframes'][0]['samples'])
    assert samples.shape == (3, robot.model.nq)
    assert np.allclose(samples[2, 7 + 15:7 + 22], 0.2) and np.allclose(samples[2, 7 + 22:7 + 29], -0.2)
    assert np.allclose(samples[:, 7:7 + 12], robot.home[7:7 + 12])


def test_waist_and_source(robot):
    samples = np.array(project_from_motion_bytes(robot, episode(body=(0.1, 0.0, 0.0)), 'x.json', 30)['keyframes'][0]['samples'])
    assert np.isclose(samples[0, 7 + 12], 0.1)
    states = np.array(project_from_motion_bytes(robot, episode(), 'x.json', 30, 'states')['keyframes'][0]['samples'])
    assert np.allclose(states[2, 7 + 15:7 + 22], 0.0)


def test_rejects_bad_arm_length(robot):
    bad = json.loads(episode()); bad['data'][0]['actions']['left_arm']['qpos'] = [0.0]
    with pytest.raises(ValueError):
        project_from_motion_bytes(robot, json.dumps(bad).encode(), 'x.json', 30)


def test_robot_studio_sample(robot):
    from pathlib import Path
    path = Path.home() / 'robot-studio/episodes/pick cube/episode_0005/data.json'
    if not path.is_file():
        pytest.skip('robot-studio sample missing')
    project = project_from_motion_bytes(robot, path.read_bytes(), 'data.json', 30)
    assert len(project['keyframes'][0]['samples']) == 564


def split_project(robot, at):
    project = project_from_motion_bytes(robot, episode(frames=11), 'x.json', 30)
    clip = project['keyframes'][0]
    samples, duration = clip['samples'], clip['duration']
    halves = [samples[:at + 1], samples[at:]]
    times = [duration * at / 10, duration * (10 - at) / 10]
    project['keyframes'] = [{**clip, 'name': f'p{i}', 'duration': times[i], 'qpos': half[0], 'samples': half}
                            for i, half in enumerate(halves)]
    return project


def test_split_clips_play_like_the_original(robot):
    from motioncreator.motion import compile_motion, validate_project
    whole = project_from_motion_bytes(robot, episode(frames=11), 'x.json', 30)
    split = split_project(robot, 4)
    validate_project(robot, split)
    a, b = compile_motion(robot, whole, 30), compile_motion(robot, split, 30)
    assert abs(a['time'][-1] - b['time'][-1]) < 1 / 30 + 1e-9
    assert np.allclose(a['qpos'][-1], b['qpos'][-1])
    assert np.all(np.diff(b['time']) > 0)


def test_clips_mix_with_pose_keyframes_but_stay_unpinned(robot):
    from motioncreator.motion import new_project, validate_project
    split = split_project(robot, 4)
    split['keyframes'].append(new_project(robot)['keyframes'][0])
    validate_project(robot, split)
    split['keyframes'][0]['pins'] = ['left_foot']
    with pytest.raises(ValueError):
        validate_project(robot, split)


def clip_pose_project(robot):
    """clip A | pose (edited) | clip B"""
    from motioncreator.motion import FEET
    project = split_project(robot, 4)
    a, b = project['keyframes']
    edited = {'name': 'edit', 'duration': 1., 'qpos': list(b['qpos']), 'pins': list(FEET), 'angle_pins': []}
    edited['qpos'][7 + 7 + 15] += 0.4   # left shoulder pitch (qpos index 7 + joint 15)
    project['keyframes'] = [a, edited, b]
    return project


def test_clip_pose_clip_blends_into_the_next_clip(robot):
    from motioncreator.motion import compile_motion, validate_project
    project = clip_pose_project(robot)
    validate_project(robot, project)
    motion = compile_motion(robot, project, 30)
    times, q = np.asarray(motion['time']), np.asarray(motion['qpos'])
    assert np.all(np.diff(times) > 0)
    assert np.allclose(q[-1], project['keyframes'][2]['samples'][-1])
    assert len(motion['keyframe_times']) == 3 and motion['keyframe_times'] == sorted(motion['keyframe_times'])
    # the edited pose is actually visited on the way
    assert np.abs(q[:, 7 + 15 + 7] - project['keyframes'][2]['qpos'][7 + 15 + 7]).max() > 0.3


def test_pose_then_clip_uses_the_clip_transition_time(robot):
    from motioncreator.motion import compile_motion
    project = clip_pose_project(robot)
    short = compile_motion(robot, {**project, 'keyframes': [*project['keyframes'][:2], {**project['keyframes'][2], 'transition': 0.5}]}, 30)
    long = compile_motion(robot, {**project, 'keyframes': [*project['keyframes'][:2], {**project['keyframes'][2], 'transition': 2.0}]}, 30)
    assert abs((long['time'][-1] - short['time'][-1]) - 1.5) < 2 / 30
