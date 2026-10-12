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
