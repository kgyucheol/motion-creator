import json

import numpy as np
import pytest

from motioncreator.motion import project_from_motion_bytes
from motioncreator.replay import config as C
from motioncreator.replay.arm_sdk_writer import Weight, clip_step, gains, limit_command, validate_settings
from motioncreator.replay.backends import CommandError, MockBackend
from motioncreator.replay.plan import ARM_COLUMNS, Gravity, build_plan, check_dynamic, check_replayable, check_trajectory
from motioncreator.replay.player import ReplayPlayer, min_jerk, move_duration
from motioncreator.replay.session import ReplaySession
from motioncreator.robot import Robot

HOME_ARMS = np.array([-.5, .2, 0., .9, 0., 0., 0., -.5, -.2, 0., .9, 0., 0., 0.])


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture(scope='module')
def robot():
    return Robot()


def reach(frames=61, amplitude=.25):
    """Both arms swing out and back from HOME_ARMS."""
    swing = amplitude * np.sin(np.pi * np.arange(frames) / (frames - 1))
    traj = np.tile(HOME_ARMS, (frames, 1))
    traj[:, [0, 7]] -= swing[:, None]
    traj[:, [3, 10]] += swing[:, None]
    return traj


def plan_checks(traj, fps=30.0):
    lower, upper = np.full(14, -3.0), np.full(14, 3.0)
    checks = check_trajectory(traj, fps, lower, upper)
    return checks


def make_player(traj=None, fps=30.0, lag=0.0):
    clock = Clock()
    backend = MockBackend(HOME_ARMS, clock=clock, threaded=False, lag=lag)
    player = ReplayPlayer(lambda: backend, backend.state, clock=clock, threaded=False)
    traj = reach() if traj is None else traj
    player.load(traj, fps, plan_checks(traj, fps))
    return player, backend, clock


def run(player, backend, clock, seconds, dt=0.02):
    for _ in range(int(round(seconds / dt))):
        clock.now += dt
        backend.step()
        player.tick()


# ---- checks ----------------------------------------------------------------------------------------------------------------
def test_trajectory_checks_limits_and_speed():
    ok = plan_checks(reach())
    assert ok['ok'] and ok['speed_cap'] >= 1.0
    far = reach()
    far[10, 0] = 5.0
    assert not plan_checks(far)['ok']
    fast = np.vstack([HOME_ARMS, HOME_ARMS + 2.0])           # 2 rad in one frame = 60 rad/s
    assert not plan_checks(fast)['ok']
    nan = reach()
    nan[3, 2] = np.nan
    assert not plan_checks(nan)['ok']
    quick = plan_checks(reach(frames=11, amplitude=.9))      # fast enough to need a slower multiplier, but not hopeless
    assert quick['ok'] and 0 < quick['speed_cap'] < 1.0 and quick['speed_hard'] > quick['speed_cap']


def test_only_arm_motion_is_replayable():
    still = np.zeros((10, 36))
    still[:, 3] = 1.0
    assert check_replayable(still)[0]['level'] == 'ok'
    moved = still.copy()
    moved[5:, 7 + 3] = .3                                      # a knee
    moved[5:, 2] = .05                                         # pelvis height
    moved[5:, 7 + 14] = .1                                     # waist pitch
    item = check_replayable(moved)[0]
    assert item['level'] == 'warn' and '다리' in item['text'] and '골반 높이' in item['text'] and '허리' in item['text']


def test_dynamic_reports_gate_the_start():
    levels = {r['status']: check_dynamic([{'index': 0, 'status': r['status']}])[0]['level']
              for r in [{'status': s} for s in ('ok', 'unchanged', 'approx', 'unreachable', 'too_far', 'missing_object')]}
    assert levels == {'ok': 'ok', 'unchanged': 'ok', 'approx': 'warn', 'unreachable': 'bad', 'too_far': 'bad', 'missing_object': 'bad'}


def test_plan_comes_from_a_compiled_motion(robot):
    frames = [{'actions': {'left_arm': {'qpos': HOME_ARMS[:7].tolist()}, 'right_arm': {'qpos': HOME_ARMS[7:].tolist()}, 'body': {'qpos': []}},
               'states': {'left_arm': {'qpos': HOME_ARMS[:7].tolist()}, 'right_arm': {'qpos': HOME_ARMS[7:].tolist()}, 'body': {'qpos': []}}}
              for _ in range(30)]
    content = json.dumps({'info': {'image': {'fps': 30.0}}, 'text': {'goal': 'g'}, 'data': frames}).encode()
    from motioncreator.motion import compile_motion
    project = project_from_motion_bytes(robot, content, 'x.json', 30)
    plan = build_plan(robot, compile_motion(robot, project, 30), 30)
    assert plan['traj'].shape[1] == 14 and plan['checks']['ok']
    np.testing.assert_allclose(plan['traj'][0], HOME_ARMS, atol=1e-6)


def test_gravity_torque_is_finite_and_follows_the_pose(robot):
    gravity = Gravity(robot)
    rest = gravity.torque(np.asarray(robot.home)[ARM_COLUMNS])
    held_out = HOME_ARMS.copy()
    held_out[[0, 7]] = -1.4                                    # arms forward: more shoulder torque
    reach_torque = gravity.torque(held_out)
    assert rest.shape == (14,) and np.isfinite(rest).all()
    assert abs(reach_torque[0]) > abs(rest[0])


# ---- the arm_sdk writer's pure parts -------------------------------------------------------------------------------------
def test_writer_helpers():
    weight = Weight()
    weight.ramp_to(1.0, 2.0, 0.0)
    assert weight.step(1.0) == pytest.approx(.5) and weight.step(3.0) == 1.0 and weight.done
    stepped = clip_step(np.zeros(2), np.array([1.0, .5]), vmax=1.0, dt=.1)
    np.testing.assert_allclose(stepped, [.1, .05])
    limited = limit_command(np.zeros(1), np.array([5.0]), np.array([0.0]), vmax=100., dt=1., lead=.25)
    assert limited[0] == .25
    assert gains(15)[0] == C.ARM_SDK_DEFAULTS['kp_arm'] and gains(19)[0] == C.ARM_SDK_DEFAULTS['kp_wrist'] and gains(4) == (80.0, 3.0)
    assert validate_settings({})['hz'] == 250
    for bad in ({'hz': 77}, {'ff': 2}, {'kp_arm': float('nan')}):
        with pytest.raises(ValueError):
            validate_settings(bad)


def test_min_jerk_and_move_duration():
    assert min_jerk(0) == 0 and min_jerk(1) == 1 and min_jerk(.5) == pytest.approx(.5)
    assert move_duration(np.array([.01]), .6) == C.MIN_MOVE_S
    assert move_duration(np.array([1.2]), .6) == pytest.approx(3.75)


# ---- the player ------------------------------------------------------------------------------------------------------------
def test_full_run_approach_play_return_release():
    player, backend, clock = make_player()
    info = player.start(1.0, auto_return=True)
    assert player.state == 'approach' and backend.engaged and info['speed'] == 1.0
    run(player, backend, clock, 1.0)
    assert player.state == 'playing'
    run(player, backend, clock, 3.0)
    assert player.state in ('playing', 'finished', 'returning', 'idle')
    run(player, backend, clock, 10.0)
    assert player.state == 'idle' and not backend.engaged and player.fault is None
    np.testing.assert_allclose(backend.q[C.ARM_SLICE], HOME_ARMS, atol=1e-6)
    sent = np.array(backend.sent)
    assert np.abs(np.diff(sent, axis=0)).max() < .05          # no jumps between commands at 50 Hz


def test_the_first_command_is_where_the_arms_are():
    player, backend, clock = make_player()
    backend.q[C.ARM_SLICE] += .3
    measured = backend.q[C.ARM_SLICE].copy()
    player.start(1.0, auto_return=False)
    run(player, backend, clock, .02)
    np.testing.assert_allclose(backend.sent[0], measured, atol=.05)


def test_stop_keeps_commanding_and_home_returns_then_releases():
    player, backend, clock = make_player(traj=reach(301))
    player.start(1.0, auto_return=False)
    run(player, backend, clock, 3.0)
    player.stop()
    assert player.state == 'stopped' and backend.engaged
    before = len(backend.sent)
    run(player, backend, clock, .5)
    assert len(backend.sent) > before and backend.engaged
    player.go_home()
    run(player, backend, clock, 8.0)
    assert player.state == 'idle' and not backend.engaged


def test_pause_freezes_the_position():
    player, backend, clock = make_player(traj=reach(301))
    player.start(1.0, auto_return=False)
    run(player, backend, clock, 2.0)
    player.pause(True)
    frame = player.pos
    run(player, backend, clock, 1.0)
    assert player.pos == frame
    player.pause(False)
    run(player, backend, clock, 1.0)
    assert player.pos > frame


def test_stale_state_releases_control():
    player, backend, clock = make_player()
    player.start(1.0, auto_return=False)
    run(player, backend, clock, 1.0)
    stale = backend.state()
    stale.stamp = clock.now - 2.0
    player.get_state = lambda: stale
    player.tick()
    assert player.state == 'idle' and not backend.engaged and '수신' in player.fault


def test_a_tracking_fault_holds_instead_of_pushing_on():
    player, backend, clock = make_player(traj=reach(61, .6), lag=1e9)   # arms that never move
    player.start(1.0, auto_return=False)
    run(player, backend, clock, 8.0)
    assert player.state == 'stopped' and player.fault and 'rad' in player.fault and backend.engaged


def test_refusals():
    player, backend, clock = make_player(traj=np.vstack([HOME_ARMS, HOME_ARMS + 2.0]))
    with pytest.raises(CommandError):
        player.start(1.0, auto_return=True)                    # failed the preflight
    player, backend, clock = make_player()
    player.get_state = lambda: None
    with pytest.raises(CommandError):
        player.start(1.0, auto_return=True)                    # no robot state
    player, backend, clock = make_player()
    player.start(1.0, auto_return=True)
    with pytest.raises(CommandError):
        player.start(1.0, auto_return=True)                    # already running


def test_speed_is_limited_to_what_the_checks_allow():
    quick = reach(frames=11, amplitude=.9)
    player, backend, clock = make_player(traj=quick)
    cap = player.checks['speed_cap']
    info = player.start(5.0, auto_return=False)
    assert info['speed'] == pytest.approx(cap) and info['capped']


# ---- the session -----------------------------------------------------------------------------------------------------------
def test_session_needs_a_connection_and_a_confirmation(robot):
    session = ReplaySession(lambda: robot)
    with pytest.raises(CommandError):
        session.start(1.0, True, False, confirm=True)
    session.connect(None, mock=True)
    with pytest.raises(CommandError):
        session.start(1.0, True, False, confirm=False)
    status = session.status()
    assert status['mode'] == 'mock' and status['robot']['tilt_deg'] == 0 and status['plan'] is None
    session.shutdown()


def test_read_only_never_connects_to_a_real_robot(robot):
    session = ReplaySession(lambda: robot, read_only=True)
    with pytest.raises(CommandError):
        session.connect('eth0', mock=False)
    session.connect(None, mock=True)                           # the simulated robot is always allowed
    assert session.status()['connected']
    session.shutdown()


def test_session_runs_a_prepared_project_on_the_simulated_robot(robot):
    import time
    session = ReplaySession(lambda: robot)
    frames = 31
    traj = reach(frames, .2)
    arm = lambda q: {'qpos': q.tolist()}
    data = [{'actions': {'left_arm': arm(q[:7]), 'right_arm': arm(q[7:]), 'body': {'qpos': []}},
             'states': {'left_arm': arm(q[:7]), 'right_arm': arm(q[7:]), 'body': {'qpos': []}}} for q in traj]
    content = json.dumps({'info': {'image': {'fps': 30.0}}, 'text': {'goal': 'g'}, 'data': data}).encode()
    project = project_from_motion_bytes(robot, content, 'x.json', 30)
    session.connect(None, mock=True)
    summary = session.prepare(project, 30)
    assert summary['ok'] and summary['frames'] >= frames
    session.start(4.0, auto_return=True, allow_fast=False, confirm=True)
    deadline = time.monotonic() + 30
    while session.player.state != 'idle' and time.monotonic() < deadline:
        time.sleep(0.1)
    assert session.player.state == 'idle' and session.player.fault is None
    session.shutdown()


# ---- server endpoints ------------------------------------------------------------------------------------------------------
def episode_bytes(traj, fps=30.0):
    arm = lambda q: {'qpos': q.tolist(), 'qvel': [], 'torque': []}
    data = [{'actions': {'left_arm': arm(q[:7]), 'right_arm': arm(q[7:]), 'body': {'qpos': []}},
             'states': {'left_arm': arm(q[:7]), 'right_arm': arm(q[7:]), 'body': {'qpos': []}}} for q in traj]
    return json.dumps({'info': {'image': {'fps': fps}}, 'text': {'goal': 'g'}, 'data': data}).encode()


def test_endpoints_run_a_replay_on_the_simulated_robot():
    import time
    from fastapi import HTTPException
    from motioncreator import server
    session = server.replay_session()
    server.replay_connect(server.ReplayConnectInput(mock=True))
    project = project_from_motion_bytes(server.robot, episode_bytes(reach(31, .2)), 'x.json', 30)
    summary = server.replay_prepare(server.ReplayPrepareInput(project=project, fps=30))
    assert summary['ok']
    with pytest.raises(HTTPException) as refused:
        server.replay_start(server.ReplayStartInput(speed=4.0, confirm=False))
    assert refused.value.status_code == 422
    server.replay_start(server.ReplayStartInput(speed=4.0, confirm=True))
    deadline = time.monotonic() + 30
    while server.replay_status()['player']['state'] != 'idle' and time.monotonic() < deadline:
        time.sleep(0.1)
    status = server.replay_status()
    assert status['player']['state'] == 'idle' and status['player']['fault'] is None and status['mode'] == 'mock'
    server.replay_disconnect()
    assert not server.replay_status()['connected']
    session.shutdown()


def test_exported_lerobot_episode_imports_back_to_the_same_arms():
    from motioncreator import server
    from motioncreator.lerobot_format import lerobot_episode_to_qpos
    project = project_from_motion_bytes(server.robot, episode_bytes(reach(31, .2)), 'x.json', 30)
    episode = server.export_lerobot(server.ProjectInput(project=project, fps=30))
    qpos, fps = lerobot_episode_to_qpos(server.robot, json.dumps(episode).encode())
    assert fps == 30 and len(qpos) == 31
    np.testing.assert_allclose(qpos[:, ARM_COLUMNS], reach(31, .2), atol=1e-9)


# ---- waist (sent in the same rt/arm_sdk message, as robot-studio does) -----------------------------------------------------
def waist_project(robot, turn=.4, frames=31):
    """A motion that turns the waist yaw out and back while the arms stay at HOME_ARMS."""
    from motioncreator.motion import compile_motion
    project = project_from_motion_bytes(robot, episode_bytes(np.tile(HOME_ARMS, (frames, 1))), 'x.json', 30)
    swing = turn * np.sin(np.pi * np.arange(len(project['keyframes'][0]['samples'])) / (len(project['keyframes'][0]['samples']) - 1))
    for sample, yaw in zip(project['keyframes'][0]['samples'], swing):
        sample[7 + 12] = float(yaw)
    return project, compile_motion(robot, project, 30)


def test_a_plan_with_the_waist_carries_17_joints_and_no_longer_reports_the_waist_as_dropped(robot):
    _, motion = waist_project(robot)
    arms, upper = build_plan(robot, motion, 30), build_plan(robot, motion, 30, waist=True)
    assert arms['traj'].shape[1] == 14 and arms['joints'] == C.ARM_SLICE
    assert any(item['name'] == '재생 가능성' and '허리 요' in item['text'] for item in arms['checks']['items'])
    assert upper['traj'].shape[1] == 17 and upper['joints'] == C.UPPER_SLICE and upper['checks']['ok']
    assert not any(item['name'] == '재생 가능성' and '허리 요' in item['text'] for item in upper['checks']['items'])
    assert abs(upper['traj'][:, 0].max() - .4) < .02                       # the waist yaw column is the motion's


def test_a_waist_replay_moves_the_simulated_waist_and_returns_it(robot):
    _, motion = waist_project(robot)
    plan = build_plan(robot, motion, 30, waist=True)
    clock = Clock()
    backend = MockBackend(HOME_ARMS, clock=clock, threaded=False)
    player = ReplayPlayer(lambda: backend, backend.state, clock=clock, threaded=False)
    player.load(plan['traj'], 30, plan['checks'], joints=plan['joints'])
    player.start(1.0, auto_return=True)
    yaws = []
    for _ in range(400):
        clock.now += .02
        backend.step()
        player.tick()
        yaws.append(backend.q[12])
        if player.state == 'idle':
            break
    assert player.state == 'idle' and player.fault is None
    assert max(yaws) > .35 and abs(backend.q[12]) < 1e-3                    # it turned, then came back to where it started
    assert all(len(sent) == 17 for sent in backend.sent)
    with pytest.raises(ValueError):
        player.load(plan['traj'][:, 3:], 30, plan['checks'], joints=C.UPPER_SLICE)


def test_the_writer_takes_the_arms_or_the_waist_and_arms_and_the_waist_gets_no_feedforward(robot):
    from motioncreator.replay.arm_sdk_writer import validate_joints
    from motioncreator.replay.backends import ArmSdkBackend
    assert validate_joints(None) == list(range(15, 29)) and validate_joints(range(12, 29)) == list(range(12, 29))
    with pytest.raises(ValueError):
        validate_joints(range(0, 29))
    assert gains(12) == (300.0, 3.0)                                          # the waist: as robot-studio's controller holds it
    written = []
    backend = ArmSdkBackend('eth0', lambda q14, waist: np.full(14, 2.0) if waist is not None else np.ones(14), lambda: dict(C.ARM_SDK_DEFAULTS))
    backend._write = lambda **op: written.append(op)
    backend.joints = C.UPPER_SLICE
    backend.send(np.r_[.1, 0., 0., HOME_ARMS])
    assert written[-1]['tau'][:3] == [0., 0., 0.] and written[-1]['tau'][3] == 2.0 * C.FF_DEFAULT and len(written[-1]['q']) == 17
    backend.joints = C.ARM_SLICE
    backend.send(HOME_ARMS)
    assert len(written[-1]['q']) == 14 and written[-1]['tau'][0] == 1.0 * C.FF_DEFAULT


# ---- sending one pose (the editor's or a card's) ---------------------------------------------------------------------------
def test_a_pose_is_reached_held_then_a_replay_continues_from_it_and_home_releases(robot):
    clock = Clock()
    backend = MockBackend(HOME_ARMS, clock=clock, threaded=False)
    player = ReplayPlayer(lambda: backend, backend.state, clock=clock, threaded=False)
    lower, upper = np.full(14, -3.0), np.full(14, 3.0)
    goal = HOME_ARMS + np.r_[.3, 0, 0, -.2, 0, 0, 0, .3, 0, 0, -.2, 0, 0, 0]
    player.move_to(goal, C.ARM_SLICE, lower, upper)
    run(player, backend, clock, 4.0)
    assert player.state == 'holding' and np.allclose(backend.q[C.ARM_SLICE], goal, atol=1e-6)
    run(player, backend, clock, 1.0)
    assert player.state == 'holding' and np.allclose(backend.sent[-1], goal)          # it keeps commanding the pose
    nearer = goal - np.r_[.1, 0, 0, 0, 0, 0, 0, .1, 0, 0, 0, 0, 0, 0]
    player.move_to(nearer, C.ARM_SLICE, lower, upper)                                # a second pose from the held one
    run(player, backend, clock, 3.0)
    assert player.state == 'holding' and np.allclose(backend.q[C.ARM_SLICE], nearer, atol=1e-6)
    traj = reach()
    player.load(traj, 30.0, plan_checks(traj))                                       # checking a motion while holding is fine
    player.start(1.0, auto_return=True)
    assert player.state == 'approach'
    run(player, backend, clock, 12.0)
    assert player.state == 'idle' and not backend.engaged                            # played, returned to where it engaged, released
    np.testing.assert_allclose(backend.q[C.ARM_SLICE], HOME_ARMS, atol=1e-6)


def test_a_pose_outside_the_limits_or_for_other_joints_is_refused():
    clock = Clock()
    backend = MockBackend(HOME_ARMS, clock=clock, threaded=False)
    player = ReplayPlayer(lambda: backend, backend.state, clock=clock, threaded=False)
    lower, upper = np.full(14, -1.0), np.full(14, 1.0)
    with pytest.raises(CommandError):
        player.move_to(np.full(14, 2.0), C.ARM_SLICE, lower, upper)
    player.move_to(HOME_ARMS, C.ARM_SLICE, lower, upper)
    run(player, backend, clock, 1.0)
    with pytest.raises(CommandError):
        player.move_to(np.r_[0., 0., 0., HOME_ARMS], C.UPPER_SLICE, np.full(17, -3.), np.full(17, 3.))
    with pytest.raises(CommandError):
        player.load(np.zeros((5, 17)), 30.0, plan_checks(np.zeros((5, 14))), joints=C.UPPER_SLICE)


def test_the_session_sends_an_editor_pose_with_the_waist(robot):
    import time
    session = ReplaySession(lambda: robot)
    session.connect(None, mock=True)
    q = np.asarray(robot.home, float).copy()
    q[7 + 12] = .3                                                                   # waist yaw
    q[7 + 15] = -.4                                                                  # left shoulder pitch
    with pytest.raises(CommandError):
        session.move_to(q.tolist(), True, confirm=False)
    session.move_to(q.tolist(), True, confirm=True)
    deadline = time.monotonic() + 10
    while session.player.state != 'holding' and time.monotonic() < deadline:
        time.sleep(.05)
    state = session.backend.state()
    assert session.player.state == 'holding' and abs(state.q[12] - .3) < 1e-3 and abs(state.q[15] + .4) < 1e-3
    session.home()
    deadline = time.monotonic() + 10
    while session.player.state != 'idle' and time.monotonic() < deadline:
        time.sleep(.05)
    assert session.player.state == 'idle'
    session.shutdown()
