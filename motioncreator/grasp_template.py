"""Symmetric, parametric two-hand pick motion built from a teleoperated demonstration.

`extract_template` turns the demonstration clips (one per phase) into a left/right symmetric template:
for every phase a joint-delta curve, plus the hand poses at the grasp relative to the object and the
carry displacement. `generate_motion` plays the phases back for an object on the body's centre line:

* `delta` phases add the template's joint-delta curve to wherever the previous phase ended (the action matters),
* `cartesian` phases drive both TCPs to a coordinate with IK (the target matters).

The robot is assumed to stand with the pelvis at the origin and the waist locked, as in the recording.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .dynamic_clip import ARM_WAIST_COLUMNS, RELAXATIONS, _level
from .robot import FEET, Robot

FORMAT = 'motioncreator.grasp_template.v1'
CURVE_POINTS = 61
GRID_SECONDS = .1
DEFAULT_METHODS = ('delta', 'delta', 'cartesian', 'delta', 'cartesian', 'delta', 'delta')
HANDS = ('left_hand', 'right_hand')
DEFAULT_HAND_SPEED_MPS = .15
LIMIT_FALLBACK_DEG = .5
JOINT_COLUMNS = slice(7, 7 + 29)


TEMPLATE_PATH = Path(__file__).with_name('templates') / 'pick_cube_bimanual.json'


def load_template(path=TEMPLATE_PATH):
    return json.loads(Path(path).read_text())


def mirror_map(robot: Robot):
    """Joint permutation and sign that map a configuration onto its left/right mirror image."""
    names = robot.names
    permutation = np.arange(len(names))
    sign = np.ones(len(names))
    for index, name in enumerate(names):
        if name.startswith('left_'):
            other = names.index('right_' + name[5:])
            permutation[index], permutation[other] = other, index
        if 'roll' in name or 'yaw' in name:
            sign[index] = -1.
    return permutation, sign


def mirrored(robot: Robot, q):
    permutation, sign = mirror_map(robot)
    out = np.array(q, dtype=float)
    out[JOINT_COLUMNS] = sign * out[7:][permutation]
    return out


def symmetrized(robot: Robot, q):
    out = (np.asarray(q, dtype=float) + mirrored(robot, q)) / 2
    out[:7] = np.asarray(q, dtype=float)[:7]
    return out


def _resample(samples, count):
    samples = np.asarray(samples, dtype=float)
    source = np.linspace(0., 1., len(samples))
    target = np.linspace(0., 1., count)
    return np.stack([np.interp(target, source, samples[:, column]) for column in range(samples.shape[1])], axis=1)


def _hand_poses(robot: Robot, q):
    data = robot.data(q)
    return {hand: robot.point(data, hand) for hand in HANDS}


def _mid(poses):
    return (poses['left_hand'][0] + poses['right_hand'][0]) / 2


def extract_template(robot: Robot, clips, methods=DEFAULT_METHODS):
    """Template from demonstration clips (keyframes with `samples`), one clip per phase, in playing order."""
    clips = list(clips)
    if len(clips) != len(methods):
        raise ValueError(f'The template needs {len(methods)} clips, one per phase')
    if any(not clip.get('samples') for clip in clips):
        raise ValueError('Every phase must be a motion clip')
    phases, previous_end, joint_gap = [], None, 0.
    mirrors = []
    for clip, method in zip(clips, methods):
        raw = np.stack([robot.validate_q(sample) for sample in clip['samples']])
        if previous_end is not None:
            joint_gap = max(joint_gap, float(np.abs(raw[0] - previous_end)[JOINT_COLUMNS].max()))
        previous_end = raw[-1]
        mirrors.append(float(np.degrees(np.sqrt(np.mean((raw - np.stack([mirrored(robot, q) for q in raw]))[:, ARM_WAIST_COLUMNS] ** 2)))))
        sym = _resample(np.stack([symmetrized(robot, q) for q in raw]), CURVE_POINTS)
        tcp_start, tcp_end = _hand_poses(robot, sym[0]), _hand_poses(robot, sym[-1])
        tcp = {hand: {'dpos': (tcp_end[hand][0] - tcp_start[hand][0]).tolist(),
                      'drot_xyzw': Rotation.from_matrix(tcp_end[hand][1] @ tcp_start[hand][1].T).as_quat().tolist()} for hand in HANDS}
        phases.append({'tcp': tcp, 'name': clip['name'], 'duration': float(clip['duration']), 'method': method, 'settle': False,
                       'delta': (sym - sym[0])[:, ARM_WAIST_COLUMNS].tolist(), 'start': sym[0].tolist(), 'end': sym[-1].tolist()})
    phases[2]['goal'], phases[4]['goal'] = 'object', 'carry'     # where the two cartesian phases take the hands
    phases[-1]['settle'] = True      # the last phase ends in the demonstrated rest pose whatever happened before
    first = np.asarray(phases[0]['start'])
    template = {'format': FORMAT, 'phases': phases, 'start_pose': first.tolist(),
                'recording_asymmetry_deg_rms': mirrors, 'max_boundary_gap_rad': joint_gap}

    contact = _hand_poses(robot, np.asarray(phases[2]['end']))
    centre = _mid(contact)
    template['contact'] = {'mid': centre.tolist(), 'hands': {
        hand: {'position': (contact[hand][0] - centre).tolist(), 'quaternion_xyzw': Rotation.from_matrix(contact[hand][1]).as_quat().tolist()}
        for hand in HANDS}}
    before, after = _hand_poses(robot, np.asarray(phases[4]['start'])), _hand_poses(robot, np.asarray(phases[4]['end']))
    template['carry'] = {'target': _mid(after).tolist(), 'delta': (_mid(after) - _mid(before)).tolist()}
    return template


def _smooth(u):
    return u * u * (3 - 2 * u)


def _solve_clip(robot: Robot, start_q, goals, count, duration, posture_curve):
    """Move both TCPs from where `start_q` has them to `goals` ({hand: (position, rotation)}) and return (samples, report)."""
    base = _hand_poses(robot, start_q)
    steps = max(1, round(duration / GRID_SECONDS))
    grid = np.unique(np.r_[np.round(np.linspace(0, count - 1, steps + 1)).astype(int), count - 1])
    slerps = {hand: Slerp([0., 1.], Rotation.from_matrix(np.stack([base[hand][1], goals[hand][1]]))) for hand in HANDS}
    pins = (*FEET, 'pelvis')
    lock = ['pelvis', 'waist']
    solved, worst_mm, worst_deg = {}, 0., 0.
    previous = np.asarray(start_q, dtype=float)
    for index in grid:
        u = index / max(1, count - 1)
        s = _smooth(u)
        targets = {hand: base[hand][0] + s * (goals[hand][0] - base[hand][0]) for hand in HANDS}
        orientations = {hand: slerps[hand]([s])[0].as_quat() for hand in HANDS}
        posture = np.array(start_q, dtype=float)
        posture[ARM_WAIST_COLUMNS] = np.clip(posture[ARM_WAIST_COLUMNS] + posture_curve[index],
                                             robot.lower[3 + 12:3 + 29], robot.upper[3 + 12:3 + 29])
        best, initial = None, previous
        for posture_weight, evaluations in RELAXATIONS:
            candidate, info = robot.solve(initial, start_q, pins=pins, angle_pins=lock, mode='free', resistance=0.,
                                          selected_targets=targets, orientation_targets=orientations,
                                          posture_reference=posture, posture_weight=posture_weight, max_nfev=evaluations)
            if best is None or info['target_error_mm'] < best[1]['target_error_mm']:
                best = (candidate, info)
            if info['converged'] and not info.get('rejected'):
                break
            initial = candidate
        result = np.array(start_q, dtype=float)
        result[ARM_WAIST_COLUMNS] = best[0][ARM_WAIST_COLUMNS]
        solved[int(index)] = result
        previous = result
        achieved = _hand_poses(robot, result)
        for hand in HANDS:
            worst_mm = max(worst_mm, float(np.linalg.norm(achieved[hand][0] - targets[hand])) * 1000)
            worst_deg = max(worst_deg, float(np.degrees((Rotation.from_matrix(achieved[hand][1]) * Rotation.from_quat(orientations[hand]).inv()).magnitude())))
    indices = sorted(solved)
    values = np.stack([solved[i] for i in indices])
    out = np.tile(np.asarray(start_q, dtype=float), (count, 1))
    for column in range(ARM_WAIST_COLUMNS.start, ARM_WAIST_COLUMNS.stop):
        out[:, column] = np.interp(np.arange(count), indices, values[:, column])
    final_mm = max(float(np.linalg.norm(_hand_poses(robot, out[-1])[hand][0] - goals[hand][0])) * 1000 for hand in HANDS)
    return out, {'status': _level(worst_mm, worst_deg), 'max_position_error_mm': worst_mm, 'max_angle_error_deg': worst_deg,
                 'final_position_error_mm': final_mm}


def count_for(duration, fps):
    return max(2, round(duration * fps) + 1)


def _delta_clip(robot: Robot, phase, q, curve, count):
    """The phase's joint-delta curve added to `q`, clamped to the joint limits. Returns (samples, clamped degrees)."""
    curve = _resample(curve, count)
    samples = np.tile(q, (count, 1))
    arms = slice(ARM_WAIST_COLUMNS.start, ARM_WAIST_COLUMNS.stop)
    if phase.get('settle'):
        # follow the demonstrated path, starting from where the robot actually is and fading that offset out
        template_start = np.asarray(phase['start'])[arms]
        fade = _smooth(np.linspace(0., 1., count))[:, None]
        samples[:, arms] = template_start + curve + (q[arms] - template_start) * (1 - fade)
    else:
        samples[:, arms] = q[arms] + curve
    clipped = np.clip(samples[:, arms], robot.lower[3 + 12:3 + 29], robot.upper[3 + 12:3 + 29])
    overshoot = float(np.degrees(np.abs(clipped - samples[:, arms]).max()))
    samples[:, arms] = clipped
    return samples, overshoot


def _object_goals(template, position, yaw_deg):
    turn = Rotation.from_euler('z', yaw_deg, degrees=True)
    centre = np.asarray(position, dtype=float)
    return {hand: (centre + turn.apply(spec['position']), (turn * Rotation.from_quat(spec['quaternion_xyzw'])).as_matrix())
            for hand, spec in template['contact']['hands'].items()}


def generate_motion(robot: Robot, template, object_position, object_yaw_deg=0., fps=30, carry_target=None,
                    hand_speed_mps=DEFAULT_HAND_SPEED_MPS):
    """Return (keyframes, reports): one motion clip per phase for an object centred at `object_position`.

    `carry_target` is the point (x, y, z) the hands' midpoint should reach in the carry phase; without it the
    place the demonstration carried to is used.
    """
    if template.get('format') != FORMAT:
        raise ValueError('Unknown grasp template')
    if not 1 <= fps <= 120:
        raise ValueError('FPS must be 1–120')
    position = np.asarray(object_position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Object position must be three finite values')
    q = robot.validate_q(template['start_pose'])
    clips, reports = [], []
    for index, phase in enumerate(template['phases']):
        curve = np.asarray(phase['delta'], dtype=float)
        duration = float(phase['duration'])
        report = {'index': index, 'name': phase['name'], 'method': phase['method']}
        method = phase['method']
        if method == 'delta':
            samples, overshoot = _delta_clip(robot, phase, q, curve, count_for(duration, fps))
            report['status'] = 'ok' if overshoot < 1e-6 else 'limit'
            report['clamped_deg'] = overshoot
            if overshoot > LIMIT_FALLBACK_DEG and not phase.get('settle'):
                # the demonstrated joint motion does not fit for this object: do what it did to the hands instead
                method = 'cartesian'
                report.update(method='cartesian', fallback='joint limit', clamped_deg=overshoot)
        if method == 'cartesian':
            now = _hand_poses(robot, q)
            if phase.get('goal') == 'object':
                goals = _object_goals(template, position, object_yaw_deg)
            elif phase['method'] == 'delta':
                goals = {hand: (now[hand][0] + np.asarray(spec['dpos']), (Rotation.from_quat(spec['drot_xyzw']) * Rotation.from_matrix(now[hand][1])).as_matrix())
                         for hand, spec in phase['tcp'].items()}
            else:
                # a coordinate, not an offset from the object: where the load is taken to does not depend on where it was picked up
                shift = np.asarray(carry_target if carry_target is not None else template['carry']['target'], dtype=float) - _mid(now)
                goals = {hand: (now[hand][0] + shift, now[hand][1]) for hand in HANDS}
            travel = max(float(np.linalg.norm(goals[hand][0] - now[hand][0])) for hand in HANDS)
            duration = max(duration, travel / hand_speed_mps)
            count = count_for(duration, fps)
            samples, solved = _solve_clip(robot, q, goals, count, duration, _resample(curve, count))
            report.update(solved, travel_mm=travel * 1000)
        q = samples[-1].copy()
        report['duration'] = (len(samples) - 1) / fps
        reports.append(report)
        clips.append({'name': phase['name'], 'duration': (len(samples) - 1) / fps, 'qpos': samples[0].tolist(),
                      'pins': [], 'samples': samples.tolist()})
    return clips, reports
