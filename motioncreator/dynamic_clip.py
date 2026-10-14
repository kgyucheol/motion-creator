"""Dynamic motion clips: re-aim the hands of an imported clip at a scene object that has been moved.

A clip may carry a `dynamic` binding: the object the clip grasps, the pose that object had when the motion was
recorded (`reference_pose`), the clip samples where the hands start to follow the object and let go of it
(`window`), and which hands follow. The stored samples are never changed. `resolve_dynamic_clip` derives the
samples to play from the original samples plus the object's current pose:

    delta          = (object now) . (object at reference)^-1          rigid transform in world coordinates
    hand target(t) = blend(original hand(t), delta . original hand(t), w(t))
    sample(t)      = whole-body IK(hand targets, posture of the original sample, feet and pelvis pinned)

`w(t)` rises 0 -> 1 over the approach, stays 1 while the object is held and falls 1 -> 0 afterwards, so outside the
window the clip is exactly the original and neighbouring clips and poses still meet it where they did before.
Only the arms and the waist are changed; legs and root come from the original samples.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .robot import FEET, Robot

HANDS = ('left', 'right')
FREEDOM = ('yaw_xyz', 'full6d')
WAIST_MODES = ('free', 'lock')
WINDOW_KEYS = ('ramp_in_start', 'hold_start', 'hold_end', 'ramp_out_end')
DEFAULT_MAX_SHIFT_M = .25
ARM_WAIST_COLUMNS = slice(7 + 12, 7 + 29)       # waist yaw/roll/pitch, then the 14 arm joints (Unitree order)
GRID_SECONDS = .1
WINDOW_LIMIT = 100000
RELAXATIONS = ((.8, 180), (.3, 280), (.08, 400))   # posture weight, IK evaluations: tighten, then relax the preference
OK_MM, OK_DEG = 3., 2.
APPROX_MM, APPROX_DEG = 20., 10.


def validate_dynamic(spec, sample_count):
    """Return the normalised binding or raise ValueError (the message is shown to the user)."""
    if not isinstance(spec, dict):
        raise ValueError('Dynamic grasp settings must be an object')
    object_id = spec.get('object_id')
    if not isinstance(object_id, str) or not object_id:
        raise ValueError('Dynamic grasp needs an object')
    reference = spec.get('reference_pose')
    if not isinstance(reference, dict):
        raise ValueError('Dynamic grasp needs the object reference pose')
    position = np.asarray(reference.get('position'), dtype=float)
    quaternion = np.asarray(reference.get('quaternion_xyzw'), dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or np.abs(position).max() > 5:
        raise ValueError('Reference position must be three finite values within 5 m')
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-3:
        raise ValueError('Reference orientation must be a normalised xyzw quaternion')
    hands = spec.get('hands', list(HANDS))
    if not isinstance(hands, list) or not hands or any(hand not in HANDS for hand in hands) or len(set(hands)) != len(hands):
        raise ValueError('Dynamic grasp hands must be left and/or right')
    window = spec.get('window')
    if not isinstance(window, dict) or any(not isinstance(window.get(key), int) or isinstance(window.get(key), bool) for key in WINDOW_KEYS):
        raise ValueError('Dynamic grasp window needs four sample numbers')
    a, b, c, d = (window[key] for key in WINDOW_KEYS)
    # Numbers may lie outside the clip: splitting a clip shifts them, so a part keeps exactly the weights it had.
    if not -WINDOW_LIMIT <= a <= b <= c <= d <= WINDOW_LIMIT or d <= a:
        raise ValueError('Dynamic grasp window needs start <= hold start <= hold end <= end, with end after start')
    if d < 0 or a > sample_count - 1:
        raise ValueError(f'Dynamic grasp window must overlap the clip (samples 0–{sample_count - 1})')
    freedom = spec.get('freedom', 'yaw_xyz')
    waist = spec.get('waist', 'free')
    if freedom not in FREEDOM or waist not in WAIST_MODES:
        raise ValueError('Unknown dynamic grasp freedom or waist mode')
    shift = float(spec.get('max_shift_m', DEFAULT_MAX_SHIFT_M))
    if not np.isfinite(shift) or not .01 <= shift <= 2:
        raise ValueError('Maximum object shift must be 0.01–2 m')
    return {'object_id': object_id, 'reference_pose': {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist()},
            'hands': list(hands), 'window': dict(window), 'freedom': freedom, 'waist': waist, 'max_shift_m': shift}


def _smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3. - 2. * u)


def window_weights(window, count):
    """w(i) for every sample: smoothstep up over [ramp_in_start, hold_start], 1 on the hold, down over [hold_end, ramp_out_end].

    The window numbers are sample numbers of the clip but may lie outside 0..count-1 (see validate_dynamic).
    """
    a, b, c, d = (window[key] for key in WINDOW_KEYS)
    index = np.arange(count, dtype=float)
    weights = np.zeros(count)
    rising = (index >= a) & (index < b)
    weights[rising] = _smoothstep((index[rising] - a) / (b - a))
    weights[(index >= b) & (index <= c)] = 1.
    falling = (index > c) & (index <= d)
    weights[falling] = 1. - _smoothstep((index[falling] - c) / (d - c))
    return weights


def object_delta(spec, now_position, now_quaternion_xyzw):
    """(rotation, reference position, current position) of the rigid transform that carries the reference pose onto the current one."""
    reference = spec['reference_pose']
    reference_rotation = Rotation.from_quat(reference['quaternion_xyzw'])
    now_rotation = Rotation.from_quat(now_quaternion_xyzw)
    relative = now_rotation * reference_rotation.inv()
    if spec['freedom'] == 'yaw_xyz':
        relative = Rotation.from_euler('z', relative.as_euler('zyx')[0])
    return relative, np.asarray(reference['position'], dtype=float), np.asarray(now_position, dtype=float)


def _level(position_mm, angle_deg):
    if position_mm < OK_MM and angle_deg < OK_DEG:
        return 'ok'
    if position_mm < APPROX_MM and angle_deg < APPROX_DEG:
        return 'approx'
    return 'unreachable'


def _max_joint_speed(samples, dt):
    if len(samples) < 2:
        return 0.
    return float(np.abs(np.diff(samples[:, ARM_WAIST_COLUMNS], axis=0)).max() / dt)


def _report(status, spec, **extra):
    return {'status': status, 'object_id': spec['object_id'], 'hands': list(spec['hands']), **extra}


def resolve_dynamic_clip(robot: Robot, clip, now_position, now_quaternion_xyzw, spec=None):
    """Return (samples, report) for a clip with a `dynamic` binding. `samples` has the clip's shape and original length.

    Never raises for an object that is too far or a pose that cannot be reached: the report carries the status
    (`ok`, `approx`, `unreachable`, `too_far`, `unchanged`) and the best samples found, so editing stays responsive.
    """
    source = np.asarray(clip['samples'], dtype=float)
    spec = spec or validate_dynamic(clip['dynamic'], len(source))
    rotation, reference_position, now_position = object_delta(spec, now_position, now_quaternion_xyzw)
    shift = float(np.linalg.norm(now_position - reference_position))
    angle = float(np.degrees(rotation.magnitude()))
    base = {'shift_mm': shift * 1000, 'rotation_deg': angle}
    if shift > spec['max_shift_m']:
        return source, _report('too_far', spec, **base, max_shift_mm=spec['max_shift_m'] * 1000)
    if shift < 1e-6 and angle < 1e-4:
        return source, _report('unchanged', spec, **base, max_position_error_mm=0., max_angle_error_deg=0.)

    count = len(source)
    dt = float(clip['duration']) / max(1, count - 1)
    weights = window_weights(spec['window'], count)
    a, b, c, d = (spec['window'][key] for key in WINDOW_KEYS)
    step = max(1, round(GRID_SECONDS / dt))
    low, high = max(a, 0), min(d, count - 1)
    # The first and last sample are always solved when they are inside the window, so two clips cut from one
    # (which share that sample) resolve it to exactly the same pose.
    grid = sorted({*range(low, high + 1, step), low, high})
    lock = ['pelvis', 'waist'] if spec['waist'] == 'lock' else ['pelvis']
    pins = (*FEET, 'pelvis')
    solved = {}
    held = []
    solves = 0
    for index in grid:
        original = source[index]
        if weights[index] <= 0:
            solved[index] = original
            continue
        weight = float(weights[index])
        data = robot.data(original)
        targets, orientations = {}, {}
        for side in spec['hands']:
            hand = f'{side}_hand'
            position, matrix = robot.point(data, hand)
            moved = now_position + rotation.apply(position - reference_position)
            moved_rotation = rotation * Rotation.from_matrix(matrix)
            original_rotation = Rotation.from_matrix(matrix)
            blended = Slerp([0., 1.], Rotation.concatenate([original_rotation, moved_rotation]))([weight])[0]
            targets[hand] = position + weight * (moved - position)
            orientations[hand] = blended.as_quat()
        best = None
        initial = original          # not the previous solution: the result at a sample must not depend on its neighbours
        for posture_weight, evaluations in RELAXATIONS:
            candidate, info = robot.solve(
                initial, original, pins=pins, angle_pins=lock, mode='free', resistance=0.,
                selected_targets=targets, orientation_targets=orientations,
                posture_reference=original, posture_weight=posture_weight, max_nfev=evaluations)
            solves += 1
            if best is None or info['target_error_mm'] < best[1]['target_error_mm']:
                best = (candidate, info)
            if info['converged'] and not info.get('rejected'):
                break
            initial = candidate
        result = original.copy()
        result[ARM_WAIST_COLUMNS] = best[0][ARM_WAIST_COLUMNS]
        solved[index] = result
        # error measured on what is actually played: arms and waist from the solve, everything else original
        achieved = robot.data(result)
        position_error, angle_error = 0., 0.
        for hand in targets:
            position, matrix = robot.point(achieved, hand)
            position_error = max(position_error, float(np.linalg.norm(position - targets[hand])) * 1000)
            angle_error = max(angle_error, float(np.degrees(
                (Rotation.from_matrix(matrix) * Rotation.from_quat(orientations[hand]).inv()).magnitude())))
        held.append((weight, position_error, angle_error))

    # judge the grip on the samples that follow the object completely; a clip cut inside the ramps has none
    judged = [entry for entry in held if entry[0] >= .999] or held
    worst_mm = max((entry[1] for entry in judged), default=0.)
    worst_deg = max((entry[2] for entry in judged), default=0.)
    out = source.copy()
    indices = sorted(solved)
    values = np.stack([solved[i] for i in indices])
    for column in range(ARM_WAIST_COLUMNS.start, ARM_WAIST_COLUMNS.stop):
        out[low:high + 1, column] = np.interp(np.arange(low, high + 1), indices, values[:, column])
    speed = _max_joint_speed(out, dt)
    status = _level(worst_mm, worst_deg)
    return out, _report(status, spec, **base, max_position_error_mm=worst_mm, max_angle_error_deg=worst_deg,
                        max_joint_speed_rad_s=speed, original_max_joint_speed_rad_s=_max_joint_speed(source, dt),
                        solves=solves)


def scene_object_pose(project, object_id):
    from .motion import project_scene_objects
    for item in project_scene_objects(project):
        if item.get('id') == object_id:
            return item['position'], item['quaternion_xyzw']
    return None


def resolve_project_dynamics(robot: Robot, project):
    """Replace the samples of every dynamic clip by the resolved ones. Returns (project, {keyframe index: report}).

    The input project is not modified. A clip whose object is missing from the scene plays unchanged and reports `missing_object`.
    """
    frames = project['keyframes']
    if not any(frame.get('dynamic') for frame in frames):
        return project, {}
    reports, resolved = {}, []
    for index, frame in enumerate(frames):
        if not frame.get('dynamic'):
            resolved.append(frame)
            continue
        spec = validate_dynamic(frame['dynamic'], len(frame['samples']))
        pose = scene_object_pose(project, spec['object_id'])
        if pose is None:
            reports[index] = _report('missing_object', spec)
            resolved.append(frame)
            continue
        samples, report = resolve_dynamic_clip(robot, frame, *pose, spec=spec)
        reports[index] = report
        resolved.append({**frame, 'samples': samples.tolist(), 'qpos': samples[0].tolist()})
    return {**project, 'keyframes': resolved}, reports
