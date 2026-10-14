"""Per-card motion mode: how a keyframe or clip card moves relative to the card before it.

Every card (a pose keyframe or a motion clip) has a recorded start pose and end pose and, optionally, a `motion`:

* `recorded`  plays as recorded; whatever offset the previous cards left fades out over this card,
* `delta`     continues from where the previous card really ended and adds the recorded change (the action matters),
* `cartesian` the end is moved so the hands reach a target (a tracked scene object or an x, y, z point);
              the recorded shape in between is kept (the target matters).

Cards are resolved in order. What is carried from card to card is the *offset* (resolved minus recorded) of the
waist and arm joints at the card's end, so a retargeted card pulls every following `delta` card with it.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .dynamic_clip import (APPROX_DEG, APPROX_MM, ARM_WAIST_COLUMNS, DEFAULT_MAX_SHIFT_M, FREEDOM, HANDS, OK_DEG, OK_MM,
                           RELAXATIONS, WAIST_MODES, _level, object_delta, scene_object_pose)
from .robot import FEET, Robot

MODES = ('recorded', 'delta', 'cartesian')
TARGET_KINDS = ('object', 'point')
COLUMNS = slice(ARM_WAIST_COLUMNS.start, ARM_WAIST_COLUMNS.stop)


def validate_motion(spec):
    """Return the normalised `motion` of a card or raise ValueError (the message is shown to the user)."""
    if not isinstance(spec, dict):
        raise ValueError('Card motion must be an object')
    mode = spec.get('mode', 'recorded')
    if mode not in MODES:
        raise ValueError('Card motion mode must be recorded, delta or cartesian')
    out = {'mode': mode}
    if mode != 'cartesian':
        return out
    target = spec.get('target')
    if not isinstance(target, dict) or target.get('kind') not in TARGET_KINDS:
        raise ValueError('A cartesian card needs a target: an object or a point')
    if target['kind'] == 'object':
        object_id = target.get('object_id')
        if not isinstance(object_id, str) or not object_id:
            raise ValueError('The object target needs an object id')
        reference = target.get('reference_pose')
        if not isinstance(reference, dict):
            raise ValueError('The object target needs its reference pose')
        position = np.asarray(reference.get('position'), dtype=float)
        quaternion = np.asarray(reference.get('quaternion_xyzw'), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or np.abs(position).max() > 5:
            raise ValueError('Reference position must be three finite values within 5 m')
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-3:
            raise ValueError('Reference orientation must be a normalised xyzw quaternion')
        out['target'] = {'kind': 'object', 'object_id': object_id,
                         'reference_pose': {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}}
    else:
        point = np.asarray(target.get('point'), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all() or np.abs(point).max() > 5:
            raise ValueError('The point target needs three finite values within 5 m')
        out['target'] = {'kind': 'point', 'point': point.tolist()}
    hands = spec.get('hands', list(HANDS))
    if not isinstance(hands, list) or not hands or any(hand not in HANDS for hand in hands) or len(set(hands)) != len(hands):
        raise ValueError('Card motion hands must be left and/or right')
    freedom, waist = spec.get('freedom', 'yaw_xyz'), spec.get('waist', 'lock')
    if freedom not in FREEDOM or waist not in WAIST_MODES:
        raise ValueError('Unknown card motion freedom or waist mode')
    shift = float(spec.get('max_shift_m', DEFAULT_MAX_SHIFT_M))
    if not np.isfinite(shift) or not .01 <= shift <= 2:
        raise ValueError('Maximum target shift must be 0.01–2 m')
    offset = spec.get('offset') or {}
    offset_position = np.asarray(offset.get('position', [0., 0., 0.]), dtype=float)
    offset_yaw = float(offset.get('yaw_deg', 0.))
    if offset_position.shape != (3,) or not np.isfinite(offset_position).all() or np.abs(offset_position).max() > 1:
        raise ValueError('TCP offset must be three finite values within 1 m')
    if not np.isfinite(offset_yaw) or abs(offset_yaw) > 180:
        raise ValueError('TCP offset yaw must be within ±180 degrees')
    out.update(hands=list(hands), freedom=freedom, waist=waist, max_shift_m=shift,
               offset={'position': offset_position.tolist(), 'yaw_deg': offset_yaw})
    return out


def _smooth(u):
    return u * u * (3 - 2 * u)


def _card_ends(robot: Robot, frame):
    if frame.get('samples') is not None:
        return robot.validate_q(frame['samples'][0]), robot.validate_q(frame['samples'][-1])
    q = robot.validate_q(frame['qpos'])
    return q, q


def _report(status, index, mode, **extra):
    return {'index': index, 'mode': mode, 'status': status, **extra}


def _hand_goals(robot: Robot, end_q, spec, project, index):
    """(goals, base, report-or-None): where each hand has to go for a cartesian card, or why that is not possible."""
    target = spec['target']
    data = robot.data(end_q)
    hands = {f'{side}_hand': robot.point(data, f'{side}_hand') for side in spec['hands']}
    if target['kind'] == 'object':
        pose = scene_object_pose(project, target['object_id'])
        if pose is None:
            return None, None, _report('missing_object', index, 'cartesian')
        reference, now_position, now_quaternion = target['reference_pose'], pose[0], pose[1]
    else:
        middle = np.mean([position for position, _ in hands.values()], axis=0)
        reference = {'position': middle.tolist(), 'quaternion_xyzw': [0., 0., 0., 1.]}
        now_position, now_quaternion = target['point'], [0., 0., 0., 1.]
    rotation, reference_position, now_position = object_delta({**spec, 'reference_pose': reference}, now_position, now_quaternion)
    goals = {hand: (now_position + rotation.apply(position - reference_position), (rotation * Rotation.from_matrix(matrix)))
             for hand, (position, matrix) in hands.items()}
    shift = float(np.linalg.norm(now_position - reference_position))
    angle = float(np.degrees(rotation.magnitude()))
    return goals, {'shift_mm': shift * 1000, 'rotation_deg': angle, 'hands': hands}, None


def _solve_end(robot: Robot, end_q, spec, goals):
    """The waist and arm joints that put the hands on `goals`, starting from the recorded end. Returns (q, errors)."""
    targets = {hand: goal[0] for hand, goal in goals.items()}
    orientations = {hand: goal[1].as_quat() for hand, goal in goals.items()}
    lock = ['pelvis', 'waist'] if spec['waist'] == 'lock' else ['pelvis']
    best, initial = None, end_q
    for posture_weight, evaluations in RELAXATIONS:
        candidate, info = robot.solve(initial, end_q, pins=(*FEET, 'pelvis'), angle_pins=lock, mode='free', resistance=0.,
                                      selected_targets=targets, orientation_targets=orientations,
                                      posture_reference=end_q, posture_weight=posture_weight, max_nfev=evaluations)
        if best is None or info['target_error_mm'] < best[1]['target_error_mm']:
            best = (candidate, info)
        if info['converged'] and not info.get('rejected'):
            break
        initial = candidate
    result = np.array(end_q, dtype=float)
    result[COLUMNS] = best[0][COLUMNS]
    achieved = robot.data(result)
    position_error, angle_error = 0., 0.
    for hand, goal in goals.items():
        position, matrix = robot.point(achieved, hand)
        position_error = max(position_error, float(np.linalg.norm(position - goal[0])) * 1000)
        angle_error = max(angle_error, float(np.degrees((Rotation.from_matrix(matrix) * goal[1].inv()).magnitude())))
    return result, position_error, angle_error


def resolve_project_motions(robot: Robot, project):
    """Apply every card's motion mode. Returns (project, {card index: report}); the input project is not modified."""
    frames = project['keyframes']
    if not any(frame.get('motion') for frame in frames):
        return project, {}
    lower, upper = robot.lower[3 + 12:3 + 29], robot.upper[3 + 12:3 + 29]
    offset = np.zeros(COLUMNS.stop - COLUMNS.start)
    reports, resolved = {}, []
    broken = None            # the first card whose target could not be reached: what follows inherits its error
    for index, frame in enumerate(frames):
        spec = validate_motion(frame['motion']) if frame.get('motion') else {'mode': 'recorded'}
        start, end = _card_ends(robot, frame)
        entry, mode = offset, spec['mode']
        plain = {key: value for key, value in frame.items() if key != 'motion'}      # resolved: later passes must not apply it again
        report = None
        if mode == 'recorded':
            exit_offset = np.zeros_like(offset)
        elif mode == 'delta':
            exit_offset = offset
        else:
            goals, base, failure = _hand_goals(robot, end, spec, project, index)
            exit_offset = np.zeros_like(offset)
            if failure:
                report = failure
            elif base['shift_mm'] / 1000 > spec['max_shift_m']:
                report = _report('too_far', index, mode, shift_mm=base['shift_mm'], max_shift_mm=spec['max_shift_m'] * 1000)
            elif base['shift_mm'] < 1e-3 and base['rotation_deg'] < 1e-4 and not np.any(spec['offset']['position']) \
                    and not spec['offset']['yaw_deg']:
                report = _report('unchanged', index, mode, shift_mm=0., rotation_deg=0.)
            else:
                solved, position_error, angle_error = _solve_end(robot, end, spec, goals)
                exit_offset = solved[COLUMNS] - end[COLUMNS]
                report = _report(_level(position_error, angle_error), index, mode, shift_mm=base['shift_mm'],
                                 rotation_deg=base['rotation_deg'], max_position_error_mm=position_error, max_angle_error_deg=angle_error)
                if report['status'] == 'unreachable' and broken is None:
                    broken = index
        if report is None:
            report = _report('ok', index, mode)
        report['offset_deg'] = float(np.degrees(np.abs(exit_offset).max()))
        if broken is not None and index > broken and mode != 'recorded':
            report['after_unreachable'] = broken

        if frame.get('samples') is not None:
            source = np.stack([robot.validate_q(sample) for sample in frame['samples']])
            progress = _smooth(np.linspace(0., 1., len(source)))[:, None] if len(source) > 1 else np.ones((1, 1))
            blend = entry + (exit_offset - entry) * (progress if mode != 'delta' else 0.)
            samples = source.copy()
            samples[:, COLUMNS] = np.clip(source[:, COLUMNS] + blend, lower, upper)
            overshoot = float(np.degrees(np.abs(samples[:, COLUMNS] - (source[:, COLUMNS] + blend)).max()))
            resolved.append({**plain, 'samples': samples.tolist(), 'qpos': samples[0].tolist()})
        else:
            q = end.copy()
            q[COLUMNS] = np.clip(end[COLUMNS] + exit_offset, lower, upper)
            overshoot = float(np.degrees(np.abs(q[COLUMNS] - (end[COLUMNS] + exit_offset)).max()))
            resolved.append({**plain, 'qpos': q.tolist()})
        if overshoot > .5:
            report['clamped_deg'] = overshoot
            if report['status'] in ('ok', 'unchanged'):
                report['status'] = 'limit'
        offset = exit_offset
        if mode != 'recorded' or np.any(entry):
            reports[index] = report
    return {**project, 'keyframes': resolved}, reports
