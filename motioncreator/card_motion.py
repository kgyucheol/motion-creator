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
from scipy.spatial.transform import Rotation, Slerp

from .dynamic_clip import (APPROX_DEG, APPROX_MM, ARM_WAIST_COLUMNS, DEFAULT_MAX_SHIFT_M, FREEDOM, HANDS, OK_DEG, OK_MM,
                           RELAXATIONS, WAIST_MODES, _level, object_delta, scene_object_pose)
from .robot import FEET, Robot
from .row_ends import STAND_IN_SHAPES, row_goals, row_of

SEARCH_STEPS = 7
REACH_MM, REACH_DEG = 10., 5.        # a hand this close to its goal counts as having reached it; past that the move stops short
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
        if target.get('ends') is not None:
            out['target']['ends'] = _validate_ends(target['ends'])
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
    shift = float(spec.get('max_shift_m', 1. if target.get('ends') else DEFAULT_MAX_SHIFT_M))     # a row-end grasp moves to a different row, not a nudge
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


def _validate_ends(spec):
    """The stand-in row the card was authored against (`ends.reference`), so the hands can be moved to the ends of another row."""
    reference = spec.get('reference') if isinstance(spec, dict) else None
    if not isinstance(reference, dict) or reference.get('shape') not in STAND_IN_SHAPES:
        raise ValueError('Row-end grasp needs a box or cylinder stand-in as its reference')
    size = np.asarray(reference.get('size'), dtype=float)
    position = np.asarray(reference.get('position'), dtype=float)
    quaternion = np.asarray(reference.get('quaternion_xyzw'), dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or size.min() <= 0 or size.max() > 5:
        raise ValueError('Row-end reference size must be three positive values within 5 m')
    if position.shape != (3,) or not np.isfinite(position).all() or np.abs(position).max() > 5:
        raise ValueError('Row-end reference position must be three finite values within 5 m')
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-3:
        raise ValueError('Row-end reference orientation must be a normalised xyzw quaternion')
    if row_of({'shape': reference['shape'], 'size': size, 'position': position, 'quaternion_xyzw': quaternion}) is None:
        raise ValueError('Row-end reference must lie roughly horizontal')
    out = {'reference': {'shape': reference['shape'], 'size': size.tolist(), 'position': position.tolist(),
                         'quaternion_xyzw': quaternion.tolist()}}
    if spec.get('tcp') is not None:
        tcp = spec['tcp']
        values = [tcp.get(key) for key in ('inward_m', 'lateral_m', 'up_m')] if isinstance(tcp, dict) else []
        if len(values) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and abs(v) <= .5 for v in values):
            raise ValueError('Row-end TCP placement needs inward, lateral and up within 0.5 m')
        out['tcp'] = dict(zip(('inward_m', 'lateral_m', 'up_m'), map(float, values)))
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
    if target['kind'] == 'object' and target.get('ends'):
        return _row_end_goals(spec, hands, project, index)
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


def _row_end_goals(spec, hands, project, index):
    """Goals that put each hand at its end of the tracked row (see row_ends), plus the card's own TCP offset."""
    from .motion import project_scene_objects
    target = spec['target']
    item = next((item for item in project_scene_objects(project) if item.get('id') == target['object_id']), None)
    if item is None:
        return None, None, _report('missing_object', index, 'cartesian')
    row = row_of(item)
    if row is None:
        return None, None, _report('not_a_row', index, 'cartesian')
    reference = row_of(target['ends']['reference'])
    tcp = target['ends'].get('tcp')
    goals, _ = row_goals(reference, row, hands, None if tcp is None else (tcp['inward_m'], tcp['lateral_m'], tcp['up_m']))
    offset = spec['offset']
    turn = Rotation.from_euler('z', offset['yaw_deg'], degrees=True)
    goals = {hand: (position + np.asarray(offset['position']), turn * rotation) for hand, (position, rotation) in goals.items()}
    shift = max(float(np.linalg.norm(goals[hand][0] - hands[hand][0])) for hand in goals)
    angle = max(float(np.degrees((goals[hand][1] * Rotation.from_matrix(hands[hand][1]).inv()).magnitude())) for hand in goals)
    return goals, {'shift_mm': shift * 1000, 'rotation_deg': angle, 'hands': hands}, None


ROW_RELAXATIONS = (*RELAXATIONS, (.02, 400))       # row-end grasps may straighten the elbows freely: last, nearly no posture preference
WAIST_PITCH_STEPS_DEG = (5., 10., 15., 20., 25., 29.)    # forward bend tried, smallest first, only when the arms alone do not reach (the joint stops at 29.8°)


def _solve_once(robot: Robot, end_q, spec, goals, relaxations, waist_pitch_deg=None):
    """One IK solve from the recorded end. With `waist_pitch_deg` the waist is released and its pitch aimed at the recorded value plus that bend."""
    targets = {hand: goal[0] for hand, goal in goals.items()}
    orientations = {hand: goal[1].as_quat() for hand, goal in goals.items()}
    locked = spec['waist'] == 'lock' and waist_pitch_deg is None
    lock = ['pelvis', 'waist'] if locked else ['pelvis']
    waist_joints = {}
    if waist_pitch_deg is not None:
        for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
            waist_joints[name] = float(end_q[7 + robot.names.index(name)])
        waist_joints['waist_pitch_joint'] = min(waist_joints['waist_pitch_joint'] + np.radians(waist_pitch_deg),
                                                float(robot.upper[7 + robot.names.index('waist_pitch_joint')]) - 1e-3)
    best, initial = None, end_q
    for posture_weight, evaluations in relaxations:
        candidate, info = robot.solve(initial, end_q, pins=(*FEET, 'pelvis'), angle_pins=lock, mode='free', resistance=0.,
                                      selected_targets=targets, orientation_targets=orientations, joint_targets=waist_joints or None,
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


def _solve_end(robot: Robot, end_q, spec, goals, coarse=False):
    """The waist and arm joints that put the hands on `goals`, starting from the recorded end. Returns (q, errors).

    A row-end grasp first tries with the waist as the card asks (locked by default) and the elbows free to straighten;
    only when the hands still miss does it bend the waist forward, as little as possible (5° steps up to 29°; `coarse`, used while
    searching how far the hands can go, tries only the largest bends to stay fast)."""
    target = spec.get('target') or {}
    if not (target.get('kind') == 'object' and target.get('ends')):
        return _solve_once(robot, end_q, spec, goals, RELAXATIONS)
    best = _solve_once(robot, end_q, spec, goals, ROW_RELAXATIONS)
    if best[1] < REACH_MM and best[2] < REACH_DEG:
        return best
    for bend in WAIST_PITCH_STEPS_DEG[-1:] if coarse else WAIST_PITCH_STEPS_DEG:
        trial = _solve_once(robot, end_q, spec, goals, ROW_RELAXATIONS, bend)
        if trial[1] < REACH_MM and trial[2] < REACH_DEG:
            return trial
        if trial[1] < best[1]:
            best = trial
    return best


def _scaled_goals(base, goals, fraction):
    """The goals a fraction of the way from where the recorded hands are to the real goals (position and orientation)."""
    scaled = {}
    for hand, (position, rotation) in goals.items():
        start_position, start_matrix = base['hands'][hand]
        start = Rotation.from_matrix(start_matrix)
        scaled[hand] = (start_position + fraction * (position - start_position),
                        Slerp([0., 1.], Rotation.concatenate([start, rotation]))([fraction])[0])
    return scaled


def _reach(robot: Robot, end_q, spec, goals, base):
    """Solve the end of a cartesian card, going only as far toward the goals as the arms can really reach.

    Returns (q, position error mm, angle error deg, fraction): `fraction` is how much of the requested move was made.
    The move is capped at `max_shift_m`; beyond that or beyond what the arms reach (hands within REACH_MM / REACH_DEG of
    the goal), the largest part of the move that still meets that is used."""
    shift = base['shift_mm'] / 1000
    cap = min(1., spec['max_shift_m'] / shift) if shift > 1e-9 else 1.

    def attempt(fraction):
        return _solve_end(robot, end_q, spec, goals if fraction >= 1. else _scaled_goals(base, goals, fraction), coarse=fraction < 1.)
    def reached(trial):
        return trial[1] < REACH_MM and trial[2] < REACH_DEG
    result = attempt(cap)
    if reached(result) or (cap >= 1. and _level(result[1], result[2]) != 'unreachable'):
        return (*result, cap)
    best, low, high = (np.array(end_q, dtype=float), 0., 0.), 0., cap          # no move at all is always reachable
    for _ in range(SEARCH_STEPS):
        middle = (low + high) / 2
        trial = attempt(middle)
        if reached(trial):
            best, low = trial, middle
        else:
            high = middle
    return (*best, low)


def resolve_project_motions(robot: Robot, project, advise=True):
    """Apply every card's motion mode. Returns (project, {card index: report}); the input project is not modified.

    With `advise`, a card that ends on a joint limit or out of reach also reports how far the target that causes it
    could move in the same direction without that (`safe_shift_mm`, `safe_fraction`, `caused_by`)."""
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
            elif base['shift_mm'] < 1e-3 and base['rotation_deg'] < 1e-4 and not np.any(spec['offset']['position']) \
                    and not spec['offset']['yaw_deg']:
                report = _report('unchanged', index, mode, shift_mm=0., rotation_deg=0.)
            else:
                solved, position_error, angle_error, fraction = _reach(robot, end, spec, goals, base)
                exit_offset = solved[COLUMNS] - end[COLUMNS]
                status = 'partial' if fraction < 1. else _level(position_error, angle_error)
                report = _report(status, index, mode, shift_mm=base['shift_mm'], rotation_deg=base['rotation_deg'],
                                 max_position_error_mm=position_error, max_angle_error_deg=angle_error)
                achieved = robot.data(solved)
                report['hand_errors_mm'] = {hand: ((robot.point(achieved, hand)[0] - goal[0]) * 1000).tolist() for hand, goal in goals.items()}
                if fraction < 1.:
                    report.update(reached_fraction=fraction, reached_shift_mm=fraction * base['shift_mm'],
                                  max_shift_mm=spec['max_shift_m'] * 1000)
                if status in ('unreachable', 'partial') and broken is None:
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
    if advise:
        _advise(robot, project, reports)
    return {**project, 'keyframes': resolved}, reports


FLAGGED = ('limit', 'unreachable', 'partial')


def _flagged(reports):
    return [index for index, report in reports.items() if report['status'] in FLAGGED]


def _target_shift(robot: Robot, project, index):
    """How far the target of cartesian card `index` lies from where the recording has it: (vector in m, how to scale it back)."""
    frame = project['keyframes'][index]
    spec = validate_motion(frame['motion'])
    target = spec['target']
    if target['kind'] == 'object':
        pose = scene_object_pose(project, target['object_id'])
        if pose is None:
            return None
        reference = np.asarray(target['reference_pose']['position'], dtype=float)
        shift = np.asarray(pose[0], dtype=float) - reference

        def scaled(fraction):
            moved = reference + (1 - fraction) * shift      # the reference moves toward the object: less of the move is left
            motion = {**frame['motion'], 'target': {**target, 'reference_pose': {**target['reference_pose'], 'position': moved.tolist()}}}
            return {**frame, 'motion': motion}
        return shift, scaled
    _, end = _card_ends(robot, frame)
    data = robot.data(end)
    middle = np.mean([robot.point(data, f'{side}_hand')[0] for side in spec['hands']], axis=0)
    shift = np.asarray(target['point'], dtype=float) - middle

    def scaled(fraction):
        motion = {**frame['motion'], 'target': {'kind': 'point', 'point': (middle + fraction * shift).tolist()}}
        return {**frame, 'motion': motion}
    return shift, scaled


def _advise(robot: Robot, project, reports):
    """Find, for every cartesian card that causes a flagged card, the largest part of its move that stays clear."""
    flagged = _flagged(reports)
    if not flagged:
        return
    frames = project['keyframes']
    causes = {}
    for index in flagged:
        cause = next((i for i in range(index, -1, -1) if frames[i].get('motion', {}).get('mode') == 'cartesian'), None)
        if cause is not None:
            causes.setdefault(cause, []).append(index)
    for cause, affected in causes.items():
        found = _target_shift(robot, project, cause)
        if found is None:
            continue
        shift, scaled = found

        def clear(fraction):
            trial = {**project, 'keyframes': [scaled(fraction) if i == cause else frame for i, frame in enumerate(frames)]}
            _, trial_reports = resolve_project_motions(robot, trial, advise=False)
            return not _flagged(trial_reports)
        if not clear(0.):
            continue                      # flagged even where the recording has it: moving the target is not the cure
        low, high = 0., 1.
        for _ in range(SEARCH_STEPS):
            middle = (low + high) / 2
            if clear(middle):
                low = middle
            else:
                high = middle
        advice = {'safe_fraction': low, 'safe_shift_mm': (shift * low * 1000).tolist(), 'requested_shift_mm': (shift * 1000).tolist()}
        for index in affected:
            reports[index].update(caused_by=cause, **advice)
        if cause not in affected:
            reports[cause].update(blocks=affected, **advice)      # the card to move is not the one that complains
