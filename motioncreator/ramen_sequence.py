"""Object-relative two-tool sequence generation for extracting a ramen bundle."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from .robot import FEET, Robot


FORMAT = 'motioncreator.ramen-interaction.v1'


def _vector(value, name, *, length=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {length} finite numbers')
    return result


def _tcp_pose(robot: Robot, qpos, side: str):
    position, rotation = robot.point(robot.data(qpos), f'{side}_hand')
    return position, Rotation.from_matrix(rotation).as_quat()


def _unit(value, fallback):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-8 else np.asarray(fallback, dtype=float)


def _rotation_between(first, second):
    first, second = _unit(first, [0., 0., 1.]), _unit(second, [0., 0., 1.])
    cross = np.cross(first, second)
    dot = float(np.clip(np.dot(first, second), -1., 1.))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return Rotation.identity()
        perpendicular = _unit(np.cross(first, [1., 0., 0.]), np.cross(first, [0., 1., 0.]))
        return Rotation.from_rotvec(np.pi * perpendicular)
    return Rotation.from_rotvec(np.arccos(dot) * cross / np.linalg.norm(cross))


def _align_tool_to_box(quaternion, box_up):
    """Tilt the tool's currently vertical local axis to the tilted box up axis."""
    rotation = Rotation.from_quat(quaternion)
    axes = rotation.as_matrix()
    index = int(np.argmax(np.abs(axes.T @ np.array([0., 0., 1.]))))
    vertical = axes[:, index]
    if vertical[2] < 0:
        vertical = -vertical
    return (_rotation_between(vertical, box_up) * rotation).as_quat()


def _object_reference(item, taught, robot_position):
    center = _vector(item.get('position'), 'object position')
    object_rotation = Rotation.from_quat(_vector(item.get('quaternion_xyzw'), 'object quaternion', length=4)).as_matrix()
    collision_shape = item.get('collision_shape') or item.get('shape')
    dimensions = _vector(item.get('collision_size') or item.get('size'), 'object size')
    if collision_shape != 'cylinder':
        raise ValueError('라면 꺼내기 대상은 원통 충돌체가 있는 오브젝트여야 합니다.')
    collision_rotation = (Rotation.from_quat(item['collision_quaternion_xyzw']).as_matrix()
                          if item.get('collision_quaternion_xyzw') is not None else np.eye(3))
    collision_frame = object_rotation @ collision_rotation
    bundle_axis = collision_frame[:, 2]
    radial_axes = [collision_frame[:, 0], collision_frame[:, 1]]
    diameter, length = float(max(dimensions[0], dimensions[1])), float(dimensions[2])
    bundle_axis = _unit(bundle_axis, [1., 0., 0.])
    box_up = max(radial_axes, key=lambda axis: abs(float(np.dot(axis, [0., 0., 1.])))).copy()
    if box_up[2] < 0:
        box_up *= -1
    box_up = _unit(box_up, [0., 0., 1.])
    side_axis = _unit(np.cross(box_up, bundle_axis), [0., 1., 0.])
    taught_span = taught['left'][0] - taught['right'][0]
    if np.dot(side_axis, taught_span) < 0:
        side_axis *= -1
    bodyward = np.asarray(robot_position, dtype=float) - center
    bodyward -= box_up * np.dot(bodyward, box_up)
    bodyward = _unit(bodyward, -bundle_axis)
    return {
        'center_world': center,
        'bundle_axis_world': bundle_axis,
        'box_up_world': box_up,
        'side_axis_world': side_axis,
        'bodyward_world': bodyward,
        'diameter_m': diameter,
        'length_m': length,
    }


def _interaction(phase, object_id, targets, axes, settings, mode, reference):
    return {
        'format': FORMAT,
        'task': 'ramen_extract',
        'phase': phase,
        'object_id': object_id,
        'mode': mode,
        'tcp_targets': {
            side: {
                'position': np.asarray(targets[side][0], dtype=float).tolist(),
                'quaternion_xyzw': np.asarray(targets[side][1], dtype=float).tolist(),
            }
            for side in ('left', 'right')
        },
        'insertion_axes_world': {side: np.asarray(axes[side], dtype=float).tolist()
                                 for side in ('left', 'right')},
        'object_reference': {
            key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in reference.items()
        },
        'control': {
            'lateral_stiffness_n_per_m': float(settings['lateral_stiffness_n_per_m']),
            'insertion_stiffness_n_per_m': float(settings['insertion_stiffness_n_per_m']),
            'translation_damping_ns_per_m': float(settings['translation_damping_ns_per_m']),
            'orientation_stiffness_nm_per_rad': float(settings['orientation_stiffness_nm_per_rad']),
            'orientation_damping_nms_per_rad': float(settings['orientation_damping_nms_per_rad']),
            'maximum_feedback_torque_fraction': float(settings['maximum_feedback_torque_fraction']),
            'force_limit_n': float(settings['force_limit_n']),
        },
    }


def plan_ramen_sequence(robot: Robot, qpos, pins, angle_pins, item, settings):
    """Build an object-relative, incrementally lifted extraction from a taught tool attitude."""
    if robot.model_id != 'g1-tools':
        raise ValueError('라면 꺼내기 시퀀스는 G1 그리퍼 모델에서만 사용할 수 있습니다.')
    qpos = robot.validate_q(qpos)
    pins = list(dict.fromkeys(pins or FEET))
    angle_pins = list(dict.fromkeys(angle_pins or ()))
    if not isinstance(item, dict) or not isinstance(item.get('id'), str):
        raise ValueError('대상 라면 오브젝트가 필요합니다.')
    if item.get('fixed'):
        raise ValueError('고정된 오브젝트는 꺼내기 대상으로 사용할 수 없습니다.')
    center = _vector(item.get('position'), 'object position')
    clearance = float(settings['approach_clearance_m'])
    lift = float(settings['lift_height_m'])
    extraction = float(settings['extraction_distance_m'])
    cycles = int(settings['extraction_cycles'])
    carry_tilt = np.deg2rad(float(settings['carry_tilt_deg']))
    if (not .01 <= clearance <= .30 or not .02 <= lift <= .60
            or not .02 <= extraction <= .80 or not 2 <= cycles <= 6 or not 0 <= carry_tilt <= np.deg2rad(45)):
        raise ValueError('접근·인양·운반 설정이 지원 범위를 벗어났습니다.')

    taught = {side: _tcp_pose(robot, qpos, side) for side in ('left', 'right')}
    reference = _object_reference(item, taught, qpos[:3])
    diameter = reference['diameter_m']
    up, side_axis, bodyward = (reference[key] for key in ('box_up_world', 'side_axis_world', 'bodyward_world'))
    support_center = center - up * max(.005, diameter * .5 - .005)
    half_span = diameter * .5 + .005
    inserted = {}
    for side, sign in (('left', 1.), ('right', -1.)):
        position = support_center + sign * side_axis * half_span
        inserted[side] = (position, _align_tool_to_box(taught[side][1], up))
    vertical_travel = max(clearance, diameter)
    preinsert = {side: (inserted[side][0] + up * vertical_travel, inserted[side][1])
                 for side in ('left', 'right')}
    default = {side: _tcp_pose(robot, robot.home, side) for side in ('left', 'right')}
    insertion_axes = {side: -up for side in ('left', 'right')}
    lift = max(lift, diameter)
    tilt_axis = _unit(np.cross(up, bodyward), side_axis)

    phase_seconds = float(settings['phase_seconds'])
    insertion_seconds = float(settings['insertion_seconds'])
    hold_seconds = float(settings['hold_seconds'])
    phase_specs = [
        ('default_pose', '기본자세', default, .1, 'pose'),
        ('object_align', '물체 중심 위 정렬', preinsert, phase_seconds, 'pose'),
        ('left_insert', '왼 주걱 수직 삽입', {'left': inserted['left'], 'right': preinsert['right']}, insertion_seconds, 'insertion'),
        ('right_insert', '오른 받침 수직 삽입', inserted, insertion_seconds, 'insertion'),
        ('load_check', '접촉 하중 확인', inserted, hold_seconds, 'hold'),
    ]
    previous_position = {side: inserted[side][0].copy() for side in ('left', 'right')}
    for cycle in range(1, cycles + 1):
        fraction = cycle / cycles
        tilt = Rotation.from_rotvec(tilt_axis * carry_tilt * fraction)
        orientations = {side: (tilt * Rotation.from_quat(inserted[side][1])).as_quat()
                        for side in ('left', 'right')}
        angled = {side: (previous_position[side], orientations[side]) for side in ('left', 'right')}
        lifted = {side: (inserted[side][0] + up * lift * fraction + bodyward * extraction * (cycle - 1) / cycles,
                         orientations[side]) for side in ('left', 'right')}
        pulled = {side: (inserted[side][0] + up * lift * fraction + bodyward * extraction * fraction,
                         orientations[side]) for side in ('left', 'right')}
        phase_specs.extend([
            (f'angle_relax_{cycle}', f'{cycle}차 툴 각도 완화', angled, max(.1, phase_seconds * .3), 'carry'),
            (f'lift_{cycle}', f'{cycle}차 소폭 인양', lifted, max(.1, phase_seconds * .45), 'carry'),
            (f'pull_{cycle}', f'{cycle}차 몸쪽 꺼내기', pulled, max(.1, phase_seconds * .55), 'carry'),
        ])
        previous_position = {side: pulled[side][0].copy() for side in ('left', 'right')}
    carried = phase_specs[-1][2]
    phase_specs.append(('carry_hold', '몸쪽으로 꺾어 운반자세 유지', carried, hold_seconds, 'hold'))
    frames, diagnostics = [], []
    seed = robot.home.copy()
    for phase, label, targets, duration, mode in phase_specs:
        if phase == 'default_pose':
            solved = robot.home.copy()
            info = {'converged': True, 'target_error_mm': 0., 'angle_error_deg': 0.}
        else:
            solved, info = robot.solve(
                seed, seed, pins=pins, angle_pins=[], mode='free', resistance=0.,
                selected_targets={side + '_hand': targets[side][0] for side in ('left', 'right')},
                orientation_targets={side + '_hand': targets[side][1] for side in ('left', 'right')},
                posture_reference=robot.home, posture_weight=.3 if phase == 'carry_hold' else .12, max_nfev=180,
            )
        if not info['converged']:
            raise ValueError(f'{label} IK 실패: 위치 {info["target_error_mm"]:.1f} mm · 방향 {info["angle_error_deg"]:.1f}°')
        interaction = _interaction(phase, item['id'], targets, insertion_axes, settings, mode, reference)
        frames.append({'name': label, 'duration': duration, 'qpos': solved.tolist(),
                       'pins': deepcopy(pins), 'angle_pins': [],
                       'interaction': interaction})
        diagnostics.append({'phase': phase, 'label': label, 'target_error_mm': info['target_error_mm'],
                            'angle_error_deg': info['angle_error_deg']})
        seed = solved
    return {'format': 'motioncreator.ramen-sequence.v1', 'object_id': item['id'],
            'object_pose_source': 'scene', 'object_reference': frames[0]['interaction']['object_reference'],
            'keyframes': frames, 'diagnostics': diagnostics}
