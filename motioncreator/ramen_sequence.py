"""Object-relative two-tool sequence generation for extracting a ramen bundle."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from .robot import FEET, Robot
from .tool_model import head_camera_spec


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


def _camera_observation(robot, qpos):
    specification = head_camera_spec()
    data = robot.data(qpos)
    try:
        body = robot.model.body(specification['link']).id
        position = data.xpos[body].copy()
        rotation = data.xmat[body].reshape(3, 3).copy()
    except KeyError:
        body = robot.model.body(specification['parent_link']).id
        parent_rotation = data.xmat[body].reshape(3, 3)
        position = data.xpos[body] + parent_rotation @ np.asarray(specification['mount_xyz_m'])
        rotation = parent_rotation @ Rotation.from_euler('xyz', specification['mount_rpy_rad']).as_matrix()
    return {**specification, 'world_position_m': position.tolist(),
            'world_quaternion_xyzw': Rotation.from_matrix(rotation).as_quat().tolist()}


def _locate_insertion_sites(robot, qpos, item, settings, taught):
    """Resolve the scene pose into two tool corridors.

    `scene_ground_truth` is intentionally isolated here so a future RGB pose
    provider can supply the same object pose contract without changing the
    motion-generation stages.
    """
    reference = _object_reference(item, taught, qpos[:3])
    center = reference['center_world']
    diameter = reference['diameter_m']
    up, side_axis = reference['box_up_world'], reference['side_axis_world']
    support_center = center - up * max(.005, diameter * .5 - .005)
    half_span = diameter * .5 + .005
    inserted = {}
    for side, sign in (('left', 1.), ('right', -1.)):
        position = support_center + sign * side_axis * half_span
        inserted[side] = (position, _align_tool_to_box(taught[side][1], up))
    vertical_travel = max(float(settings['approach_clearance_m']), diameter)
    approach = {side: (inserted[side][0] + up * vertical_travel, inserted[side][1])
                for side in ('left', 'right')}
    perception = {
        'pose_source': 'scene_ground_truth',
        'frame_id': 'world',
        'camera': _camera_observation(robot, qpos),
        'object_pose_world': {
            'position': _vector(item.get('position'), 'object position').tolist(),
            'quaternion_xyzw': _vector(item.get('quaternion_xyzw'), 'object quaternion', length=4).tolist(),
        },
        'insertion_sites_world': {side: inserted[side][0].tolist() for side in ('left', 'right')},
        'approach_sites_world': {side: approach[side][0].tolist() for side in ('left', 'right')},
        'approach_strategy': 'staggered_vertical_corridor',
    }
    return reference, inserted, approach, perception


def _interaction(phase, stage_id, stage_label, object_id, targets, axes, settings, mode, reference,
                 perception=None):
    result = {
        'format': FORMAT,
        'task': 'ramen_extract',
        'phase': phase,
        'stage_id': stage_id,
        'stage_label': stage_label,
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
    if perception is not None:
        result['perception'] = perception
    return result


def plan_ramen_sequence(robot: Robot, qpos, pins, angle_pins, item, settings, attention_pose=None):
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
    clearance = float(settings['approach_clearance_m'])
    lift = float(settings['lift_height_m'])
    extraction = float(settings['extraction_distance_m'])
    cycles = int(settings['extraction_cycles'])
    carry_tilt = np.deg2rad(float(settings['carry_tilt_deg']))
    if (not .01 <= clearance <= .30 or not .02 <= lift <= .60
            or not .02 <= extraction <= .80 or not 2 <= cycles <= 6 or not 0 <= carry_tilt <= np.deg2rad(45)):
        raise ValueError('접근·인양·운반 설정이 지원 범위를 벗어났습니다.')

    taught = {side: _tcp_pose(robot, qpos, side) for side in ('left', 'right')}
    start_qpos = robot.validate_q(attention_pose['qpos']) if attention_pose else robot.home.copy()
    start_pins = list(dict.fromkeys(attention_pose.get('pins', FEET))) if attention_pose else list(FEET)
    start_angle_pins = list(dict.fromkeys(attention_pose.get('angle_pins', ()))) if attention_pose else []
    reference, inserted, preinsert, perception = _locate_insertion_sites(
        robot, start_qpos, item, settings, taught)
    diameter = reference['diameter_m']
    up, side_axis, bodyward = (reference[key] for key in ('box_up_world', 'side_axis_world', 'bodyward_world'))
    attention_targets = {side: _tcp_pose(robot, start_qpos, side) for side in ('left', 'right')}
    insertion_axes = {side: -up for side in ('left', 'right')}
    lift = max(lift, diameter)
    tilt_axis = _unit(np.cross(up, bodyward), side_axis)

    phase_seconds = float(settings['phase_seconds'])
    insertion_seconds = float(settings['insertion_seconds'])
    hold_seconds = float(settings['hold_seconds'])
    phase_specs = [
        ('attention_pose', '차렷자세', attention_targets, .1, 'pose',
         'attention', '차렷자세', None, start_pins, start_angle_pins),
        ('insertion_site_search', '삽입 위치 탐색', attention_targets, max(.1, phase_seconds * .5), 'pose',
         'site_search', '삽입 위치 탐색', perception, pins, []),
        ('approach_clearance', '무충돌 접근 · 상부 안전점', preinsert, phase_seconds, 'pose',
         'collision_free_approach', '양손을 충돌없이 각 삽입지점으로 이동', None, pins, []),
        ('left_insert', '무충돌 접근 · 왼 주걱 삽입',
         {'left': inserted['left'], 'right': preinsert['right']}, insertion_seconds, 'insertion',
         'collision_free_approach', '양손을 충돌없이 각 삽입지점으로 이동', None, pins, []),
        ('right_insert', '무충돌 접근 · 오른 받침 삽입', inserted, insertion_seconds, 'insertion',
         'collision_free_approach', '양손을 충돌없이 각 삽입지점으로 이동', None, pins, []),
        ('bilateral_stabilize', '양손 지지 안정화', inserted, hold_seconds, 'hold',
         'bilateral_stabilize', '양손 지지 안정화', None, pins, []),
    ]
    lifted = {side: (inserted[side][0] + up * lift, inserted[side][1]) for side in ('left', 'right')}
    phase_specs.append(('vertical_lift', '라면묶음 수직 인양', lifted, phase_seconds, 'carry',
                        'vertical_lift', '라면묶음 수직 인양', None, pins, []))
    previous_position = {side: lifted[side][0].copy() for side in ('left', 'right')}
    for cycle in range(1, cycles + 1):
        fraction = cycle / cycles
        tilt = Rotation.from_rotvec(tilt_axis * carry_tilt * fraction)
        orientations = {side: (tilt * Rotation.from_quat(inserted[side][1])).as_quat()
                        for side in ('left', 'right')}
        angled = {side: (previous_position[side], orientations[side]) for side in ('left', 'right')}
        pulled = {side: (inserted[side][0] + up * lift + bodyward * extraction * fraction,
                         orientations[side]) for side in ('left', 'right')}
        phase_specs.extend([
            (f'extract_angle_{cycle}', f'상자 밖으로 꺼내기 · {cycle}차 각도 완화', angled,
             max(.1, phase_seconds * .3), 'carry', 'extract', '상자 밖으로 꺼내기', None, pins, []),
            (f'extract_pull_{cycle}', f'상자 밖으로 꺼내기 · {cycle}차 몸쪽 이동', pulled,
             max(.1, phase_seconds * .55), 'carry', 'extract', '상자 밖으로 꺼내기', None, pins, []),
        ])
        previous_position = {side: pulled[side][0].copy() for side in ('left', 'right')}
    carried = phase_specs[-1][2]
    phase_specs.append(('carry_hold', '운반 자세 유지', carried, hold_seconds, 'hold',
                        'carry_hold', '운반 자세 유지', None, pins, []))
    frames, diagnostics = [], []
    seed = start_qpos.copy()
    for phase, label, targets, duration, mode, stage_id, stage_label, phase_perception, frame_pins, frame_angle_pins in phase_specs:
        if phase in ('attention_pose', 'insertion_site_search'):
            solved = start_qpos.copy()
            info = {'converged': True, 'target_error_mm': 0., 'angle_error_deg': 0.}
        else:
            solved, info = robot.solve(
                seed, seed, pins=frame_pins, angle_pins=[], mode='free', resistance=0.,
                selected_targets={side + '_hand': targets[side][0] for side in ('left', 'right')},
                orientation_targets={side + '_hand': targets[side][1] for side in ('left', 'right')},
                posture_reference=start_qpos, posture_weight=.3 if phase == 'carry_hold' else .12, max_nfev=180,
            )
        if not info['converged']:
            raise ValueError(f'{label} IK 실패: 위치 {info["target_error_mm"]:.1f} mm · 방향 {info["angle_error_deg"]:.1f}°')
        interaction = _interaction(phase, stage_id, stage_label, item['id'], targets, insertion_axes,
                                   settings, mode, reference, phase_perception)
        frames.append({'name': label, 'duration': duration, 'qpos': solved.tolist(),
                       'pins': deepcopy(frame_pins), 'angle_pins': deepcopy(frame_angle_pins),
                       'interaction': interaction})
        diagnostics.append({'phase': phase, 'stage_id': stage_id, 'label': label,
                            'target_error_mm': info['target_error_mm'],
                            'angle_error_deg': info['angle_error_deg']})
        seed = solved
    return {'format': 'motioncreator.ramen-sequence.v1', 'object_id': item['id'],
            'object_pose_source': 'scene_ground_truth', 'object_reference': frames[0]['interaction']['object_reference'],
            'perception': perception, 'keyframes': frames, 'diagnostics': diagnostics}
