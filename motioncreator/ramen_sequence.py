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


def _interaction(phase, object_id, targets, axes, settings, mode):
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
    """Treat the authored pose as the fully inserted teach pose and plan around it."""
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
    carry = _vector(settings['carry_offset_m'], 'carry offset')
    if not .01 <= clearance <= .30 or not .02 <= lift <= .60 or np.linalg.norm(carry) > 1.:
        raise ValueError('접근·인양·운반 설정이 지원 범위를 벗어났습니다.')

    seated = {side: _tcp_pose(robot, qpos, side) for side in ('left', 'right')}
    axes = {}
    for side, fallback in (('left', np.array([0., 1., 0.])), ('right', np.array([0., -1., 0.]))):
        radial = seated[side][0] - center
        radial[2] = 0.
        axes[side] = radial / np.linalg.norm(radial) if np.linalg.norm(radial) > .02 else fallback
    approach = {side: (seated[side][0] + axes[side] * clearance, seated[side][1])
                for side in ('left', 'right')}
    lifted = {side: (seated[side][0] + np.array([0., 0., lift]), seated[side][1])
              for side in ('left', 'right')}
    carried = {side: (lifted[side][0] + carry, lifted[side][1]) for side in ('left', 'right')}

    phase_seconds = float(settings['phase_seconds'])
    insertion_seconds = float(settings['insertion_seconds'])
    hold_seconds = float(settings['hold_seconds'])
    phase_specs = [
        ('gap_approach', '틈 앞 대기', approach, .1, 'pose'),
        ('left_insert', '왼 주걱 천천히 삽입', {'left': seated['left'], 'right': approach['right']}, insertion_seconds, 'insertion'),
        ('right_insert', '오른 받침 천천히 삽입', seated, insertion_seconds, 'insertion'),
        ('support_hold', '양쪽 지지 안정화', seated, hold_seconds, 'hold'),
        ('vertical_lift', '라면 묶음 수직 인양', lifted, phase_seconds, 'carry'),
        ('extract', '상자 밖으로 꺼내기', carried, phase_seconds, 'carry'),
        ('carry_hold', '운반 자세 유지', carried, hold_seconds, 'hold'),
    ]
    frames, diagnostics = [], []
    seed = qpos.copy()
    for phase, label, targets, duration, mode in phase_specs:
        solved, info = robot.solve(
            seed, seed, pins=pins, angle_pins=angle_pins, mode='free', resistance=0.,
            selected_targets={side + '_hand': targets[side][0] for side in ('left', 'right')},
            orientation_targets={side + '_hand': targets[side][1] for side in ('left', 'right')},
            posture_reference=seed, posture_weight=.12, max_nfev=120,
        )
        if not info['converged']:
            raise ValueError(f'{label} IK 실패: 위치 {info["target_error_mm"]:.1f} mm · 방향 {info["angle_error_deg"]:.1f}°')
        interaction = _interaction(phase, item['id'], targets, axes, settings, mode)
        frames.append({'name': label, 'duration': duration, 'qpos': solved.tolist(),
                       'pins': deepcopy(pins), 'angle_pins': deepcopy(angle_pins),
                       'interaction': interaction})
        diagnostics.append({'phase': phase, 'label': label, 'target_error_mm': info['target_error_mm'],
                            'angle_error_deg': info['angle_error_deg']})
        seed = solved
    return {'format': 'motioncreator.ramen-sequence.v1', 'object_id': item['id'],
            'object_pose_source': 'scene', 'keyframes': frames, 'diagnostics': diagnostics}
