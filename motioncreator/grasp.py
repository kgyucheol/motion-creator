"""Two-hand grasp fitting for authored keyframes and primitive scene objects."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .grip_geometry import GRIP_PAD_FORMAT, grip_pad_contact_anchor, grip_pad_rotation
from .robot import HANDLES, Robot


GRASP_FORMAT = 'motioncreator.two-hand-grasp.v1'
HAND_KEYS = ('left_hand', 'right_hand')


def object_signature(item):
    payload = {key: item[key] for key in ('id', 'shape', 'position', 'quaternion_xyzw', 'size')}
    payload['grip_geometry'] = GRIP_PAD_FORMAT
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def lower_palm_wrist_anchors():
    """Return the center of each finger-tip-to-wrist contact plane in wrist coordinates."""
    return {side: grip_pad_contact_anchor(side) for side in ('left', 'right')}


def _validate_uv(value, label):
    uv = np.asarray(value, dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1:
        raise ValueError(f'{label} surface coordinates must be two values from -1 to 1')
    return uv


def validate_grasp_event(robot: Robot, event, objects):
    if not isinstance(event, dict) or event.get('format') != GRASP_FORMAT:
        raise ValueError('Unknown grasp interaction format')
    object_id = event.get('object_id')
    target = next((item for item in objects if item.get('id') == object_id), None)
    if target is None or target.get('shape') != 'box':
        raise ValueError('Two-hand grasp interactions require an existing box object')
    _validate_uv(event.get('left_surface_uv'), 'Left hand')
    _validate_uv(event.get('right_surface_uv'), 'Right hand')
    inset = float(event.get('inward_offset_m', 0.))
    duration = float(event.get('closure_seconds', 0.))
    target_force = float(event.get('target_force_n', 0.))
    max_force = float(event.get('max_force_n', 0.))
    if not np.isfinite(inset) or not 0 <= inset <= min(.06, float(target['size'][1]) - .03):
        raise ValueError('Grasp inward offset must be 0–0.06 m and leave at least 0.03 m between targets')
    if not np.isfinite(duration) or not .1 <= duration <= 2:
        raise ValueError('Grasp closure duration must be 0.1–2 seconds')
    if not np.isfinite(target_force) or not 1 <= target_force <= 200:
        raise ValueError('Grasp target force must be 1–200 N per hand')
    if not np.isfinite(max_force) or not target_force <= max_force <= 400:
        raise ValueError('Grasp maximum force must be at least the target and at most 400 N')
    if 'closure_qpos' in event:
        robot.validate_q(event['closure_qpos'])
    signature = event.get('object_signature')
    if signature is not None and (not isinstance(signature, str) or len(signature) != 64):
        raise ValueError('Invalid grasp object signature')
    return target


def _box_contact(item, uv, sign):
    center = np.asarray(item['position'], dtype=float)
    size = np.asarray(item['size'], dtype=float)
    rotation = Rotation.from_quat(item['quaternion_xyzw']).as_matrix()
    local = np.array([uv[0] * size[0] / 2, sign * size[1] / 2, uv[1] * size[2] / 2])
    return center + rotation @ local, rotation @ np.array([0., float(sign), 0.])


def _targets(item, left_uv, right_uv, inward_offset, twist_degrees):
    box_rotation = Rotation.from_quat(item['quaternion_xyzw']).as_matrix()
    anchors = lower_palm_wrist_anchors()
    targets, orientations, contacts = {}, {}, {}
    for side, uv, sign in (('left', left_uv, 1), ('right', right_uv, -1)):
        surface, outward = _box_contact(item, uv, sign)
        desired_contact = surface - outward * inward_offset / 2
        twist = twist_degrees if side == 'left' else -twist_degrees
        pad_rotation = box_rotation @ Rotation.from_euler('y', twist, degrees=True).as_matrix()
        # The pad is angled in the wrist frame so its proximal edge intersects
        # the wrist-yaw link and its distal edge intersects all four fingertips.
        hand_rotation = pad_rotation @ grip_pad_rotation(side).T
        wrist = desired_contact - hand_rotation @ anchors[side]
        key = f'{side}_hand'
        targets[key] = (wrist + hand_rotation @ np.asarray(HANDLES[key][1])).tolist()
        orientations[key] = Rotation.from_matrix(hand_rotation).as_quat().tolist()
        contacts[side] = surface.tolist()
    return targets, orientations, contacts


def fit_two_hand_grasp(robot: Robot, qpos, pins, item, event, angle_pins=()):
    """Fit parallel hand-pad poses to opposite box faces with an authored offset.

    The inward offset is part of the keyframe pose itself.  No hidden physics-only
    closure pose is generated, so editor, preview and WBC playback share one pose.
    """
    source = robot.validate_q(qpos)
    target = validate_grasp_event(robot, event, [item])
    left_uv = _validate_uv(event['left_surface_uv'], 'Left hand')
    right_uv = _validate_uv(event['right_surface_uv'], 'Right hand')
    solve_pins = [key for key in pins if key not in HAND_KEYS]
    solve_angle_pins = [key for key in angle_pins if key not in HAND_KEYS]
    inward_offset = float(event['inward_offset_m'])
    candidates = []
    for twist in (-30., -20., -10., 0., 10., 20., 30.):
        targets, orientations, contacts = _targets(target, left_uv, right_uv, inward_offset, twist)
        pose, info = robot.solve(source, source, pins=solve_pins, resistance=1.5, mode='elastic',
                                selected_targets=targets, orientation_targets=orientations,
                                angle_pins=solve_angle_pins, max_nfev=90,
                                posture_reference=source, posture_weight=.08)
        delta = float(np.sqrt(np.mean((pose[7:] - source[7:]) ** 2)))
        score = info['target_error_mm'] + .2 * info.get('angle_error_deg', 0.) + 25 * delta
        if not info['rejected']:
            candidates.append((score, twist, pose, info, orientations, contacts))
    if not candidates:
        raise ValueError('현재 키프레임 자세에서 양손 파지면에 도달할 수 없습니다.')
    _, twist, contact_pose, contact_info, orientations, contacts = min(candidates, key=lambda value: value[0])
    if contact_info['target_error_mm'] > 15 or contact_info.get('angle_error_deg', 0.) > 12:
        raise ValueError('양손 파지 오차가 너무 큽니다. 로봇이나 상자를 더 가까이 배치하세요.')

    fitted = {key: value for key, value in event.items() if key != 'closure_qpos'}
    fitted.update({'object_signature': object_signature(item),
              'contact_anchor': 'finger_wrist_pad', 'hand_twist_deg': twist,
              'contact_points_world': contacts})
    return {'state': robot.state(contact_pose), 'grasp': fitted,
            'solver': {'contact': contact_info}}
