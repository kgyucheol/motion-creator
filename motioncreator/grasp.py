"""Two-hand grasp fitting for authored keyframes and primitive scene objects."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import HANDLES, Robot
from .task_physics import build_scene
from .tasks import TaskSpec


GRASP_FORMAT = 'motioncreator.two-hand-grasp.v1'
HAND_KEYS = ('left_hand', 'right_hand')


def object_signature(item):
    payload = {key: item[key] for key in ('id', 'shape', 'position', 'quaternion_xyzw', 'size')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def lower_palm_wrist_anchors():
    """Return wrist-local contact anchors biased toward the lower palm/wrist pad.

    The full convex rubber-hand mesh remains the physics collider. These points only
    calibrate which part of that mesh is aligned to the authored box surface.
    """
    model, _ = build_scene(TaskSpec())
    anchors = {}
    for side in ('left', 'right'):
        geom_id = model.geom(f'{side}_grip').id
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start:start + count]
        rotation = Rotation.from_quat(model.geom_quat[geom_id][[1, 2, 3, 0]]).as_matrix()
        points = vertices @ rotation.T + model.geom_pos[geom_id]
        low, high = points.min(0), points.max(0)
        # A repeatable point near the wrist-side and lower portion of the inner pad.
        point = low + .35 * (high - low)
        point[1] = low[1] if side == 'left' else high[1]
        anchors[side] = point
    return anchors


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
        hand_rotation = box_rotation @ Rotation.from_euler('y', twist, degrees=True).as_matrix()
        wrist = desired_contact - hand_rotation @ anchors[side]
        key = f'{side}_hand'
        targets[key] = (wrist + hand_rotation @ np.asarray(HANDLES[key][1])).tolist()
        orientations[key] = Rotation.from_matrix(hand_rotation).as_quat().tolist()
        contacts[side] = surface.tolist()
    return targets, orientations, contacts


def fit_two_hand_grasp(robot: Robot, qpos, pins, item, event):
    """Fit the actual fake-hand contact anchors to opposite box faces.

    The authored keyframe becomes the zero-penetration contact pose. A second pose,
    stored on the interaction, provides the bounded inward reference used by the
    physics-only closure ramp.
    """
    source = robot.validate_q(qpos)
    target = validate_grasp_event(robot, event, [item])
    left_uv = _validate_uv(event['left_surface_uv'], 'Left hand')
    right_uv = _validate_uv(event['right_surface_uv'], 'Right hand')
    solve_pins = [key for key in pins if key not in HAND_KEYS]
    candidates = []
    for twist in (-30., -20., -10., 0., 10., 20., 30.):
        targets, orientations, contacts = _targets(target, left_uv, right_uv, 0., twist)
        pose, info = robot.solve(source, source, pins=solve_pins, resistance=1.5, mode='elastic',
                                selected_targets=targets, orientation_targets=orientations,
                                max_nfev=90, posture_reference=source, posture_weight=.08)
        delta = float(np.sqrt(np.mean((pose[7:] - source[7:]) ** 2)))
        score = info['target_error_mm'] + .2 * info.get('angle_error_deg', 0.) + 25 * delta
        if not info['rejected']:
            candidates.append((score, twist, pose, info, orientations, contacts))
    if not candidates:
        raise ValueError('현재 키프레임 자세에서 양손 파지면에 도달할 수 없습니다.')
    _, twist, contact_pose, contact_info, orientations, contacts = min(candidates, key=lambda value: value[0])
    if contact_info['target_error_mm'] > 15 or contact_info.get('angle_error_deg', 0.) > 12:
        raise ValueError('양손 파지 오차가 너무 큽니다. 로봇이나 상자를 더 가까이 배치하세요.')

    closed_targets, _, _ = _targets(target, left_uv, right_uv, float(event['inward_offset_m']), twist)
    closure_pose, closure_info = robot.solve(contact_pose, contact_pose, pins=solve_pins,
                                             resistance=1.5, mode='elastic',
                                             selected_targets=closed_targets, orientation_targets=orientations,
                                             max_nfev=90, posture_reference=contact_pose, posture_weight=.08)
    # Physics can command only the 29 joints. Keep the authored floating base fixed
    # and use the closure pose strictly as an actuated-joint target.
    closure_pose[:7] = contact_pose[:7]
    if closure_info['rejected'] or closure_info['target_error_mm'] > 15:
        raise ValueError('설정한 안쪽 오프셋까지 안전하게 닫을 수 없습니다. 값을 줄여주세요.')
    fitted = {**event, 'object_signature': object_signature(item),
              'contact_anchor': 'lower_palm_wrist', 'hand_twist_deg': twist,
              'contact_points_world': contacts, 'closure_qpos': closure_pose.tolist()}
    return {'state': robot.state(contact_pose), 'grasp': fitted,
            'solver': {'contact': contact_info, 'closure': closure_info}}
