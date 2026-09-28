"""Validation and interpolation for object-interaction TCP references."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .ramen_sequence import FORMAT


SIDES = ('left', 'right')
MODES = {'pose', 'insertion', 'hold', 'carry'}


def validate_interaction(value, object_ids=()):
    if not isinstance(value, dict) or value.get('format') != FORMAT or value.get('task') != 'ramen_extract':
        raise ValueError('Interaction metadata format is invalid')
    if value.get('mode') not in MODES or not isinstance(value.get('phase'), str):
        raise ValueError('Interaction phase or mode is invalid')
    if object_ids and value.get('object_id') not in object_ids:
        raise ValueError('Interaction target object does not exist')
    for side in SIDES:
        target = value.get('tcp_targets', {}).get(side, {})
        position = np.asarray(target.get('position'), dtype=float)
        quaternion = np.asarray(target.get('quaternion_xyzw'), dtype=float)
        axis = np.asarray(value.get('insertion_axes_world', {}).get(side), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or np.max(np.abs(position)) > 5:
            raise ValueError('Interaction TCP position is invalid')
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-4:
            raise ValueError('Interaction TCP orientation is invalid')
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > 1e-4:
            raise ValueError('Interaction insertion axis is invalid')
    reference = value.get('object_reference')
    if reference is not None:
        for key in ('center_world', 'bundle_axis_world', 'box_up_world', 'side_axis_world', 'bodyward_world'):
            vector = np.asarray(reference.get(key), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f'Interaction object reference {key} is invalid')
            if key != 'center_world' and abs(np.linalg.norm(vector) - 1) > 1e-4:
                raise ValueError(f'Interaction object reference {key} must be normalized')
        for key in ('diameter_m', 'length_m'):
            number = reference.get(key)
            if not isinstance(number, (int, float)) or isinstance(number, bool) or not np.isfinite(number) or not 0 < number <= 5:
                raise ValueError(f'Interaction object reference {key} is invalid')
    control = value.get('control')
    required = {
        'lateral_stiffness_n_per_m': (0., 3000.),
        'insertion_stiffness_n_per_m': (0., 1000.),
        'translation_damping_ns_per_m': (0., 300.),
        'orientation_stiffness_nm_per_rad': (0., 300.),
        'orientation_damping_nms_per_rad': (0., 50.),
        'maximum_feedback_torque_fraction': (.01, 1.),
        'force_limit_n': (.1, 1000.),
    }
    if not isinstance(control, dict):
        raise ValueError('Interaction control settings are required')
    for key, limits in required.items():
        number = control.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not np.isfinite(number) or not limits[0] <= number <= limits[1]:
            raise ValueError(f'Interaction control setting {key} is invalid')
    return value


def interpolate_interaction(first, second, progress):
    """Interpolate authored TCP references while using destination control semantics."""
    if not first or not second:
        return None
    progress = float(np.clip(progress, 0., 1.))
    output = {**second, 'tcp_targets': {}}
    for side in SIDES:
        a = first['tcp_targets'][side]
        b = second['tcp_targets'][side]
        pa, pb = np.asarray(a['position']), np.asarray(b['position'])
        rotations = Rotation.from_quat([a['quaternion_xyzw'], b['quaternion_xyzw']])
        quaternion = Slerp([0., 1.], rotations)([progress]).as_quat()[0]
        output['tcp_targets'][side] = {
            'position': ((1 - progress) * pa + progress * pb).tolist(),
            'quaternion_xyzw': quaternion.tolist(),
        }
    return output


def interaction_timeline(project):
    frames = project.get('keyframes', [])
    elapsed = 0.
    result = []
    for first, second in zip(frames, frames[1:]):
        duration = float(second['duration'])
        result.append((elapsed, elapsed + duration, first.get('interaction'), second.get('interaction')))
        elapsed += duration
    return result


def interaction_at(timeline, time):
    if not timeline:
        return None
    for start, end, first, second in timeline:
        if time <= end + 1e-9:
            if not first or not second:
                return None
            width = max(1e-9, end - start)
            u = np.clip((time - start) / width, 0., 1.)
            smooth = u * u * u * (10 + u * (-15 + 6 * u))
            return interpolate_interaction(first, second, smooth)
    return timeline[-1][3]
