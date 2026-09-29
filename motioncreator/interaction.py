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
    if value.get('stage_id') is not None and (not isinstance(value['stage_id'], str) or not value['stage_id']):
        raise ValueError('Interaction stage ID is invalid')
    if value.get('stage_label') is not None and (not isinstance(value['stage_label'], str) or not value['stage_label']):
        raise ValueError('Interaction stage label is invalid')
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
    perception = value.get('perception')
    if perception is not None:
        if perception.get('pose_source') not in ('scene_ground_truth', 'rgb_pose_estimator') or perception.get('frame_id') != 'world':
            raise ValueError('Interaction perception source or frame is invalid')
        camera = perception.get('camera', {})
        for key, length in (('mount_xyz_m', 3), ('mount_rpy_rad', 3),
                            ('world_position_m', 3), ('world_quaternion_xyzw', 4)):
            vector = np.asarray(camera.get(key), dtype=float)
            if vector.shape != (length,) or not np.isfinite(vector).all():
                raise ValueError(f'Interaction camera {key} is invalid')
        if abs(np.linalg.norm(camera['world_quaternion_xyzw']) - 1) > 1e-4:
            raise ValueError('Interaction camera quaternion must be normalized')
        object_pose = perception.get('object_pose_world', {})
        object_position = np.asarray(object_pose.get('position'), dtype=float)
        object_quaternion = np.asarray(object_pose.get('quaternion_xyzw'), dtype=float)
        if object_position.shape != (3,) or not np.isfinite(object_position).all():
            raise ValueError('Perceived object position is invalid')
        if (object_quaternion.shape != (4,) or not np.isfinite(object_quaternion).all()
                or abs(np.linalg.norm(object_quaternion) - 1) > 1e-4):
            raise ValueError('Perceived object quaternion is invalid')
        for collection in ('insertion_sites_world', 'approach_sites_world'):
            for side in SIDES:
                position = np.asarray(perception.get(collection, {}).get(side), dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError(f'Perception {collection} {side} is invalid')
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


def project_with_consistent_interactions(robot, project, position_tolerance_m=.03, orientation_tolerance_deg=15.):
    """Ignore legacy TCP targets that no longer describe their keyframe's joint pose.

    Manual edits can change qpos while retaining the old auto-generated metadata. Keep
    the saved project intact; only the playback copy loses those stale targets.
    """
    frames = project.get('keyframes', [])
    if not any(frame.get('interaction') for frame in frames):
        return project
    updated = []
    for frame in frames:
        interaction = frame.get('interaction')
        if not interaction:
            updated.append(frame)
            continue
        data = robot.data(frame['qpos'])
        consistent = True
        for side in SIDES:
            position, orientation = robot.point(data, f'{side}_hand')
            target = interaction['tcp_targets'][side]
            position_error = np.linalg.norm(position - np.asarray(target['position'], dtype=float))
            target_orientation = Rotation.from_quat(target['quaternion_xyzw']).as_matrix()
            orientation_error = np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(target_orientation @ orientation.T).as_rotvec()))
            if position_error > position_tolerance_m or orientation_error > orientation_tolerance_deg:
                consistent = False
                break
        updated.append(frame if consistent else {key: value for key, value in frame.items() if key != 'interaction'})
    return {**project, 'keyframes': updated}


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
