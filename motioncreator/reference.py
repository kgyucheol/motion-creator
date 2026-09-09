"""Validated, pickle-free reference interchange. Also reads pre-v2 editor NPZs."""
import json
import numpy as np


def load_reference(path):
    with np.load(path, allow_pickle=False) as archive:
        required = ('time', 'qpos', 'dof_pos', 'joint_names', 'root_pos', 'root_quat_wxyz', 'metadata_json')
        if any(key not in archive for key in required):
            raise ValueError('Missing reference fields')
        data = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(data['metadata_json']))
    if meta.get('quaternion_order') != 'wxyz' or meta.get('units') != {'position': 'm', 'angle': 'rad', 'time': 's'}:
        raise ValueError('Expected m/rad/s and wxyz reference conventions')
    if meta.get('coordinate_system') != 'right-handed, +X forward, +Y left, +Z up':
        raise ValueError('Unsupported reference coordinate system')
    if meta.get('reference_schema', 'motioncreator.reference.v1') not in ('motioncreator.reference.v1', 'motioncreator.reference.v2'):
        raise ValueError('Unsupported reference schema')
    names = data['joint_names'].tolist()
    if not isinstance(names, list) or not names or any(not isinstance(k, str) for k in names) or len(set(names)) != len(names):
        raise ValueError('Joint names must be unique')
    time = data['time']
    if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or abs(time[0]) > 1e-8:
        raise ValueError('Invalid reference time')
    fps = float(meta.get('fps', 0))
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError('Invalid reference FPS')
    if len(time) > 1 and not np.allclose(np.diff(time), 1/fps, atol=1e-7, rtol=1e-6):
        raise ValueError('Reference time must match a uniform FPS grid')
    if 'fps' in data and float(data['fps']) != fps:
        raise ValueError('Conflicting reference FPS fields')
    n, j = len(time), len(names)
    for key, shape in [('qpos', (n, j+7)), ('dof_pos', (n, j)), ('root_pos', (n, 3)), ('root_quat_wxyz', (n, 4))]:
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key} shape or non-finite data')
    if not np.allclose(np.linalg.norm(data['root_quat_wxyz'], axis=-1), 1, atol=1e-5, rtol=0):
        raise ValueError('Root quaternion must be normalized')
    if not np.allclose(data['qpos'], np.concatenate([data['root_pos'], data['root_quat_wxyz'], data['dof_pos']], axis=-1), atol=1e-8, rtol=0):
        raise ValueError('qpos conflicts with named root/joint fields')
    if 'contacts' in data:
        if data['contacts'].shape != (n, 2) or not np.isin(data['contacts'], [0, 1]).all() or meta.get('contact_order') != ['left_foot', 'right_foot']:
            raise ValueError('Invalid authored support flags')
    return data, meta


def joint_permutation(source_names, target_names):
    if len(set(target_names)) != len(target_names) or set(source_names) != set(target_names):
        raise ValueError('Source and target joint names do not match exactly')
    return [source_names.index(key) for key in target_names]
