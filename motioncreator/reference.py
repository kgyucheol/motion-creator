"""Validated, pickle-free reference interchange. Also reads pre-v2 editor NPZs."""
import json
import numpy as np
import mujoco
from .kimodo_format import KIMODO_KEYS, kimodo_g1_to_qpos


def _load_kimodo_reference(path, data):
    from .robot import Robot
    robot = Robot()
    qpos = kimodo_g1_to_qpos(robot, data)
    metadata_path = path.with_suffix('.metadata.json')
    if not metadata_path.is_file() and path.name == 'motion.npz':
        metadata_path = path.parent / 'metadata.json'
    if metadata_path.is_file():
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    else:
        meta = {
            'format': 'motioncreator.g1.v1', 'reference_schema': 'motioncreator.reference.v2',
            'model_sha256': robot.fingerprint, 'joint_names': robot.names, 'fps': 30,
            'quaternion_order': 'wxyz',
            'coordinate_system': 'right-handed, +X forward, +Y left, +Z up',
            'units': {'position': 'm', 'angle': 'rad', 'time': 's'},
            'contact_order': ['left_foot', 'right_foot'],
            'npz_format': 'kimodo.g1.34',
            'npz_coordinate_system': 'right-handed, +Z forward, +Y up',
        }
    fps = float(meta.get('fps', 30))
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError('Invalid reference FPS')
    time = np.arange(len(qpos), dtype=float) / fps
    qvel = np.zeros((len(qpos), robot.model.nv))
    for index in range(len(qpos)):
        lo, hi = max(0, index - 1), min(len(qpos) - 1, index + 1)
        if hi > lo:
            mujoco.mj_differentiatePos(robot.model, qvel[index], time[hi] - time[lo], qpos[lo], qpos[hi])
    # Kimodo stores float32 rotation matrices. Remove only sub-microradian
    # differentiation noise introduced when a truly fixed hinge is recovered.
    qvel[np.abs(qvel) < 1e-6] = 0
    qacc = np.gradient(qvel, time, axis=0) if len(time) > 1 else np.zeros_like(qvel)
    contacts = np.column_stack((data['foot_contacts'][:, :2].any(axis=1), data['foot_contacts'][:, 2:].any(axis=1)))
    body_pos, body_quat, body_vel, body_ang_vel = [], [], [], []
    jp, jr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
    for q, velocity in zip(qpos, qvel):
        state = robot.data(q)
        positions, quaternions, linear, angular = [], [], [], []
        for body_id in range(1, robot.model.nbody):
            positions.append(state.xpos[body_id].copy())
            quaternions.append(state.xquat[body_id].copy())
            mujoco.mj_jacBody(robot.model, state, jp, jr, body_id)
            linear.append(jp @ velocity)
            angular.append(jr @ velocity)
        body_pos.append(positions)
        body_quat.append(quaternions)
        body_vel.append(linear)
        body_ang_vel.append(angular)
    body_pos = np.asarray(body_pos)
    body_quat = np.asarray(body_quat)
    body_vel = np.asarray(body_vel)
    body_ang_vel = np.asarray(body_ang_vel)
    result = {
        **data, 'time': time, 'qpos': qpos, 'qvel': qvel, 'qacc': qacc, 'fps': np.array(fps),
        'root_pos': qpos[:, :3].copy(), 'root_quat_wxyz': qpos[:, 3:7].copy(),
        'root_lin_vel_world': qvel[:, :3].copy(), 'root_ang_vel_world': body_ang_vel[:, 0].copy(),
        'dof_pos': qpos[:, 7:].copy(), 'dof_vel': qvel[:, 6:].copy(),
        'joint_names': np.array(robot.names), 'contacts': contacts,
        'body_names': np.array([robot.model.body(i).name for i in range(1, robot.model.nbody)]),
        'body_pos': body_pos, 'body_quat_wxyz': body_quat,
        'body_lin_vel_world': body_vel, 'body_ang_vel_world': body_ang_vel,
        'body_parent_indices': robot.model.body_parentid[1:] - 1,
        'metadata_json': np.array(json.dumps(meta, ensure_ascii=False)),
    }
    return result, meta


def load_reference(path):
    from pathlib import Path
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) == set(KIMODO_KEYS):
            data = {key: archive[key].copy() for key in archive.files}
            return _load_kimodo_reference(path, data)
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
