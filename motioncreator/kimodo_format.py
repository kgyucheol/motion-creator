"""Kimodo G1 NPZ interchange (34-joint, y-up, +z-forward)."""
# Skeleton naming/order and coordinate convention follow NVIDIA Kimodo
# (Apache-2.0); conversion and validation are implemented locally with NumPy.
import numpy as np
from scipy.spatial.transform import Rotation


MUJOCO_TO_KIMODO = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])

# This is the public G1Skeleton34 order used by Kimodo. Four terminal joints do
# not exist as articulated bodies in the MuJoCo model and are added below.
KIMODO_BODY_NAMES = (
    'pelvis',
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', None,
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', None,
    'waist_yaw_link', 'waist_roll_link', 'torso_link',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link',
    'left_wrist_yaw_link', None,
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link',
    'right_wrist_yaw_link', None,
)
KIMODO_PARENTS = np.array([
    -1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32,
])
KIMODO_KEYS = ('posed_joints', 'global_rot_mats', 'local_rot_mats', 'root_positions', 'foot_contacts')
_TERMINAL_OFFSETS = {
    7: np.array([0.0, -0.035, 0.14]),
    14: np.array([0.0, -0.035, 0.14]),
    25: np.array([0.0, 0.0, 0.1]),
    33: np.array([0.0, 0.0, 0.1]),
}
_FOOT_INDICES = np.array([6, 7, 13, 14])


def export_kimodo_g1(robot, qpos, fps):
    """Convert MuJoCo qpos frames to the exact array contract of a Kimodo G1 NPZ."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != robot.model.nq:
        raise ValueError(f'Expected qpos shape [T, {robot.model.nq}]')
    nframes = len(qpos)
    positions = np.empty((nframes, 34, 3), dtype=np.float32)
    global_rotations = np.empty((nframes, 34, 3, 3), dtype=np.float32)
    c = MUJOCO_TO_KIMODO

    for frame, q in enumerate(qpos):
        data = robot.data(q)
        for index, body_name in enumerate(KIMODO_BODY_NAMES):
            if body_name is None:
                parent = KIMODO_PARENTS[index]
                positions[frame, index] = positions[frame, parent] + global_rotations[frame, parent] @ _TERMINAL_OFFSETS[index]
                global_rotations[frame, index] = global_rotations[frame, parent]
                continue
            body_id = robot.model.body(body_name).id
            positions[frame, index] = c @ data.xpos[body_id]
            body_rotation = data.xmat[body_id].reshape(3, 3)
            global_rotations[frame, index] = c @ body_rotation @ c.T

    local_rotations = np.empty_like(global_rotations)
    local_rotations[:, 0] = global_rotations[:, 0]
    for index in range(1, 34):
        parent_rotation = np.swapaxes(global_rotations[:, KIMODO_PARENTS[index]], -1, -2)
        local_rotations[:, index] = parent_rotation @ global_rotations[:, index]

    velocities = np.zeros_like(positions)
    if nframes > 1:
        velocities[:-1] = float(fps) * np.diff(positions, axis=0)
        velocities[-1] = velocities[-2]
    feet = positions[:, _FOOT_INDICES]
    foot_speeds = np.linalg.norm(velocities[:, _FOOT_INDICES], axis=-1)
    foot_contacts = (foot_speeds < 0.15) & (feet[..., 1] < 0.10)

    return {
        'posed_joints': np.ascontiguousarray(positions),
        'global_rot_mats': np.ascontiguousarray(global_rotations),
        'local_rot_mats': np.ascontiguousarray(local_rotations),
        'root_positions': np.ascontiguousarray(positions[:, 0]),
        'foot_contacts': np.ascontiguousarray(foot_contacts),
    }


def validate_kimodo_g1(arrays):
    """Validate the five-array Kimodo G1 contract without requiring Kimodo."""
    if set(arrays) != set(KIMODO_KEYS):
        raise ValueError('Kimodo G1 NPZ must contain exactly: ' + ', '.join(KIMODO_KEYS))
    nframes = len(arrays['root_positions'])
    if nframes < 1:
        raise ValueError('Kimodo G1 NPZ must contain at least one frame')
    expected = {
        'posed_joints': (nframes, 34, 3),
        'global_rot_mats': (nframes, 34, 3, 3),
        'local_rot_mats': (nframes, 34, 3, 3),
        'root_positions': (nframes, 3),
        'foot_contacts': (nframes, 4),
    }
    for key, shape in expected.items():
        value = np.asarray(arrays[key])
        if value.shape != shape:
            raise ValueError(f'Invalid Kimodo {key} shape; expected {shape}, got {value.shape}')
        if key != 'foot_contacts' and not np.issubdtype(value.dtype, np.number):
            raise ValueError(f'Kimodo {key} must be numeric')
        if key != 'foot_contacts' and not np.isfinite(value).all():
            raise ValueError(f'Kimodo {key} contains non-finite values')
    if np.asarray(arrays['foot_contacts']).dtype != np.bool_:
        raise ValueError('Kimodo foot_contacts must have bool dtype')
    if not np.array_equal(arrays['posed_joints'][:, 0], arrays['root_positions']):
        raise ValueError('Kimodo root_positions conflicts with posed_joints root')
    global_rotations = np.asarray(arrays['global_rot_mats'])
    local_rotations = np.asarray(arrays['local_rot_mats'])
    reconstructed = np.empty_like(global_rotations)
    reconstructed[:, 0] = global_rotations[:, 0]
    for index in range(1, 34):
        reconstructed[:, index] = (
            np.swapaxes(global_rotations[:, KIMODO_PARENTS[index]], -1, -2)
            @ global_rotations[:, index]
        )
    if not np.allclose(reconstructed, local_rotations, atol=2e-5, rtol=0):
        raise ValueError('Kimodo local and global rotations are inconsistent')
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(global_rotations, -1, -2) @ global_rotations, identity, atol=2e-4, rtol=0):
        raise ValueError('Kimodo global rotations are not orthonormal')
    if not np.allclose(np.linalg.det(global_rotations), 1, atol=2e-4, rtol=0):
        raise ValueError('Kimodo global rotations must be proper rotation matrices')
    return nframes


def kimodo_g1_to_qpos(robot, arrays):
    """Recover this project's MuJoCo qpos from a Kimodo G1 motion."""
    nframes = validate_kimodo_g1(arrays)
    c = MUJOCO_TO_KIMODO
    rotations = c.T @ np.asarray(arrays['global_rot_mats'], dtype=float) @ c
    qpos = np.tile(robot.model.qpos0, (nframes, 1))
    qpos[:, :3] = np.asarray(arrays['root_positions'], dtype=float) @ c
    qpos[:, 3:7] = Rotation.from_matrix(rotations[:, 0]).as_quat(scalar_first=True)
    for frame in range(1, nframes):
        if np.dot(qpos[frame - 1, 3:7], qpos[frame, 3:7]) < 0:
            qpos[frame, 3:7] *= -1

    body_to_index = {name: index for index, name in enumerate(KIMODO_BODY_NAMES) if name is not None}
    for joint_name in robot.names:
        joint_id = robot.model.joint(joint_name).id
        body_id = int(robot.model.jnt_bodyid[joint_id])
        body_name = robot.model.body(body_id).name
        parent_id = int(robot.model.body_parentid[body_id])
        child_rotation = rotations[:, body_to_index[body_name]]
        parent_name = robot.model.body(parent_id).name
        parent_rotation = rotations[:, body_to_index[parent_name]]
        rest_rotation = Rotation.from_quat(robot.model.body_quat[body_id], scalar_first=True).as_matrix()
        joint_rotation = rest_rotation.T @ np.swapaxes(parent_rotation, -1, -2) @ child_rotation
        angle = Rotation.from_matrix(joint_rotation).as_rotvec() @ robot.model.jnt_axis[joint_id]
        qpos[:, robot.model.jnt_qposadr[joint_id]] = angle
    return qpos
