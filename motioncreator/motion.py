"""Versioned project files and contact-aware kinematic reference exports."""
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .robot import Robot, ROOT, FEET, HANDLES, BASIC_ROTATABLE as ROTATABLE, quat_matrix, matrix_quat

FORMAT = 'motioncreator.g1.v1'


def validate_project(robot: Robot, project):
    if project.get('format') != FORMAT or project.get('model_sha256') != robot.fingerprint:
        raise ValueError('Project format or G1 model fingerprint does not match')
    if project.get('joint_names') != robot.names:
        raise ValueError('Joint order does not match this model')
    if not isinstance(project.get('name'), str) or not isinstance(project.get('coordinate_system'), str) or not isinstance(project.get('units'), dict):
        raise ValueError('Project name, coordinate system and units are required')
    frames = project.get('keyframes', [])
    if not 1 <= len(frames) <= 100:
        raise ValueError('A project needs 1–100 keyframes')
    total = 0.
    for frame in frames:
        if not isinstance(frame.get('name'), str):
            raise ValueError('Keyframe name must be text')
        robot.validate_q(frame['qpos'])
        duration = float(frame.get('duration', 2.))
        if not np.isfinite(duration) or not .1 <= duration <= 60:
            raise ValueError('Keyframe durations must be 0.1–60 seconds')
        total += duration
        if any(k not in HANDLES for k in frame.get('pins', [])):
            raise ValueError('Unknown pinned handle')
    if total > 600:
        raise ValueError('Maximum project length is 600 seconds')
    if 'current_qpos' in project:
        robot.validate_q(project['current_qpos'])
    if any(k not in HANDLES for k in project.get('pins', [])):
        raise ValueError('Unknown pinned handle')
    if 'box' in project:
        box = project['box']
        for field in ('position', 'size'):
            value = np.asarray(box.get(field), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 5:
                raise ValueError(f'Box {field} must be three finite values within 5 m')
            if field == 'size' and np.min(value) < .01:
                raise ValueError('Box dimensions must be at least 0.01 m')
        if not isinstance(box.get('visible'), bool):
            raise ValueError('Box visibility must be a boolean')
    return project


def new_project(robot, name='G1 reference'):
    return {'format': FORMAT, 'name': name, 'model_sha256': robot.fingerprint,
            'joint_names': robot.names, 'coordinate_system': 'right-handed, +X forward, +Y left, +Z up',
            'units': {'position': 'm', 'angle': 'rad', 'time': 's'},
            'keyframes': [{'name': 'Stand', 'duration': 2., 'qpos': robot.home.tolist(), 'pins': list(FEET)}]}


def compile_motion(robot: Robot, project, fps=30):
    validate_project(robot, project)
    if not 1 <= fps <= 120:
        raise ValueError('FPS must be 1–120')
    frames = project['keyframes']
    poses, times, contacts = [robot.validate_q(frames[0]['qpos'])], [0.], [[k in frames[0].get('pins', []) for k in FEET]]
    pin_errors = []
    elapsed = 0.
    for a, b in zip(frames, frames[1:]):
        qa, qb = robot.validate_q(a['qpos']), robot.validate_q(b['qpos'])
        if np.dot(qa[3:7], poses[-1][3:7]) < 0:
            qa[3:7] *= -1
        if np.dot(qa[3:7], qb[3:7]) < 0:
            qb[3:7] *= -1
        da, db = robot.data(qa), robot.data(qb)
        pa = {k: robot.point(da, k) for k in HANDLES}
        pb = {k: robot.point(db, k) for k in HANDLES}
        pins = sorted(set(a.get('pins', [])) & set(b.get('pins', [])))
        for k in pins:
            if np.linalg.norm(pa[k][0] - pb[k][0]) > .004:
                raise ValueError(f'{HANDLES[k][2]} changes position while pinned. Unpin it in one endpoint before moving it.')
            if k in FEET and np.linalg.norm(pa[k][1] - pb[k][1]) > .02:
                raise ValueError(f'{HANDLES[k][2]} changes orientation while pinned')
        # Each destination frame's duration describes travel time from its predecessor.
        duration = float(b['duration'])
        count = max(1, round(duration * fps))
        duration = count / fps
        root_rotation = quat_matrix(qa[3:7])
        root_delta = Rotation.from_matrix(root_rotation.T @ quat_matrix(qb[3:7])).as_rotvec()
        angular_deltas = {k: Rotation.from_matrix(pa[k][1].T @ pb[k][1]).as_rotvec() for k in ROTATABLE}
        for i in range(1, count + 1):
            u = i / count
            s = u*u*u*(10 + u*(-15 + 6*u))  # quintic easing, zero endpoint velocity/acceleration
            q = (1-s)*qa + s*qb
            q[3:7] = matrix_quat(root_rotation @ Rotation.from_rotvec(s*root_delta).as_matrix())
            if np.dot(q[3:7], qa[3:7]) < 0:
                q[3:7] *= -1
            if i == count:
                q = qb.copy()
            if i != count and pins:
                rotations = {k: pa[k][1] @ Rotation.from_rotvec(s*angular_deltas[k]).as_matrix() for k in ROTATABLE}
                targets = {k: ((1-s)*pa[k][0] + s*pb[k][0], rotations.get(k, pa[k][1])) for k in HANDLES}
                # Root orientation follows SLERP exactly; only end-effector rotations need projection.
                orientations = {k: Rotation.from_matrix(rotations[k]).as_quat() for k in ROTATABLE
                                if k != 'pelvis' and not (k in FEET and k in pins) and np.linalg.norm(angular_deltas[k]) > 1e-5}
                q, info = robot.solve(q, qa, pins=pins, targets=targets, max_nfev=18,
                                     posture_reference=q, posture_weight=.8, orientation_targets=orientations)
                if info['rejected']:
                    raise ValueError('The transition cannot preserve its pins. Add intermediate keyframes or relax a pin.')
                pin_errors.append(info['pin_error_mm'])
            poses.append(q)
            times.append(elapsed + i / fps)
            contacts.append([k in pins for k in FEET])
        elapsed += duration
    qpos = np.array(poses)
    t = np.array(times)
    qvel = np.zeros((len(qpos), robot.model.nv))
    for i in range(len(qpos)):
        lo, hi = max(0, i-1), min(len(qpos)-1, i+1)
        if hi > lo:
            mujoco.mj_differentiatePos(robot.model, qvel[i], t[hi]-t[lo], qpos[lo], qpos[hi])
    qacc = np.gradient(qvel, t, axis=0) if len(t) > 1 else np.zeros_like(qvel)
    return {'time': t, 'qpos': qpos, 'qvel': qvel, 'qacc': qacc, 'contacts': np.array(contacts, dtype=bool),
            'fps': fps, 'max_pin_error_mm': max(pin_errors, default=0.)}


def save_bundle(robot: Robot, project, fps=30, directory=None, protomotions=False):
    motion = compile_motion(robot, project, fps)
    folder = Path(directory or ROOT / 'motions')
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^\w-]', '_', str(project.get('name', 'motion')), flags=re.UNICODE).strip('_')[:60] or 'motion'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    stem = f'{name}_{stamp}'
    handle_pos, handle_quat, body_pos, body_quat, com, floor = [], [], [], [], [], []
    body_vel, body_ang_vel = [], []
    jp, jr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
    for q, velocity in zip(motion['qpos'], motion['qvel']):
        d = robot.data(q)
        state = robot.state(q)
        handle_pos.append([state['handles'][k]['position'] for k in HANDLES])
        # Explicit wxyz conversion for all exported quaternions.
        handle_quat.append([[h['quaternion'][3], *h['quaternion'][:3]] for h in state['handles'].values()])
        body_pos.append(d.xpos[1:].copy())
        body_quat.append(d.xquat[1:].copy())
        linear, angular = [], []
        for body in range(1, robot.model.nbody):
            mujoco.mj_jacBody(robot.model, d, jp, jr, body)
            linear.append(jp @ velocity)
            angular.append(jr @ velocity)
        body_vel.append(linear)
        body_ang_vel.append(angular)
        com.append(state['com'])
        floor.append(state['floor_min_mm'])
    metadata = {'format': FORMAT, 'model_sha256': robot.fingerprint, 'joint_names': robot.names,
                'reference_schema': 'motioncreator.reference.v2',
                'body_pose_frame': 'world; body frame origins, not centers of mass',
                'body_velocity_frame': 'world; body frame origins, not centers of mass',
                'joint_position_semantics': 'absolute hinge angles in radians; not policy actions',
                'root_body': 'pelvis', 'root_joint': 'free, 6 unactuated DoF',
                'trajectory_type': 'kinematic reference', 'loop_mode': 'clamp; no implicit loop',
                'control_binding': None,
                'quaternion_order': 'wxyz', 'coordinate_system': project['coordinate_system'], 'units': project['units'],
                'root_angular_velocity_frame': 'local body frame (MuJoCo free joint)',
                'root_linear_velocity_frame': 'world', 'contact_order': list(FEET),
                'contact_semantics': 'authored support flags, not measured contact forces',
                'validation': {'kind': 'kinematic only', 'max_pin_error_mm': motion['max_pin_error_mm'],
                               'minimum_sole_height_mm': min(floor), 'max_joint_speed_rad_s': float(np.abs(motion['qvel'][:, 6:]).max()),
                               'self_collision_checked': False, 'dynamic_balance_checked': False},
                'interpolation': 'quintic easing, shortest-path root SLERP, orientation-aware IK for shared pins; finite-difference velocities',
                'fps': fps, 'samples': len(motion['time'])}
    npz_path = folder / (stem + '.npz')
    json_path = folder / (stem + '.json')
    metadata_path = folder / (stem + '.metadata.json')
    np.savez_compressed(npz_path, time=motion['time'], qpos=motion['qpos'], qvel=motion['qvel'], qacc=motion['qacc'],
                        fps=np.array(fps),
                        root_pos=motion['qpos'][:, :3], root_quat_wxyz=motion['qpos'][:, 3:7],
                        root_lin_vel_world=motion['qvel'][:, :3], root_ang_vel_world=np.array(body_ang_vel)[:, 0],
                        dof_pos=motion['qpos'][:, 7:], dof_vel=motion['qvel'][:, 6:],
                        joint_names=np.array(robot.names), contacts=motion['contacts'],
                        handle_names=np.array(list(HANDLES)), handle_pos=np.array(handle_pos), handle_quat_wxyz=np.array(handle_quat),
                        body_names=np.array([robot.model.body(i).name for i in range(1, robot.model.nbody)]),
                        body_pos=np.array(body_pos), body_quat_wxyz=np.array(body_quat),
                        body_lin_vel_world=np.array(body_vel), body_ang_vel_world=np.array(body_ang_vel),
                        body_parent_indices=robot.model.body_parentid[1:]-1, com=np.array(com),
                        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    json_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    result = {'files': [p.name for p in (json_path, npz_path, metadata_path)], 'directory': str(folder), 'metadata': metadata}
    if protomotions:
        from .protomotions_bridge import export_isolated
        try:
            result['files'].extend(export_isolated(npz_path))
        except (ValueError, OSError) as exc:
            # Native export failure must not hide the successfully saved editable reference.
            result['warnings'] = [f'JSON/NPZ 저장 완료. ProtoMotions 변환 실패: {exc}']
    return result
