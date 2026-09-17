"""Versioned project files and contact-aware kinematic reference exports."""
import hashlib
import io
import json
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from .robot import Robot, ROOT, FEET, HANDLES, BASIC_ROTATABLE as ROTATABLE, quat_matrix, matrix_quat
from .kimodo_format import KIMODO_KEYS, export_kimodo_g1, kimodo_g1_to_qpos

FORMAT = 'motioncreator.g1.v1'
ENVIRONMENT_FORMAT = 'motioncreator.environment.v1'
MAX_KEYFRAMES = 100
MAX_MOTION_SAMPLES = 3000
SCENE_SHAPES = {'box', 'sphere', 'cylinder'}


def project_scene_objects(project):
    """Return authored primitive objects, including the legacy single-box format."""
    if 'scene_objects' in project:
        return project['scene_objects']
    box = project.get('box')
    if not box:
        return []
    return [{'id': 'legacy-box', 'name': 'Box', 'shape': 'box',
             'position': box['position'], 'quaternion_xyzw': [0., 0., 0., 1.],
             'size': box['size'], 'mass_kg': 1., 'friction': .7,
             'color': '#b98853', 'opacity': .62, 'visible': box['visible']}]


def environment_snapshot(project):
    """Return the portable environment stored beside an exported motion."""
    return {
        'format': ENVIRONMENT_FORMAT,
        'coordinate_system': project['coordinate_system'],
        'units': project['units'],
        'physics': {
            'gravity_m_s2': [0., 0., -9.81],
            'floor': {
                'height_m': 0.,
                'friction': [1., .005, .0001],
                'friction_semantics': ['sliding', 'torsional', 'rolling'],
            },
        },
        'scene_objects': project_scene_objects(project),
    }


def validate_project(robot: Robot, project):
    if project.get('format') != FORMAT or project.get('model_sha256') != robot.fingerprint:
        raise ValueError('Project format or G1 model fingerprint does not match')
    if project.get('joint_names') != robot.names:
        raise ValueError('Joint order does not match this model')
    if not isinstance(project.get('name'), str) or not isinstance(project.get('coordinate_system'), str) or not isinstance(project.get('units'), dict):
        raise ValueError('Project name, coordinate system and units are required')
    frames = project.get('keyframes', [])
    if not 1 <= len(frames) <= MAX_KEYFRAMES:
        raise ValueError(f'A project needs 1–{MAX_KEYFRAMES} keyframes')
    if any(frame.get('samples') is not None for frame in frames) and not (len(frames) == 1 and frames[0].get('samples') is not None):
        raise ValueError('An imported motion clip must be the project\'s only keyframe')
    total = 0.
    for frame in frames:
        if not isinstance(frame.get('name'), str):
            raise ValueError('Keyframe name must be text')
        robot.validate_q(frame['qpos'])
        duration = float(frame.get('duration', 2.))
        minimum_duration = 1 / 120 if frame.get('samples') is not None else .1
        maximum_duration = 600 if frame.get('samples') is not None else 60
        if not np.isfinite(duration) or not minimum_duration <= duration <= maximum_duration:
            raise ValueError(f'Keyframe durations must be {minimum_duration:g}–{maximum_duration} seconds')
        if frame.get('samples') is not None:
            samples = np.asarray(frame['samples'], dtype=float)
            if samples.ndim != 2 or samples.shape[1] != robot.model.nq:
                raise ValueError(f'Motion clip samples must have shape [T, {robot.model.nq}]')
            if not 1 <= len(samples) <= MAX_MOTION_SAMPLES:
                raise ValueError(f'Motion clip must contain 1–{MAX_MOTION_SAMPLES} samples')
            validated_samples = np.stack([robot.validate_q(sample) for sample in samples])
            if not np.allclose(validated_samples[0], frame['qpos'], atol=1e-9, rtol=0):
                raise ValueError('Motion clip qpos must match its first sample')
            if frame.get('pins'):
                raise ValueError('Motion clips cannot use pins until they are split into keyframes')
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
    objects = project.get('scene_objects', [])
    if not isinstance(objects, list) or len(objects) > 32:
        raise ValueError('Scene objects must be a list with at most 32 entries')
    identifiers = set()
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError('Each scene object must be an object')
        identifier = item.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier):
            raise ValueError('Scene object IDs must use 1–64 letters, numbers, underscores or hyphens')
        if identifier in identifiers:
            raise ValueError('Scene object IDs must be unique')
        identifiers.add(identifier)
        if not isinstance(item.get('name'), str) or not 1 <= len(item['name']) <= 80:
            raise ValueError('Scene object names must contain 1–80 characters')
        shape = item.get('shape')
        if shape not in SCENE_SHAPES:
            raise ValueError('Scene object shape must be box, sphere or cylinder')
        for field, length in (('position', 3), ('quaternion_xyzw', 4), ('size', 3)):
            value = np.asarray(item.get(field), dtype=float)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f'Scene object {field} must contain {length} finite numbers')
            if field == 'position' and np.max(np.abs(value)) > 5:
                raise ValueError('Scene object positions must be within 5 m')
            if field == 'size' and (np.min(value) < .01 or np.max(value) > 5):
                raise ValueError('Scene object sizes must be 0.01–5 m')
        quaternion = np.asarray(item['quaternion_xyzw'], dtype=float)
        if abs(np.linalg.norm(quaternion) - 1.) > 1e-4:
            raise ValueError('Scene object quaternions must be normalized (xyzw)')
        size = np.asarray(item['size'], dtype=float)
        if shape == 'sphere' and not np.allclose(size, size[0], atol=1e-6, rtol=0):
            raise ValueError('Sphere size must use one uniform diameter')
        if shape == 'cylinder' and abs(size[0] - size[1]) > 1e-6:
            raise ValueError('Cylinder X/Y sizes must use one diameter')
        mass = float(item.get('mass_kg', 1.))
        friction = float(item.get('friction', .7))
        opacity = float(item.get('opacity', 1.))
        if not np.isfinite(mass) or not .001 <= mass <= 1000:
            raise ValueError('Scene object mass must be 0.001–1000 kg')
        if not np.isfinite(friction) or not 0 <= friction <= 2:
            raise ValueError('Scene object friction must be 0–2')
        if not np.isfinite(opacity) or not .05 <= opacity <= 1:
            raise ValueError('Scene object opacity must be 0.05–1')
        if not isinstance(item.get('color'), str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', item['color']):
            raise ValueError('Scene object color must use #RRGGBB')
        if not isinstance(item.get('visible'), bool):
            raise ValueError('Scene object visibility must be a boolean')
    from .grasp import validate_grasp_event
    for frame in frames:
        grasp = frame.get('grasp')
        if grasp is not None:
            if frame.get('samples') is not None:
                raise ValueError('Grasp interactions require editable keyframes, not an imported motion clip')
            validate_grasp_event(robot, grasp, objects)
    return project


def new_project(robot, name='G1 reference'):
    return {'format': FORMAT, 'name': name, 'model_sha256': robot.fingerprint,
            'joint_names': robot.names, 'coordinate_system': 'right-handed, +X forward, +Y left, +Z up',
            'units': {'position': 'm', 'angle': 'rad', 'time': 's'},
            'keyframes': [{'name': 'Stand', 'duration': 2., 'qpos': robot.home.tolist(), 'pins': list(FEET)}]}


def project_from_motion_bytes(robot: Robot, content: bytes, filename: str, fps: int = 30):
    """Create an editable project from a Kimodo NPZ or 36-column G1 CSV."""
    if not 1 <= fps <= 120:
        raise ValueError('FPS must be 1–120')
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == '.csv':
            qpos = np.loadtxt(io.BytesIO(content), delimiter=',')
            qpos = np.atleast_2d(qpos)
        elif suffix == '.npz':
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                keys = set(archive.files)
                if keys == set(KIMODO_KEYS):
                    arrays = {key: archive[key] for key in KIMODO_KEYS}
                    qpos = kimodo_g1_to_qpos(robot, arrays)
                elif 'qpos' in keys:
                    qpos = np.asarray(archive['qpos'], dtype=float)
                else:
                    raise ValueError('NPZ must contain Kimodo G1 arrays or a qpos array')
        else:
            raise ValueError('Only .npz and .csv motion files are supported')
    except (EOFError, OSError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError(f'Could not read {suffix.upper()[1:]} motion file') from exc

    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != robot.model.nq:
        raise ValueError(f'Motion must contain one or more rows of {robot.model.nq} qpos values')
    if not 1 <= len(qpos) <= MAX_MOTION_SAMPLES:
        raise ValueError(f'Motion must contain 1–{MAX_MOTION_SAMPLES} frames')
    validated = np.stack([robot.validate_q(frame) for frame in qpos])
    for index in range(1, len(validated)):
        if np.dot(validated[index - 1, 3:7], validated[index, 3:7]) < 0:
            validated[index, 3:7] *= -1

    project = new_project(robot, Path(filename).stem or 'Imported motion')
    duration = max(1 / fps, (len(validated) - 1) / fps)
    project['keyframes'] = [{'name': 'Imported motion clip', 'duration': duration,
                             'qpos': validated[0].tolist(), 'pins': [], 'samples': validated.tolist()}]
    project['current_qpos'] = validated[0].tolist()
    project['pins'] = []
    return validate_project(robot, project)


def motion_result(robot: Robot, qpos, times, contacts, fps, pin_errors=()):
    qpos = np.asarray(qpos)
    times = np.asarray(times)
    qvel = np.zeros((len(qpos), robot.model.nv))
    for index in range(len(qpos)):
        lo, hi = max(0, index - 1), min(len(qpos) - 1, index + 1)
        if hi > lo:
            mujoco.mj_differentiatePos(robot.model, qvel[index], times[hi] - times[lo], qpos[lo], qpos[hi])
    qacc = np.gradient(qvel, times, axis=0) if len(times) > 1 else np.zeros_like(qvel)
    return {'time': times, 'qpos': qpos, 'qvel': qvel, 'qacc': qacc, 'contacts': np.asarray(contacts, dtype=bool),
            'fps': fps, 'max_pin_error_mm': max(pin_errors, default=0.)}


def compile_motion_clip(robot: Robot, frame, fps):
    source = np.stack([robot.validate_q(sample) for sample in frame['samples']])
    for index in range(1, len(source)):
        if np.dot(source[index - 1, 3:7], source[index, 3:7]) < 0:
            source[index, 3:7] *= -1
    if len(source) == 1:
        return motion_result(robot, source, [0.], np.zeros((1, len(FEET)), dtype=bool), fps)

    count = max(1, round(float(frame['duration']) * fps))
    times = np.arange(count + 1, dtype=float) / fps
    source_progress = np.linspace(0., 1., len(source))
    output_progress = np.arange(count + 1, dtype=float) / count
    qpos = np.empty((count + 1, robot.model.nq))
    for column in (*range(3), *range(7, robot.model.nq)):
        qpos[:, column] = np.interp(output_progress, source_progress, source[:, column])
    rotations = Rotation.from_quat(source[:, 3:7], scalar_first=True)
    qpos[:, 3:7] = Slerp(source_progress, rotations)(output_progress).as_quat(scalar_first=True)
    contacts = np.zeros((len(qpos), len(FEET)), dtype=bool)
    return motion_result(robot, qpos, times, contacts, fps)


def compile_motion(robot: Robot, project, fps=30):
    validate_project(robot, project)
    if not 1 <= fps <= 120:
        raise ValueError('FPS must be 1–120')
    frames = project['keyframes']
    if len(frames) == 1 and frames[0].get('samples') is not None:
        return compile_motion_clip(robot, frames[0], fps)
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
    return motion_result(robot, poses, times, contacts, fps, pin_errors)


def save_bundle(robot: Robot, project, fps=30, directory=None, protomotions=False):
    """Save NPZ, CSV, editable project and metadata in one self-contained folder."""
    motion = compile_motion(robot, project, fps)
    root_folder = Path(directory or ROOT / 'motions')
    root_folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^\w-]', '_', str(project.get('name', 'motion')), flags=re.UNICODE).strip('_')[:60] or 'motion'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    stem = f'{name}_{stamp}'
    folder = root_folder / stem
    folder.mkdir()
    floor = []
    for q in motion['qpos']:
        state = robot.state(q)
        floor.append(state['floor_min_mm'])
    environment = environment_snapshot(project)
    environment_text = json.dumps(environment, ensure_ascii=False, indent=2)
    environment_sha256 = hashlib.sha256(environment_text.encode('utf-8')).hexdigest()
    metadata = {'format': FORMAT, 'model_sha256': robot.fingerprint, 'joint_names': robot.names,
                'reference_schema': 'motioncreator.reference.v2',
                'npz_format': 'kimodo.g1.34',
                'npz_coordinate_system': 'right-handed, +Z forward, +Y up',
                'body_pose_frame': 'world; body frame origins, not centers of mass',
                'body_velocity_frame': 'world; body frame origins, not centers of mass',
                'joint_position_semantics': 'absolute hinge angles in radians; not policy actions',
                'root_body': 'pelvis', 'root_joint': 'free, 6 unactuated DoF',
                'trajectory_type': 'kinematic reference', 'loop_mode': 'clamp; no implicit loop',
                'control_binding': None,
                'quaternion_order': 'wxyz', 'coordinate_system': project['coordinate_system'], 'units': project['units'],
                'root_angular_velocity_frame': 'local body frame (MuJoCo free joint)',
                'root_linear_velocity_frame': 'world', 'contact_order': list(FEET),
                'npz_foot_contact_order': ['left heel', 'left toe', 'right heel', 'right toe'],
                'contact_semantics': 'Kimodo geometric heuristic: speed < 0.15 m/s and height < 0.10 m; not measured contact forces',
                'validation': {'kind': 'kinematic only', 'max_pin_error_mm': motion['max_pin_error_mm'],
                               'minimum_sole_height_mm': min(floor), 'max_joint_speed_rad_s': float(np.abs(motion['qvel'][:, 6:]).max()),
                               'self_collision_checked': False, 'dynamic_balance_checked': False},
                'interpolation': ('imported clip resampled with piecewise-linear position/joints and root SLERP'
                                  if project['keyframes'][0].get('samples') is not None else
                                  'quintic easing, shortest-path root SLERP, orientation-aware IK for shared pins; finite-difference velocities'),
                'environment_file': 'environment.json', 'environment_sha256': environment_sha256,
                'fps': fps, 'samples': len(motion['time'])}
    npz_path = folder / 'motion.npz'
    csv_path = folder / 'motion.csv'
    json_path = folder / 'project.json'
    environment_path = folder / 'environment.json'
    metadata_path = folder / 'metadata.json'
    np.savez_compressed(npz_path, **export_kimodo_g1(robot, motion['qpos'], fps))
    np.savetxt(csv_path, motion['qpos'], delimiter=',')
    json_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding='utf-8')
    environment_path.write_text(environment_text, encoding='utf-8')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    def relative(path):
        return str(path.relative_to(root_folder))
    exported = [npz_path, csv_path, json_path, environment_path, metadata_path]
    result = {'files': [relative(path) for path in exported], 'directory': str(folder), 'folder': stem,
              'metadata': metadata, 'project_file': relative(json_path), 'environment_file': relative(environment_path),
              'metadata_file': relative(metadata_path),
              'npz_file': relative(npz_path), 'csv_file': relative(csv_path)}
    if protomotions:
        from .protomotions_bridge import export_isolated
        try:
            result['files'].extend(str(Path(stem) / name) for name in export_isolated(npz_path))
        except (ValueError, OSError) as exc:
            # Native export failure must not hide the successfully saved editable reference.
            result['warnings'] = [f'선택한 모션 형식 저장 완료. ProtoMotions 변환 실패: {exc}']
    return result
