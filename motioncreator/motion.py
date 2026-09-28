"""Versioned project files and contact-aware kinematic reference exports."""
import hashlib
import io
import json
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from .robot import (Robot, ROOT, FEET, HANDLES, HINGES, ANGLE_LOCKABLE,
                    BASIC_ROTATABLE as ROTATABLE, quat_matrix, matrix_quat)
from .kimodo_format import KIMODO_KEYS, export_kimodo_g1, kimodo_g1_to_qpos

FORMAT = 'motioncreator.g1.v1'
ENVIRONMENT_FORMAT = 'motioncreator.environment.v1'
MAX_KEYFRAMES = 100
MAX_MOTION_SAMPLES = 3000
SCENE_SHAPES = {'box', 'sphere', 'cylinder', 'open_box'}
PROJECT_ID_PATTERN = re.compile(r'[0-9a-f]{32}')
PROJECT_ID_NAMESPACE = uuid.UUID('e0ad0f86-1a30-4e2f-a20f-67ad84175df8')
LEGACY_AUTO_PROJECT_NAMES = {'g1 reference', 'g1_reference'}
GENERIC_KEYFRAME_NAMES = {
    'stand', 'pose', 'frame', 'keyframe', 'start', 'start pose', 'imported motion clip',
    'standing', '서기', '서있기', '기본 서기', '기본 서기 자세', '자세', '키프레임', '시작', '시작 자세',
}


def _created_now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _normalize_project_naming(project):
    name = project.get('name')
    if not isinstance(name, str):
        return
    mode = project.get('name_mode')
    if mode is None:
        legacy_default = name.strip().casefold() in LEGACY_AUTO_PROJECT_NAMES
        legacy_display = project.get('display_name')
        mode = 'auto' if not name.strip() or (legacy_default and legacy_display in (None, name)) else 'manual'
    if mode not in ('auto', 'manual'):
        raise ValueError('Project name mode must be auto or manual')
    if mode == 'manual' and not name.strip():
        mode = 'auto'
    project['name_mode'] = mode
    if mode == 'auto':
        project['name'] = ''


def saved_project_id(relative_project_path):
    relative = Path(relative_project_path)
    return uuid.uuid5(PROJECT_ID_NAMESPACE, relative.as_posix()).hex


def prepare_saved_project(project, relative_project_path, modified_at=None):
    """Restore stable identity for a project opened from the server's saved list."""
    if project.get('project_id') is None:
        project['project_id'] = saved_project_id(relative_project_path)
    if project.get('created_at') is None:
        match = re.search(r'_(\d{8})T(\d{6})(?:_|$)', Path(relative_project_path).parent.name)
        if match:
            created = datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).astimezone()
        else:
            created = datetime.fromtimestamp(modified_at or datetime.now().timestamp()).astimezone()
        project['created_at'] = created.isoformat(timespec='seconds')
    return ensure_project_identity(project)


def ensure_project_identity(project):
    """Upgrade a legacy project with stable save identity fields."""
    _normalize_project_naming(project)
    identifier = project.get('project_id')
    if identifier is None:
        identifier = uuid.uuid4().hex
        project['project_id'] = identifier
    if not isinstance(identifier, str) or not PROJECT_ID_PATTERN.fullmatch(identifier):
        raise ValueError('Project ID must be 32 lowercase hexadecimal characters')
    created_at = project.get('created_at')
    if created_at is None:
        created_at = _created_now()
        project['created_at'] = created_at
    if not isinstance(created_at, str):
        raise ValueError('Project creation time must be ISO 8601 text')
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError('Project creation time must use ISO 8601') from exc
    if parsed.tzinfo is None:
        raise ValueError('Project creation time must include a timezone')
    return project


def automatic_project_name(project):
    """Build a readable default name from keyframe intent and creation date."""
    ensure_project_identity(project)
    names = []
    for frame in project.get('keyframes', []):
        value = re.sub(r'\s+', ' ', str(frame.get('name', '')).strip())
        if not value or value.casefold() in GENERIC_KEYFRAME_NAMES or value.casefold() in {name.casefold() for name in names}:
            continue
        names.append(value)
        if len(names) == 3:
            break
    core = '-'.join(names) if names else 'G1_모션'
    core = re.sub(r'[\\/:*?"<>|]+', '_', core).strip(' ._-')[:60] or 'G1_모션'
    date = datetime.fromisoformat(project['created_at']).strftime('%Y%m%d')
    return f'{core}_{date}'


def project_display_name(project):
    ensure_project_identity(project)
    manual = str(project.get('name', '')).strip()
    return manual if project['name_mode'] == 'manual' and manual else automatic_project_name(project)


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
        'scene_groups': project.get('scene_groups', []),
    }


def validate_project(robot: Robot, project):
    ensure_project_identity(project)
    if project.get('format') != FORMAT or project.get('model_sha256') not in robot.compatible_fingerprints:
        raise ValueError('Project format or G1 model fingerprint does not match')
    project['model_sha256'] = robot.fingerprint
    if project.get('joint_names') != robot.names:
        raise ValueError('Joint order does not match this model')
    if not isinstance(project.get('name'), str) or not isinstance(project.get('coordinate_system'), str) or not isinstance(project.get('units'), dict):
        raise ValueError('Project name, coordinate system and units are required')
    if len(project['name']) > 80:
        raise ValueError('Project name must contain at most 80 characters')
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
            if frame.get('angle_pins'):
                raise ValueError('Motion clips cannot use angle pins until they are split into keyframes')
        total += duration
        if any(k not in HANDLES for k in frame.get('pins', [])):
            raise ValueError('Unknown pinned handle')
        if any(k not in ANGLE_LOCKABLE for k in frame.get('angle_pins', [])):
            raise ValueError('Unknown angle-pinned handle')
    if total > 600:
        raise ValueError('Maximum project length is 600 seconds')
    if 'current_qpos' in project:
        robot.validate_q(project['current_qpos'])
    if any(k not in HANDLES for k in project.get('pins', [])):
        raise ValueError('Unknown pinned handle')
    if any(k not in ANGLE_LOCKABLE for k in project.get('angle_pins', [])):
        raise ValueError('Unknown angle-pinned handle')
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
        if 'fixed' in item and not isinstance(item['fixed'], bool):
            raise ValueError('Scene object fixed must be a boolean')
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
            raise ValueError('Scene object shape must be box, sphere, cylinder or open_box')
        asset_id = item.get('asset_id')
        if asset_id is not None:
            if not isinstance(asset_id, str) or not re.fullmatch(r'[0-9a-f]{24}', asset_id):
                raise ValueError('Scene object asset ID is invalid')
            try:
                from .scene_assets import asset_metadata
                metadata = asset_metadata(asset_id)
            except FileNotFoundError:
                metadata = None
            if metadata is not None:
                part_id = item.get('asset_part_id')
                part = next((value for value in metadata.get('parts', []) if value['part_id'] == part_id), None)
                if part_id is not None and part is None:
                    raise ValueError('Scene object asset part does not exist')
                if part is not None:
                    item['asset_node_name'] = part['node_name']
                if 'asset_axis_transform_xyzw' not in item and metadata['source_format'] == 'glb':
                    old_size = np.asarray(item.get('size'), dtype=float)
                    old_lower = np.asarray(item.get('asset_bounds_min'), dtype=float)
                    old_upper = np.asarray(item.get('asset_bounds_max'), dtype=float)
                    old_dimensions = old_upper - old_lower
                    new_dimensions = np.asarray(metadata['dimensions'], dtype=float)
                    if old_size.shape == (3,) and old_dimensions.shape == (3,) and np.all(old_dimensions > 0):
                        if shape in ('box', 'open_box'):
                            item['size'] = old_size[[0, 2, 1]].tolist()
                        elif shape == 'cylinder':
                            radial_scale = old_size[0] / max(old_dimensions[0], old_dimensions[1])
                            height_scale = old_size[2] / old_dimensions[2]
                            diameter = max(new_dimensions[0], new_dimensions[1]) * radial_scale
                            item['size'] = [diameter, diameter, new_dimensions[2] * height_scale]
                item['asset_bounds_min'] = (part or metadata)['bounds_min']
                item['asset_bounds_max'] = (part or metadata)['bounds_max']
                item['asset_axis_transform_xyzw'] = metadata['axis_transform_xyzw']
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
        if shape == 'open_box':
            thickness = float(item.get('wall_thickness_m', .02))
            if not np.isfinite(thickness) or not .001 <= thickness < min(size) / 3:
                raise ValueError('Open-box wall thickness must fit inside its dimensions')
        if asset_id is not None:
            part_id = item.get('asset_part_id')
            if part_id is not None and (not isinstance(part_id, str) or not re.fullmatch(r'part-[0-9]{2}-[a-z0-9-]{1,32}', part_id)):
                raise ValueError('Scene object asset part ID is invalid')
            node_name = item.get('asset_node_name')
            if node_name is not None and (not isinstance(node_name, str) or not 1 <= len(node_name) <= 256):
                raise ValueError('Scene object asset node name is invalid')
            for field in ('asset_bounds_min', 'asset_bounds_max'):
                value = np.asarray(item.get(field), dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f'Scene object {field} must contain three finite numbers')
            if np.any(np.asarray(item['asset_bounds_max']) <= np.asarray(item['asset_bounds_min'])):
                raise ValueError('Scene object asset bounds must have positive dimensions')
            axis_transform = np.asarray(item.get('asset_axis_transform_xyzw', [0., 0., 0., 1.]), dtype=float)
            if axis_transform.shape != (4,) or not np.isfinite(axis_transform).all() or abs(np.linalg.norm(axis_transform) - 1.) > 1e-4:
                raise ValueError('Scene object asset axis transform must be a normalized xyzw quaternion')
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
        placement = item.get('placement')
        if placement is not None:
            if not isinstance(placement, dict):
                raise ValueError('Scene object placement settings must be an object')
            for key in ('prevent_overlap', 'surface_snap', 'ground_lock'):
                if not isinstance(placement.get(key), bool):
                    raise ValueError(f'Scene object placement {key} must be a boolean')
    scene_groups = project.get('scene_groups', [])
    if not isinstance(scene_groups, list) or len(scene_groups) > 16:
        raise ValueError('Scene groups must be a list with at most 16 entries')
    grouped_members = set()
    group_ids = set()
    for group in scene_groups:
        if not isinstance(group, dict) or not isinstance(group.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', group['id']):
            raise ValueError('Scene group IDs must use 1–64 letters, numbers, underscores or hyphens')
        if group['id'] in group_ids:
            raise ValueError('Scene group IDs must be unique')
        group_ids.add(group['id'])
        if not isinstance(group.get('name'), str) or not 1 <= len(group['name']) <= 80:
            raise ValueError('Scene group names must contain 1–80 characters')
        members = group.get('member_ids')
        if not isinstance(members, list) or len(members) < 2 or any(member not in identifiers for member in members):
            raise ValueError('Scene groups require at least two existing objects')
        if len(set(members)) != len(members) or grouped_members.intersection(members):
            raise ValueError('A scene object can belong to only one group')
        grouped_members.update(members)
        for field, length in (('position', 3), ('quaternion_xyzw', 4)):
            value = np.asarray(group.get(field), dtype=float)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f'Scene group {field} must contain {length} finite numbers')
        if abs(np.linalg.norm(group['quaternion_xyzw']) - 1.) > 1e-4:
            raise ValueError('Scene group quaternion must be normalized')
    from .grasp import validate_grasp_event
    for frame in frames:
        grasp = frame.get('grasp')
        if grasp is not None:
            if frame.get('samples') is not None:
                raise ValueError('Grasp interactions require editable keyframes, not an imported motion clip')
            validate_grasp_event(robot, grasp, objects)
    return project


def new_project(robot, name=''):
    return {'format': FORMAT, 'name': name, 'name_mode': 'manual' if name.strip() else 'auto',
            'project_id': uuid.uuid4().hex, 'created_at': _created_now(),
            'model_sha256': robot.fingerprint,
            'joint_names': robot.names, 'coordinate_system': 'right-handed, +X forward, +Y left, +Z up',
            'units': {'position': 'm', 'angle': 'rad', 'time': 's'},
            'angle_pins': [],
            'keyframes': [{'name': 'Stand', 'duration': 2., 'qpos': robot.home.tolist(),
                           'pins': list(FEET), 'angle_pins': []}]}


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
    project['angle_pins'] = []
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
        angle_pins = sorted(set(a.get('angle_pins', [])) & set(b.get('angle_pins', [])))
        for k in pins:
            if np.linalg.norm(pa[k][0] - pb[k][0]) > .004:
                raise ValueError(f'{HANDLES[k][2]} changes position while pinned. Unpin it in one endpoint before moving it.')
            if k in FEET and np.linalg.norm(pa[k][1] - pb[k][1]) > .02:
                raise ValueError(f'{HANDLES[k][2]} changes orientation while pinned')
        for k in angle_pins:
            if k in HINGES:
                address = robot.model.joint(HINGES[k]).qposadr[0]
                if abs(qa[address] - qb[address]) > np.deg2rad(.5):
                    raise ValueError(f'{HANDLES[k][2]} changes joint angle while angle-pinned')
            elif np.linalg.norm(Rotation.from_matrix(pa[k][1].T @ pb[k][1]).as_rotvec()) > np.deg2rad(.5):
                raise ValueError(f'{HANDLES[k][2]} changes orientation while angle-pinned')
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
            if i != count and (pins or angle_pins):
                rotations = {k: pa[k][1] @ Rotation.from_rotvec(s*angular_deltas[k]).as_matrix() for k in ROTATABLE}
                targets = {k: ((1-s)*pa[k][0] + s*pb[k][0], rotations.get(k, pa[k][1])) for k in HANDLES}
                # Root orientation follows SLERP exactly; only end-effector rotations need projection.
                orientations = {k: Rotation.from_matrix(rotations[k]).as_quat() for k in ROTATABLE
                                if k != 'pelvis' and k not in angle_pins and not (k in FEET and k in pins)
                                and np.linalg.norm(angular_deltas[k]) > 1e-5}
                q, info = robot.solve(q, qa, pins=pins, targets=targets, max_nfev=18,
                                     posture_reference=q, posture_weight=.8, orientation_targets=orientations,
                                     angle_pins=angle_pins)
                if info['rejected']:
                    raise ValueError('The transition cannot preserve its pins. Add intermediate keyframes or relax a pin.')
                pin_errors.append(info['pin_error_mm'])
            poses.append(q)
            times.append(elapsed + i / fps)
            contacts.append([k in pins for k in FEET])
        elapsed += duration
    return motion_result(robot, poses, times, contacts, fps, pin_errors)


def _existing_project_folder(root_folder, project_id):
    matches = []
    for project_file in root_folder.glob('*/project.json'):
        try:
            stored = json.loads(project_file.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        stored_id = stored.get('project_id') or saved_project_id(project_file.relative_to(root_folder))
        if stored_id == project_id:
            matches.append(project_file.parent)
    if len(matches) > 1:
        raise ValueError('Multiple saved folders use this project ID; resolve the duplicate before saving')
    return matches[0] if matches else None


def _atomic_bytes(path, content):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path, content):
    _atomic_bytes(path, content.encode('utf-8'))


def save_bundle(robot: Robot, project, fps=30, directory=None, protomotions=False, save_as=False):
    """Create or atomically update one self-contained project folder."""
    if save_as:
        project = json.loads(json.dumps(project))
        project['project_id'] = uuid.uuid4().hex
        project['created_at'] = _created_now()
    ensure_project_identity(project)
    motion = compile_motion(robot, project, fps)
    root_folder = Path(directory or ROOT / 'motions')
    root_folder.mkdir(parents=True, exist_ok=True)
    display_name = project_display_name(project)
    project['display_name'] = display_name
    name = re.sub(r'[^\w-]', '_', display_name, flags=re.UNICODE).strip('_')[:72] or 'motion'
    created = datetime.fromisoformat(project['created_at'])
    if project['name_mode'] == 'auto':
        name = f'{name}T{created.strftime("%H%M%S")}'
    else:
        name = f'{name}_{created.strftime("%Y%m%dT%H%M%S")}'
    folder = _existing_project_folder(root_folder, project['project_id'])
    reused = folder is not None
    desired_folder = root_folder / f'{name}_{project["project_id"][:8]}'
    if folder is None:
        folder = desired_folder
        folder.mkdir()
    elif folder != desired_folder:
        if desired_folder.exists():
            raise ValueError('A different saved project already uses the requested project name')
        folder.rename(desired_folder)
        folder = desired_folder
    stem = folder.name
    floor = []
    for q in motion['qpos']:
        state = robot.state(q)
        floor.append(state['floor_min_mm'])
    environment = environment_snapshot(project)
    environment_text = json.dumps(environment, ensure_ascii=False, indent=2)
    environment_sha256 = hashlib.sha256(environment_text.encode('utf-8')).hexdigest()
    metadata = {'format': FORMAT, 'model_sha256': robot.fingerprint, 'joint_names': robot.names,
                'project_id': project['project_id'], 'created_at': project['created_at'],
                'display_name': display_name,
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
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **export_kimodo_g1(robot, motion['qpos'], fps))
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, motion['qpos'], delimiter=',')
    _atomic_bytes(npz_path, npz_buffer.getvalue())
    _atomic_text(csv_path, csv_buffer.getvalue())
    _atomic_text(json_path, json.dumps(project, ensure_ascii=False, indent=2))
    _atomic_text(environment_path, environment_text)
    _atomic_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    def relative(path):
        return str(path.relative_to(root_folder))
    exported = [npz_path, csv_path, json_path, environment_path, metadata_path]
    result = {'files': [relative(path) for path in exported], 'directory': str(folder), 'folder': stem,
              'project': project, 'display_name': display_name, 'reused': reused,
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
