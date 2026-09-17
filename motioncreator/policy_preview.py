"""Isolated, cancellable PD or SONIC physics playback on a free G1."""
import atexit
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .motion import compile_motion, project_scene_objects, validate_project
from .grasp import object_signature
from .grip_geometry import grip_pad_center, grip_pad_half_size, grip_pad_quaternion_wxyz
from .hand_collision import physical_hand_geom_names
from .robot import MODEL_PATH, ROOT, Robot
from .sonic import KP, KD, Reference, SonicCPU
from .task_jobs import atomic_json

MAX_SECONDS = 60


def runtime():
    paths = [ROOT / '.conda-policy/bin/python', *[
        ROOT / 'external/task-models/sonic' / name
        for name in ('model_encoder.onnx', 'model_decoder.onnx', 'observation_config.yaml')]]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    return {'available': not missing, 'physics_available': True, 'missing': missing,
            'device': 'cpu', 'max_seconds': MAX_SECONDS}


def _numbers(values):
    return ' '.join(str(float(value)) for value in values)


def project_from_keyframe(project, start_frame_index=0):
    """Create an isolated physics timeline beginning at an authored keyframe."""
    frames = project.get('keyframes', [])
    if not isinstance(start_frame_index, int) or isinstance(start_frame_index, bool):
        raise ValueError('Physics start keyframe index must be an integer')
    if not 0 <= start_frame_index < len(frames):
        raise ValueError('Physics start keyframe index is out of range')
    selected = copy.deepcopy(project)
    selected['keyframes'] = copy.deepcopy(frames[start_frame_index:])
    selected['current_qpos'] = copy.deepcopy(selected['keyframes'][0]['qpos'])
    selected['pins'] = copy.deepcopy(selected['keyframes'][0].get('pins', []))
    return selected


def preview_duration(project):
    frames = project['keyframes']
    base = float(frames[0]['duration']) if len(frames) == 1 else sum(float(frame['duration']) for frame in frames[1:])
    grasp = frames[0].get('grasp')
    return base + (float(grasp['closure_seconds']) if grasp and grasp.get('closure_qpos') is not None else 0.)


def compile_preview_motion(robot, project, fps=50):
    """Prepend a physics-only closing ramp when the first keyframe has a fitted grasp."""
    grasp = project['keyframes'][0].get('grasp')
    if not grasp or grasp.get('closure_qpos') is None:
        motion = compile_motion(robot, project, fps=fps)
        return motion['time'], motion['qpos'], None
    objects = project_scene_objects(project)
    item = next((value for value in objects if value['id'] == grasp['object_id']), None)
    if item is None or object_signature(item) != grasp.get('object_signature'):
        raise ValueError('파지 설정 후 대상 상자가 변경되었습니다. 양손 파지 자세를 다시 맞춰주세요.')
    closed = copy.deepcopy(project)
    contact_qpos = np.asarray(closed['keyframes'][0]['qpos'], dtype=float)
    closure_qpos = robot.validate_q(grasp['closure_qpos'])
    closed['keyframes'][0]['qpos'] = closure_qpos.tolist()
    closed_motion = compile_motion(robot, closed, fps=fps)
    base_times, base_poses = closed_motion['time'], closed_motion['qpos']
    if len(base_times) == 1:
        hold = float(closed['keyframes'][0]['duration'])
        base_times = np.array([0., hold])
        base_poses = np.stack([closure_qpos, closure_qpos])
    duration = float(grasp['closure_seconds'])
    count = max(1, round(duration * fps))
    close_times = np.arange(count + 1, dtype=float) / fps
    progress = (close_times / close_times[-1])[:, None]
    close_poses = np.repeat(contact_qpos[None], len(close_times), axis=0)
    close_poses[:, 7:] = contact_qpos[7:] + progress * (closure_qpos[7:] - contact_qpos[7:])
    return (np.r_[close_times, duration + base_times[1:]],
            np.concatenate([close_poses, base_poses[1:]], axis=0), grasp)


def _grounded_position(item):
    position = np.asarray(item['position'], dtype=float).copy()
    size = np.asarray(item['size'], dtype=float)
    if item['shape'] == 'sphere':
        extent = size[0] / 2
    else:
        x, y, z, w = np.asarray(item['quaternion_xyzw'], dtype=float)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)
        if item['shape'] == 'cylinder':
            extent = size[0] / 2 * np.hypot(r20, r21) + size[2] / 2 * abs(r22)
        else:
            extent = abs(r20) * size[0] / 2 + abs(r21) * size[1] / 2 + abs(r22) * size[2] / 2
    position[2] = max(position[2], extent)
    return position


def build_model(robot, project=None):
    root = ET.parse(MODEL_PATH).getroot()
    root.find('compiler').set('meshdir', str(ROOT / 'assets/g1/meshes'))
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='0.002', integrator='implicitfast', cone='elliptic',
                         iterations='80', gravity='0 0 -9.81')
    for index, name in enumerate(robot.names):
        root.find(f".//joint[@name='{name}']").set('armature', str(float(KP[index] / (20 * np.pi) ** 2)))
    objects = project_scene_objects(project or {})
    if objects:
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        for side in ('left', 'right'):
            body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
            physical_hand_geom_names(root, side)
            ET.SubElement(body, 'geom', name=f'{side}_preview_grip', type='box',
                          pos=_numbers(grip_pad_center(side)), quat=_numbers(grip_pad_quaternion_wxyz(side)),
                          size=_numbers(grip_pad_half_size()),
                          contype='0', conaffinity='0', group='3', density='0',
                          rgba='.15 .9 .72 .45' if side == 'left' else '1 .62 .25 .45')
        world = root.find('worldbody')
        object_geoms = []
        for index, item in enumerate(objects):
            body = ET.SubElement(world, 'body', name=f'preview_object_{index}',
                                 pos=_numbers(_grounded_position(item)),
                                 quat=_numbers(np.asarray(item['quaternion_xyzw'])[[3, 0, 1, 2]]))
            ET.SubElement(body, 'freejoint', name=f'preview_object_joint_{index}')
            size = np.asarray(item['size'], dtype=float)
            mj_size = size / 2 if item['shape'] == 'box' else [size[0] / 2] if item['shape'] == 'sphere' else [size[0] / 2, size[2] / 2]
            geom_name = f'preview_object_geom_{index}'
            ET.SubElement(body, 'geom', name=geom_name, type=item['shape'], size=_numbers(mj_size),
                          mass=str(float(item['mass_kg'])), friction=_numbers([item['friction'], .005, .0001]),
                          contype='0', conaffinity='0')
            object_geoms.append(geom_name)
            for side in ('left', 'right'):
                for hand_index in range(2):
                    ET.SubElement(contact, 'pair', geom1=f'{side}_physical_hand_{hand_index}', geom2=geom_name,
                                  condim='3', friction=_numbers([item['friction'], item['friction'], 0, 0, 0]),
                                  solref='.01 1', solimp='.95 .99 .001')
            ET.SubElement(contact, 'pair', geom1='floor', geom2=geom_name, condim='3',
                          friction=_numbers([item['friction'], item['friction'], 0, 0, 0]))
        for first, geom1 in enumerate(object_geoms):
            for geom2 in object_geoms[first + 1:]:
                ET.SubElement(contact, 'pair', geom1=geom1, geom2=geom2, condim='3')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    if model.nq != 36 + 7 * len(objects) or model.nu != 29 or model.neq:
        raise ValueError('Unexpected G1 physics model')
    return model


def simulate(project, progress=lambda value: None, *, controller='gear-sonic', start_frame_index=0, sonic_policy=None):
    if controller not in ('pd', 'gear-sonic'):
        raise ValueError('Unknown physics controller')
    robot = Robot()
    validate_project(robot, project)
    project = project_from_keyframe(project, start_frame_index)
    start_frame_name = project['keyframes'][0]['name']
    times, poses, grasp = compile_preview_motion(robot, project, fps=50)
    # A single authored pose is a hold, so it can also be tested under gravity.
    if len(times) == 1:
        times = np.array([0., float(project['keyframes'][0]['duration'])])
        poses = np.repeat(poses, 2, axis=0)
    if times[-1] > MAX_SECONDS:
        raise ValueError(f'물리 미리보기는 {MAX_SECONDS}초 이하 모션을 지원합니다.')
    objects = project_scene_objects(project)
    model = build_model(robot, project)
    data = mujoco.MjData(model)
    ref = Reference(times, poses, robot.model)
    policy = (sonic_policy or SonicCPU()) if controller == 'gear-sonic' else None
    if policy is not None and hasattr(policy, 'reset'):
        policy.reset()
    data.qpos[:36] = poses[0]
    # Preserve the reference start; do not silently shift/ground or teleport it.
    mujoco.mj_forward(model, data)
    joints = np.array([model.joint(name).id for name in robot.names])
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    motors = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0] for joint in joints])
    limits = model.jnt_actfrcrange[joints]
    grasp_object = next((index for index, item in enumerate(objects)
                         if grasp and item['id'] == grasp['object_id']), None)
    grasp_stats = {'left': {'max_normal_n': 0., 'contact_samples': 0},
                   'right': {'max_normal_n': 0., 'contact_samples': 0},
                   'bilateral_samples': 0, 'max_penetration_m': 0.}

    def update_grasp_stats():
        if grasp_object is None:
            return
        object_geom = model.geom(f'preview_object_geom_{grasp_object}').id
        per_hand = {'left': 0., 'right': 0.}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if object_geom not in (contact.geom1, contact.geom2):
                continue
            grasp_stats['max_penetration_m'] = max(grasp_stats['max_penetration_m'], -float(contact.dist))
            other = contact.geom1 if contact.geom2 == object_geom else contact.geom2
            for side in ('left', 'right'):
                actual_hand = {model.geom(f'{side}_physical_hand_{index}').id for index in range(2)}
                if other not in actual_hand:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, wrench)
                per_hand[side] += max(0., float(wrench[0]))
        for side, normal in per_hand.items():
            grasp_stats[side]['max_normal_n'] = max(grasp_stats[side]['max_normal_n'], normal)
            if normal > 0:
                grasp_stats[side]['contact_samples'] += 1
        if all(value > 0 for value in per_hand.values()):
            grasp_stats['bilateral_samples'] += 1
    def object_state():
        return {item['id']: {
            'position': data.xpos[model.body(f'preview_object_{index}').id].tolist(),
            'quaternion_xyzw': data.xquat[model.body(f'preview_object_{index}').id][[1, 2, 3, 0]].tolist(),
        } for index, item in enumerate(objects)}

    replay_times, states = [0.], [robot.state(data.qpos[:36].copy())]
    object_states = [object_state()]
    errors, saturation = [], []
    reason = 'completed'
    count = max(1, round(float(times[-1]) / .02))
    for step in range(count):
        # PD-only uses the reference joint angles directly: no learned balance
        # controller, root forces, pose teleportation, or ONNX dependency.
        target = (policy.action(data.qpos.copy(), data.qvel.copy(), ref, float(data.time), float(times[-1]))
                  if policy is not None else ref.sample(data.time)[0][0, 7:])
        for _ in range(10):
            torque = KP * (target - data.qpos[qa]) - KD * data.qvel[va]
            saturation.append(float(np.mean((torque < limits[:, 0]) | (torque > limits[:, 1]))))
            data.ctrl[motors] = np.clip(torque, limits[:, 0], limits[:, 1])
            mujoco.mj_step(model, data)
            update_grasp_stats()
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or any(data.warning[k].number for k in (mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC,
                    mujoco.mjtWarning.mjWARN_BADCTRL))):
            reason = 'numerical_instability'
            break
        mujoco.mj_forward(model, data)
        expected, _ = ref.sample(data.time)
        errors.append(float(np.mean((data.qpos[qa] - expected[0, 7:]) ** 2)))
        if data.qpos[2] < .25 or data.xmat[model.body('pelvis').id].reshape(3, 3)[2, 2] < np.cos(np.pi / 4):
            reason = 'fallen'
        if step % 2 == 1 or step == count - 1 or reason != 'completed':
            replay_times.append(float(data.time))
            states.append(robot.state(data.qpos[:36].copy()))
            object_states.append(object_state())
        if step % 10 == 0:
            progress((step + 1) / count)
        if reason != 'completed':
            break
    grasp_summary = None
    if grasp is not None:
        target_force = float(grasp['target_force_n'])
        max_force = float(grasp['max_force_n'])
        grasp_summary = {**grasp_stats, 'object_id': grasp['object_id'],
                         'target_force_n': target_force, 'max_force_n': max_force,
                         'bilateral_contact': grasp_stats['bilateral_samples'] > 0,
                         'target_reached': all(grasp_stats[side]['max_normal_n'] >= target_force for side in ('left', 'right')),
                         'force_limit_exceeded': any(grasp_stats[side]['max_normal_n'] > max_force for side in ('left', 'right'))}
    return {'time': replay_times, 'states': states, 'object_states': object_states, 'max_pin_error_mm': 0.,
            'physics': True, 'policy': 'gear-sonic' if policy is not None else None, 'controller': controller, 'summary': {
                'reason': reason, 'sim_seconds': float(replay_times[-1]),
                'reference_seconds': float(times[-1]), 'joint_rmse_rad': float(np.sqrt(np.mean(errors))) if errors else 0.,
                'max_torque_saturation': max(saturation, default=0.),
                'device': 'cpu', 'physics_hz': 500, 'policy_hz': 50 if policy is not None else 0,
                'target_hz': 50, 'start_frame_index': start_frame_index, 'start_frame_name': start_frame_name,
                'grasp': grasp_summary}}


class PreviewJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.sonic_process = None

    @staticmethod
    def _worker_environment():
        return {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONNOUSERSITE': '1',
                'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}

    def _sonic_worker(self, env):
        if self.sonic_process is not None and self.sonic_process.poll() is None:
            return self.sonic_process
        executable = str(ROOT / '.conda-policy/bin/python')
        self.sonic_process = subprocess.Popen(
            [executable, '-m', 'motioncreator.policy_preview', '--persistent-worker'],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, text=True, bufsize=1,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self.sonic_process

    def start(self, project, controller='gear-sonic', start_frame_index=0):
        if controller not in ('pd', 'gear-sonic'):
            raise ValueError('Unknown physics controller')
        robot = Robot()
        validate_project(robot, project)
        selected_project = project_from_keyframe(project, start_frame_index)
        duration = preview_duration(selected_project)
        if duration > MAX_SECONDS:
            raise ValueError(f'물리 미리보기는 {MAX_SECONDS}초 이하 모션을 지원합니다.')
        if controller == 'gear-sonic' and not runtime()['available']:
            raise ValueError('SONIC CPU 실행 환경 또는 모델이 없습니다. scripts/setup-task-cpu.sh를 확인하세요.')
        with self.lock:
            if any(job['status'] == 'running' for job in self.jobs.values()):
                raise ValueError('물리 계산이 진행 중입니다. 완료하거나 취소한 뒤 다시 실행하세요.')
            # Keep a few recent replays without accumulating temporary data indefinitely.
            while len(self.jobs) >= 4:
                old_id = next(iter(self.jobs))
                self.jobs.pop(old_id)['temporary'].cleanup()
            temporary = tempfile.TemporaryDirectory(prefix='motioncreator-sonic-')
            folder = Path(temporary.name)
            atomic_json(folder / 'project.json', project)
            env = self._worker_environment()
            try:
                if controller == 'gear-sonic':
                    process = self._sonic_worker(env)
                    if process.stdin is None:
                        raise OSError('GEAR-SONIC worker input is unavailable')
                    process.stdin.write(json.dumps({'folder': str(folder), 'start_frame_index': start_frame_index}) + '\n')
                    process.stdin.flush()
                    persistent = True
                else:
                    with (folder / 'worker.log').open('w') as log:
                        process = subprocess.Popen([sys.executable, '-m',
                            'motioncreator.policy_preview', str(folder), controller, str(start_frame_index)], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
                    persistent = False
            except (OSError, BrokenPipeError, ValueError) as exc:
                if controller == 'gear-sonic':
                    if 'process' in locals() and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                    self.sonic_process = None
                temporary.cleanup()
                raise ValueError(f'물리 CPU 작업을 시작할 수 없습니다: {exc}') from exc
            identifier = uuid.uuid4().hex
            job = {'status': 'running', 'process': process, 'folder': folder, 'temporary': temporary,
                   'persistent': persistent}
            self.jobs[identifier] = job
            threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return {'id': identifier, 'status': 'running', 'progress': 0., 'start_frame_index': start_frame_index}

    def _watch(self, job):
        if job.get('persistent'):
            return self._watch_persistent(job)
        try:
            code = job['process'].wait(timeout=600)
            message = ''
        except subprocess.TimeoutExpired:
            job['process'].kill()
            job['process'].wait()
            code, message = -1, 'CPU 계산이 10분을 초과했습니다. 짧은 모션으로 다시 실행하세요.'
        with self.lock:
            if job['status'] != 'running':
                return
            job['status'] = 'completed' if code == 0 and (job['folder'] / 'result.json').is_file() else 'failed'
            if job['status'] == 'failed':
                error_path = job['folder'] / 'error.json'
                job['message'] = message or (json.loads(error_path.read_text())['message'] if error_path.is_file()
                                            else '물리 CPU 작업이 종료되었습니다. 실행 환경을 확인하세요.')

    def _watch_persistent(self, job):
        deadline = time.monotonic() + 600
        result_path = job['folder'] / 'result.json'
        error_path = job['folder'] / 'error.json'
        message = ''
        while time.monotonic() < deadline:
            with self.lock:
                if job['status'] != 'running':
                    return
            if result_path.is_file() or error_path.is_file():
                break
            if job['process'].poll() is not None:
                message = 'GEAR-SONIC 상주 워커가 종료되었습니다. 다음 재생에서 다시 시작합니다.'
                break
            time.sleep(.1)
        else:
            message = 'CPU 계산이 10분을 초과했습니다. 짧은 모션으로 다시 실행하세요.'
            job['process'].terminate()
            try:
                job['process'].wait(timeout=2)
            except subprocess.TimeoutExpired:
                job['process'].kill(); job['process'].wait()
        with self.lock:
            if job['status'] != 'running':
                return
            job['status'] = 'completed' if result_path.is_file() else 'failed'
            if job['status'] == 'failed':
                job['message'] = message or (json.loads(error_path.read_text())['message'] if error_path.is_file()
                                            else 'GEAR-SONIC 물리 작업이 종료되었습니다. 실행 환경을 확인하세요.')
            if job['process'].poll() is not None and self.sonic_process is job['process']:
                self.sonic_process = None

    def status(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise KeyError('미리보기 작업을 찾을 수 없습니다. 다시 재생하세요.')
            path = job['folder'] / 'progress.json'
            progress = json.loads(path.read_text())['progress'] if path.is_file() else 0.
            if job['status'] == 'completed':
                progress = 1.
            return {'id': identifier, 'status': job['status'], 'progress': progress, 'message': job.get('message', '')}

    def result(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None or job['status'] != 'completed':
                raise ValueError('미리보기 결과가 준비되지 않았습니다.')
            return json.loads((job['folder'] / 'result.json').read_text())

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise KeyError('미리보기 작업을 찾을 수 없습니다.')
            if job['status'] == 'running':
                job['status'] = 'cancelled'
                job['process'].terminate()
                try:
                    job['process'].wait(timeout=2)
                except subprocess.TimeoutExpired:
                    job['process'].kill()
                    job['process'].wait()
                if job.get('persistent') and self.sonic_process is job['process']:
                    self.sonic_process = None
        return self.status(identifier)

    def close(self):
        for identifier in list(self.jobs):
            self.cancel(identifier)
            self.jobs[identifier]['temporary'].cleanup()
        if self.sonic_process is not None and self.sonic_process.poll() is None:
            self.sonic_process.terminate()
            try:
                self.sonic_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.sonic_process.kill(); self.sonic_process.wait()
        self.sonic_process = None


def run_persistent_worker(stream=None):
    """Serve multiple GEAR-SONIC previews with one pair of ONNX sessions."""
    policy = None
    for line in stream or sys.stdin:
        if not line.strip():
            continue
        command = json.loads(line)
        folder = Path(command['folder'])
        try:
            if policy is None:
                policy = SonicCPU()
            result = simulate(json.loads((folder / 'project.json').read_text()),
                              lambda value: atomic_json(folder / 'progress.json', {'progress': value}),
                              controller='gear-sonic', start_frame_index=int(command.get('start_frame_index', 0)),
                              sonic_policy=policy)
            atomic_json(folder / 'result.json', result)
        except Exception as exc:
            atomic_json(folder / 'error.json', {'message': str(exc)})


jobs = PreviewJobs()
atexit.register(jobs.close)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--persistent-worker':
        run_persistent_worker()
        raise SystemExit(0)
    folder = Path(sys.argv[1])
    try:
        result = simulate(json.loads((folder / 'project.json').read_text()),
                          lambda value: atomic_json(folder / 'progress.json', {'progress': value}),
                          controller=sys.argv[2] if len(sys.argv) > 2 else 'gear-sonic',
                          start_frame_index=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        atomic_json(folder / 'result.json', result)
    except Exception as exc:
        atomic_json(folder / 'error.json', {'message': str(exc)})
        raise
