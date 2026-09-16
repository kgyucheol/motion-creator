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
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .motion import compile_motion, project_scene_objects, validate_project
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
            visual = body.find(f"geom[@mesh='{side}_rubber_hand']")
            grip = copy.deepcopy(visual)
            grip.attrib.update(name=f'{side}_preview_grip', contype='1', conaffinity='1', group='3', density='0')
            body.append(grip)
        world = root.find('worldbody')
        for index, item in enumerate(objects):
            body = ET.SubElement(world, 'body', name=f'preview_object_{index}',
                                 pos=_numbers(item['position']),
                                 quat=_numbers(np.asarray(item['quaternion_xyzw'])[[3, 0, 1, 2]]))
            ET.SubElement(body, 'freejoint', name=f'preview_object_joint_{index}')
            size = np.asarray(item['size'], dtype=float)
            mj_size = size / 2 if item['shape'] == 'box' else [size[0] / 2] if item['shape'] == 'sphere' else [size[0] / 2, size[2] / 2]
            geom_name = f'preview_object_geom_{index}'
            ET.SubElement(body, 'geom', name=geom_name, type=item['shape'], size=_numbers(mj_size),
                          mass=str(float(item['mass_kg'])), friction=_numbers([item['friction'], .005, .0001]))
            for side in ('left', 'right'):
                ET.SubElement(contact, 'pair', geom1=f'{side}_preview_grip', geom2=geom_name,
                              condim='3', friction=_numbers([item['friction'], item['friction'], 0, 0, 0]),
                              solref='.01 1', solimp='.95 .99 .001')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    if model.nq != 36 + 7 * len(objects) or model.nu != 29 or model.neq:
        raise ValueError('Unexpected G1 physics model')
    return model


def simulate(project, progress=lambda value: None, *, controller='gear-sonic'):
    if controller not in ('pd', 'gear-sonic'):
        raise ValueError('Unknown physics controller')
    robot = Robot()
    motion = compile_motion(robot, project, fps=50)
    times, poses = motion['time'], motion['qpos']
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
    policy = SonicCPU() if controller == 'gear-sonic' else None
    data.qpos[:36] = poses[0]
    # Preserve the reference start; do not silently shift/ground or teleport it.
    mujoco.mj_forward(model, data)
    joints = np.array([model.joint(name).id for name in robot.names])
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    motors = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0] for joint in joints])
    limits = model.jnt_actfrcrange[joints]
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
    return {'time': replay_times, 'states': states, 'object_states': object_states, 'max_pin_error_mm': 0.,
            'physics': True, 'policy': 'gear-sonic' if policy is not None else None, 'controller': controller, 'summary': {
                'reason': reason, 'sim_seconds': float(replay_times[-1]),
                'reference_seconds': float(times[-1]), 'joint_rmse_rad': float(np.sqrt(np.mean(errors))) if errors else 0.,
                'max_torque_saturation': max(saturation, default=0.),
                'device': 'cpu', 'physics_hz': 500, 'policy_hz': 50 if policy is not None else 0,
                'target_hz': 50}}


class PreviewJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}

    def start(self, project, controller='gear-sonic'):
        if controller not in ('pd', 'gear-sonic'):
            raise ValueError('Unknown physics controller')
        validate_project(Robot(), project)
        frames = project['keyframes']
        duration = frames[0]['duration'] if len(frames) == 1 else sum(f['duration'] for f in frames[1:])
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
            env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONNOUSERSITE': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}
            try:
                with (folder / 'worker.log').open('w') as log:
                    executable = str(ROOT / '.conda-policy/bin/python') if controller == 'gear-sonic' else sys.executable
                    process = subprocess.Popen([executable, '-m',
                        'motioncreator.policy_preview', str(folder), controller], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                temporary.cleanup()
                raise ValueError(f'물리 CPU 작업을 시작할 수 없습니다: {exc}') from exc
            identifier = uuid.uuid4().hex
            job = {'status': 'running', 'process': process, 'folder': folder, 'temporary': temporary}
            self.jobs[identifier] = job
            threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return {'id': identifier, 'status': 'running', 'progress': 0.}

    def _watch(self, job):
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
        return self.status(identifier)

    def close(self):
        for identifier in list(self.jobs):
            self.cancel(identifier)
            self.jobs[identifier]['temporary'].cleanup()


jobs = PreviewJobs()
atexit.register(jobs.close)


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    try:
        result = simulate(json.loads((folder / 'project.json').read_text()),
                          lambda value: atomic_json(folder / 'progress.json', {'progress': value}),
                          controller=sys.argv[2] if len(sys.argv) > 2 else 'gear-sonic')
        atomic_json(folder / 'result.json', result)
    except Exception as exc:
        atomic_json(folder / 'error.json', {'message': str(exc)})
        raise
