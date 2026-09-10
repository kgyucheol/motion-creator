"""Optional ProtoMotions 3 adapter; heavy dependencies stay in a separate process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from .reference import load_reference, joint_permutation

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / 'integrations/protomotions-profile.json'
UPSTREAM = ROOT / 'external/ProtoMotions'
PYTHON = ROOT / '.conda-protomotions/bin/python'


def export_isolated(source):
    if not PYTHON.is_file() or not (UPSTREAM / 'protomotions/components/pose_lib.py').is_file():
        raise ValueError('변환 환경이 없습니다. ./scripts/setup-protomotions.sh를 실행하세요.')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MUJOCO_GL': 'egl'}
    try:
        result = subprocess.run([str(PYTHON), '-m', 'motioncreator.protomotions_bridge', str(source)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('ProtoMotions 변환 시간이 180초를 초과했습니다.') from exc
    if result.returncode:
        raise ValueError((result.stderr or result.stdout)[-1500:])
    return json.loads(result.stdout.strip().splitlines()[-1])['files']


def convert_reference(source, output=None, upstream=UPSTREAM):
    """Preserve timing/root/angles; recompute target FK, never silently ground-shift."""
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    profile = json.loads(PROFILE_PATH.read_text())
    target = upstream / profile['target_mjcf']
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != profile['target_mjcf_sha256']:
        raise ValueError('ProtoMotions G1 asset differs from the verified profile')
    revision = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if revision != profile['revision']:
        raise ValueError('ProtoMotions revision differs from the verified profile')
    data, meta = load_reference(source)
    if meta.get('model_sha256') not in [profile['source_mjcf_sha256'], *profile.get('legacy_source_mjcf_sha256', [])]:
        raise ValueError('Source G1 model differs from the verified profile')
    if len(data['time']) < 2:
        raise ValueError('ProtoMotions에는 2프레임 이상이 필요합니다. 같은 자세를 복제해 유지 시간을 지정하세요.')
    sys.path.insert(0, str(upstream))
    import torch
    from protomotions.components.pose_lib import extract_kinematic_info, fk_batch_mjcf_with_velocities, compute_cartesian_velocity
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    torch.set_num_threads(1)
    kin = extract_kinematic_info(str(target)).to(torch.device('cpu'))
    if kin.body_names[0] != profile['root_body'] or kin.num_dofs != 29:
        raise ValueError('Unexpected target root or joint count')
    permutation = joint_permutation(data['joint_names'].tolist(), kin.dof_names)
    dofs = data['dof_pos'][:, permutation]
    lower, upper = kin.dof_limits_lower.numpy(), kin.dof_limits_upper.numpy()
    invalid = np.any((dofs < lower-1e-6) | (dofs > upper+1e-6), axis=0)
    if invalid.any():
        raise ValueError('Target G1 joint limits exceeded: ' + ', '.join(np.array(kin.dof_names)[invalid]))
    qpos = np.concatenate([data['root_pos'], data['root_quat_wxyz'], dofs], axis=-1)
    # Sign-continuous root quaternions avoid representation flips in downstream consumers.
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    for i in range(1, len(qpos)):
        if np.dot(qpos[i-1, 3:7], qpos[i, 3:7]) < 0:
            qpos[i, 3:7] *= -1
    fps = float(meta['fps'])
    state = fk_batch_mjcf_with_velocities(kin, torch.tensor(qpos, dtype=torch.float32),
                                        fps=fps, velocity_max_horizon=1)
    state.dof_pos = torch.tensor(dofs, dtype=torch.float32)
    state.dof_vel = compute_cartesian_velocity(state.dof_pos.unsqueeze(1), fps=fps).squeeze(1)
    if 'contacts' in data:
        flags = torch.zeros((len(qpos), kin.num_bodies), dtype=torch.bool)
        for i, key in enumerate(profile['contact_bodies']):
            flags[:, kin.body_names.index(key)] = torch.tensor(data['contacts'][:, i], dtype=torch.bool)
        state.rigid_body_contacts = flags
    for key in ('rigid_body_pos', 'rigid_body_rot', 'rigid_body_vel', 'rigid_body_ang_vel', 'dof_pos', 'dof_vel'):
        if not torch.isfinite(getattr(state, key)).all():
            raise ValueError('Non-finite target reference: ' + key)
    output = Path(output).resolve() if output else source.with_suffix('.motion')
    if output.suffix != '.motion':
        raise ValueError('Output extension must be .motion')
    library = output.with_suffix('.pt')
    report_path = output.with_suffix('.protomotions.json')
    paths = [output, library, report_path]
    if any(p.exists() for p in paths):
        raise ValueError('Output already exists; choose a new output name')
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison = {}
    if 'body_names' in data and 'body_pos' in data:
        names = data['body_names'].tolist()
        for body in profile['contact_bodies']:
            if body in names:
                difference = state.rigid_body_pos[:, kin.body_names.index(body)].numpy()-data['body_pos'][:, names.index(body)]
                comparison[body] = float(np.max(np.linalg.norm(difference, axis=-1))*1000)
    report = {
        'format': 'motioncreator.protomotions-export.v1', 'upstream': profile,
        'source_file': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_model_sha256': meta.get('model_sha256'),
        'source_reference_schema': meta.get('reference_schema', 'motioncreator.reference.v1'),
        'fps': fps, 'samples': len(qpos), 'duration_s': float(data['time'][-1]),
        'body_names': kin.body_names, 'joint_names': kin.dof_names,
        'joint_permutation_from_source': permutation,
        'quaternion_order': 'xyzw', 'body_pose_frame': 'world; body origins',
        'body_velocity_frame': 'world', 'units': meta['units'], 'coordinate_system': meta['coordinate_system'],
        'velocities': 'ProtoMotions forward difference, horizon=1; last valid velocity repeated',
        'root_translation_modified': False, 'height_correction_m': 0.,
        'contacts': 'authored support flags on ankle_roll bodies; not measured forces or geometric contact detection',
        'source_target_ankle_origin_max_difference_mm': comparison,
        'validation': {'native_motionlib_load': True, 'dynamic_balance_checked': False, 'controller_tracking_checked': False},
        'controller_binding': None,
        'sim2sim_required': ['checkpoint and observation/action contract', 'joint/body ordering',
                            'control and physics periods', 'PD gains, action scale, default pose and torque limits',
                            'target MJCF and contact/material settings', 'payload and grasp constraints'],
    }
    # Stage all files first so a failed upstream load never leaves a misleading .motion.
    with tempfile.TemporaryDirectory(prefix='.proto-', dir=output.parent) as staging:
        staged = Path(staging) / output.name
        torch.save(state.to_dict(), staged)
        lib = MotionLib(MotionLibConfig(motion_file=str(staged)), device='cpu')
        report['packaged_contacts_available'] = lib.contacts is not None
        lib.motion_files = (str(output),)
        lib.save_to_file(Path(staging) / library.name)
        (Path(staging) / report_path.name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        for destination in paths:
            (Path(staging) / destination.name).replace(destination)
    return {'files': [p.name for p in paths], 'directory': str(output.parent), 'report': report}


def main():
    parser = argparse.ArgumentParser(description='Convert G1 NPZ to ProtoMotions .motion and .pt using the verified target G1 FK')
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--upstream', type=Path, default=UPSTREAM)
    args = parser.parse_args()
    result = convert_reference(args.source, args.output, args.upstream)
    print(json.dumps({'files': result['files'], 'directory': result['directory']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
