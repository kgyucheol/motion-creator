import json
import numpy as np
import pytest
from motioncreator.robot import Robot
from motioncreator.motion import new_project, save_bundle
from motioncreator.reference import load_reference, joint_permutation


@pytest.fixture
def reference(tmp_path):
    r = Robot()
    p = new_project(r)
    p['keyframes'][0]['pins'] = []
    q = r.home.copy(); q[0] += .1
    q[3:7] = [.995004165278, 0, 0, .099833416647]
    p['keyframes'].append({'name': 'Move and rotate', 'duration': 1., 'qpos': q.tolist(), 'pins': []})
    result = save_bundle(r, p, fps=15, directory=tmp_path)
    return tmp_path / result['npz_file']


def test_reference_world_velocities_and_metadata(reference):
    d, meta = load_reference(reference)
    assert meta['reference_schema'] == 'motioncreator.reference.v2'
    assert meta['control_binding'] is None
    assert d['body_parent_indices'][0] == -1
    assert d['body_lin_vel_world'].shape == d['body_pos'].shape
    assert np.allclose(d['body_lin_vel_world'][:, 0], d['root_lin_vel_world'])
    assert np.allclose(d['body_ang_vel_world'][:, 0], d['root_ang_vel_world'])
    # Rotation induces tangential body velocity in addition to root translation.
    from scipy.spatial.transform import Rotation
    world_omega = Rotation.from_quat(d['root_quat_wxyz'][:, [1,2,3,0]]).apply(d['qvel'][:, 3:6])
    assert np.allclose(world_omega, d['root_ang_vel_world'])
    expected = d['root_lin_vel_world'][:, None] + np.cross(world_omega[:, None], d['body_pos']-d['root_pos'][:, None])
    assert np.allclose(d['body_lin_vel_world'], expected, atol=1e-8)


def test_legacy_reference_and_name_mapping(reference, tmp_path):
    d, meta = load_reference(reference)
    meta.pop('reference_schema'); d['metadata_json'] = np.array(json.dumps(meta)); d.pop('fps')
    path = tmp_path/'old.npz'; np.savez(path, **d)
    legacy, _ = load_reference(path)
    assert np.array_equal(legacy['qpos'], d['qpos'])
    assert joint_permutation(['b','a'], ['a','b']) == [1,0]
    with pytest.raises(ValueError, match='match'):
        joint_permutation(['b','a'], ['c','b'])


@pytest.mark.parametrize('case', ['time', 'quaternion', 'names', 'units', 'qpos', 'fps'])
def test_invalid_reference_is_rejected(reference, tmp_path, case):
    d, meta = load_reference(reference)
    if case == 'time': d['time'][2] = d['time'][1]
    if case == 'quaternion': d['root_quat_wxyz'][0] *= 2
    if case == 'names': d['joint_names'][1] = d['joint_names'][0]
    if case == 'units': meta['units']['position'] = 'cm'
    if case == 'qpos': d['qpos'][0, 0] += .1
    if case == 'fps': d['fps'] = 60
    d['metadata_json'] = np.array(json.dumps(meta))
    path = tmp_path/'invalid.npz'; np.savez(path, **d)
    with pytest.raises(ValueError): load_reference(path)


def test_native_failure_preserves_original_files(tmp_path, monkeypatch):
    from motioncreator import protomotions_bridge
    def fail(path): raise ValueError('test unavailable environment')
    monkeypatch.setattr(protomotions_bridge, 'export_isolated', fail)
    r = Robot(); result = save_bundle(r, new_project(r), directory=tmp_path, protomotions=True)
    assert len(result['files']) == 5 and result['warnings']
    assert all((tmp_path/name).is_file() for name in result['files'])
