import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from motioncreator.robot import Robot, FEET
from motioncreator.demo import crouch_demo
from motioncreator.motion import automatic_project_name, compile_motion, new_project, prepare_saved_project, project_from_motion_bytes, save_bundle, validate_project
from motioncreator.reference import load_reference
from motioncreator import server
from motioncreator.server import app


@pytest.fixture(scope='module')
def robot():
    return Robot()


def test_grounded_real_29_dof_model(robot):
    assert robot.model.nq == 36 and robot.model.nv == 35
    assert len(robot.names) == 29
    assert abs(robot.state(robot.home)['floor_min_mm']) < .01


def test_project_keeps_a_saved_attention_pose_separate_from_default_stand(robot, tmp_path):
    project = new_project(robot)
    attention = robot.home.copy()
    attention[robot.model.joint('left_elbow_joint').qposadr[0]] += .1
    project['attention_pose'] = {'qpos': attention.tolist(), 'pins': list(FEET),
                                 'angle_pins': ['left_elbow']}
    validated = validate_project(robot, project)
    np.testing.assert_allclose(validated['keyframes'][0]['qpos'], robot.home)
    np.testing.assert_allclose(validated['attention_pose']['qpos'], attention)
    bundle = save_bundle(robot, validated, fps=15, directory=tmp_path)
    reopened = json.loads((tmp_path / bundle['project_file']).read_text())
    np.testing.assert_allclose(reopened['attention_pose']['qpos'], attention)


def test_crouch_keeps_feet_and_distant_hands(robot):
    target = robot.home[:3].copy()
    target[2] -= .15
    q, info = robot.solve(robot.home, robot.home, 'pelvis', target, FEET, max_nfev=60)
    assert info['converged'] and info['target_error_mm'] < 2
    a, b = robot.data(robot.home), robot.data(q)
    for key in FEET:
        p0, r0 = robot.point(a, key)
        p1, r1 = robot.point(b, key)
        assert np.linalg.norm(p1-p0) < .0001
        assert np.linalg.norm(r1-r0) < .001
    for key in ('left_hand', 'right_hand'):
        assert np.linalg.norm(robot.point(a, key)[0] - robot.point(b, key)[0]) < .015
    assert q[robot.model.joint('left_knee_joint').qposadr[0]] > 1.
    robot.validate_q(q)


def test_explicit_hand_pin_and_unreachable_target(robot):
    target = robot.home[:3].copy()
    target[2] -= .10
    q, info = robot.solve(robot.home, robot.home, 'pelvis', target, [*FEET, 'left_hand'], max_nfev=60)
    assert not info['rejected']
    assert np.linalg.norm(robot.point(robot.data(q), 'left_hand')[0] - robot.point(robot.data(robot.home), 'left_hand')[0]) < .003
    unreachable = robot.point(robot.data(q), 'right_hand')[0] + [2, 0, 0]
    out, info = robot.solve(q, q, 'right_hand', unreachable, FEET)
    assert not info['converged']
    robot.validate_q(out)
    for key in FEET:
        assert np.linalg.norm(robot.point(robot.data(q), key)[0] - robot.point(robot.data(out), key)[0]) < .003


def test_motion_endpoints_contacts_and_smoothness(robot):
    project = crouch_demo(robot)
    motion = compile_motion(robot, project, fps=30)
    assert motion['qpos'].shape == (121, 36)
    assert motion['qvel'].shape == (121, 35)
    assert np.allclose(np.diff(motion['time']), 1/30)
    assert motion['contacts'].all()
    assert np.array_equal(motion['qpos'][0], project['keyframes'][0]['qpos'])
    assert np.array_equal(motion['qpos'][60], project['keyframes'][1]['qpos'])
    assert np.array_equal(motion['qpos'][-1], project['keyframes'][2]['qpos'])
    # Catch branch changes and discontinuities at the endpoints after IK projection.
    assert np.max(np.abs(np.diff(motion['qpos'][:, 7:], axis=0))) < .10
    assert np.max(np.abs(motion['qvel'][[0, -1]])) < .05
    for key in FEET:
        initial = robot.point(robot.data(robot.home), key)[0]
        assert max(np.linalg.norm(robot.point(robot.data(q), key)[0] - initial) for q in motion['qpos']) < .001
    assert all(np.isfinite(motion[k]).all() for k in ('qpos', 'qvel', 'qacc'))


def test_generated_middle_keyframe_keeps_velocity_through_the_middle(robot):
    project = new_project(robot)
    address = robot.model.joint('left_elbow_joint').qposadr[0]
    frames = []
    for index in range(3):
        qpos = robot.home.copy()
        qpos[address] += .2 * index
        frames.append({'name': f'Pose {index}', 'duration': 1., 'qpos': qpos.tolist(),
                       'pins': [], 'angle_pins': [], **({'generated_transition': True} if index == 1 else {})})
    project['keyframes'] = frames
    assert validate_project(robot, project)['keyframes'][1]['generated_transition'] is True
    motion = compile_motion(robot, project, fps=60)
    np.testing.assert_allclose(motion['qpos'][60], frames[1]['qpos'])
    before = (motion['qpos'][60, address] - motion['qpos'][59, address]) * 60
    after = (motion['qpos'][61, address] - motion['qpos'][60, address]) * 60
    assert before > .2 and after > .2
    assert abs(before - after) < .02


def test_project_and_npz_roundtrip(robot, tmp_path):
    project = crouch_demo(robot)
    project['name'] = '../../outside/한글 모션'
    project['scene_objects'] = [{
        'id': 'crate', 'name': 'Crate', 'shape': 'box', 'position': [.6, 0., .2],
        'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.4, .3, .4], 'mass_kg': 2.,
        'friction': .7, 'color': '#336699', 'opacity': .8, 'visible': True,
        'placement': {'prevent_overlap': False, 'surface_snap': True, 'ground_lock': False},
    }, {
        'id': 'cup', 'name': 'Cup', 'parent_id': 'crate', 'shape': 'cylinder', 'position': [.6, 0., .5],
        'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.1, .1, .2], 'mass_kg': .2,
        'friction': .7, 'color': '#cc9966', 'opacity': 1., 'visible': True,
    }]
    bundle = save_bundle(robot, project, fps=15, directory=tmp_path)
    for filename in bundle['files']:
        assert (tmp_path / filename).parent.parent == tmp_path
    assert [Path(name).name for name in bundle['files']] == ['motion.npz', 'motion.csv', 'project.json', 'environment.json', 'metadata.json']
    editable = json.loads((tmp_path / bundle['project_file']).read_text())
    validate_project(robot, editable)
    environment_path = tmp_path / bundle['environment_file']
    environment = json.loads(environment_path.read_text())
    assert environment['format'] == 'motioncreator.environment.v1'
    assert environment['coordinate_system'] == project['coordinate_system']
    assert environment['physics']['gravity_m_s2'] == [0., 0., -9.81]
    assert environment['physics']['floor']['friction'] == [1., .005, .0001]
    assert environment['scene_objects'] == project['scene_objects']
    assert editable['scene_objects'][1]['parent_id'] == 'crate'
    assert editable['scene_objects'][0]['placement'] == {
        'prevent_overlap': False, 'surface_snap': True, 'ground_lock': False,
    }
    assert bundle['metadata']['environment_sha256'] == hashlib.sha256(environment_path.read_bytes()).hexdigest()
    with np.load(tmp_path / bundle['npz_file'], allow_pickle=False) as data:
        assert set(data.files) == {'posed_joints', 'global_rot_mats', 'local_rot_mats', 'root_positions', 'foot_contacts'}
        assert data['posed_joints'].shape == (61, 34, 3)
        assert data['global_rot_mats'].shape == (61, 34, 3, 3)
        assert data['local_rot_mats'].shape == (61, 34, 3, 3)
        assert data['root_positions'].shape == (61, 3)
        assert data['foot_contacts'].shape == (61, 4)
        assert data['posed_joints'].dtype == np.float32
        assert data['foot_contacts'].dtype == np.bool_
    csv = np.loadtxt(tmp_path / bundle['csv_file'], delimiter=',')
    assert csv.shape == (61, 36)
    assert np.array_equal(csv, compile_motion(robot, project, fps=15)['qpos'])
    data, meta = load_reference(tmp_path / bundle['npz_file'])
    assert meta['npz_format'] == 'kimodo.g1.34'
    assert np.allclose(data['qpos'], compile_motion(robot, project, fps=15)['qpos'], atol=2e-6)
    assert data['dof_pos'].shape == (61, 29)
    assert data['joint_names'].tolist() == robot.names
    assert data['root_quat_wxyz'].shape == (61, 4)
    assert np.allclose(np.linalg.norm(data['root_quat_wxyz'], axis=1), 1)


def test_scene_object_parent_links_require_existing_acyclic_objects(robot):
    project = new_project(robot)
    base = {'shape': 'box', 'position': [.6, 0., .2], 'quaternion_xyzw': [0., 0., 0., 1.],
            'size': [.2, .2, .2], 'mass_kg': 1., 'friction': .7, 'color': '#336699',
            'opacity': 1., 'visible': True}
    project['scene_objects'] = [{**base, 'id': 'parent', 'name': 'Parent'},
                                {**base, 'id': 'child', 'name': 'Child', 'parent_id': 'parent'}]
    validate_project(robot, project)
    project['scene_objects'][0]['parent_id'] = 'child'
    with pytest.raises(ValueError, match='cycle'):
        validate_project(robot, project)
    project['scene_objects'][0]['parent_id'] = None
    project['scene_objects'][1]['parent_id'] = 'missing'
    with pytest.raises(ValueError, match='parent'):
        validate_project(robot, project)


def test_automatic_project_name_uses_keyframe_intent_and_creation_date(robot):
    project = new_project(robot)
    project['created_at'] = '2026-09-17T09:30:00+09:00'
    project['keyframes'] = [
        {**project['keyframes'][0], 'name': '기본 서기 자세'},
        {**project['keyframes'][0], 'name': '상자 접근'},
        {**project['keyframes'][0], 'name': '양손 파지'},
        {**project['keyframes'][0], 'name': '들어 올리기'},
        {**project['keyframes'][0], 'name': '내려놓기'},
    ]

    assert automatic_project_name(project) == '상자 접근-양손 파지-들어 올리기_20260917'


def test_manual_project_folder_also_includes_creation_time(robot, tmp_path):
    project = new_project(robot, '파지 실험')
    project['created_at'] = '2026-09-17T09:30:45+09:00'

    result = save_bundle(robot, project, fps=15, directory=tmp_path)

    assert Path(result['directory']).name.startswith('파지_실험_20260917T093045_')


def test_repeated_save_updates_same_project_folder_and_save_as_creates_copy(robot, tmp_path):
    project = new_project(robot)
    project['created_at'] = '2026-09-17T09:30:00+09:00'
    project['keyframes'][0]['name'] = '상자 접근'

    first = save_bundle(robot, project, fps=15, directory=tmp_path)
    original_id = project['project_id']
    assert Path(first['directory']).name.startswith('상자_접근_20260917T093000_')
    project['keyframes'][0]['name'] = '상자 파지'
    second = save_bundle(robot, project, fps=15, directory=tmp_path)

    assert first['directory'] != second['directory']
    assert not Path(first['directory']).exists()
    assert first['reused'] is False
    assert second['reused'] is True
    assert second['display_name'] == '상자 파지_20260917'
    assert len(list(tmp_path.glob('*/project.json'))) == 1
    updated = json.loads((tmp_path / second['project_file']).read_text())
    assert updated['project_id'] == original_id
    assert updated['display_name'] == '상자 파지_20260917'
    assert updated['keyframes'][0]['name'] == '상자 파지'
    third = save_bundle(robot, project, fps=15, directory=tmp_path)
    assert third['directory'] == second['directory']
    assert third['reused'] is True

    copied = save_bundle(robot, project, fps=15, directory=tmp_path, save_as=True)
    assert copied['directory'] != second['directory']
    assert copied['project']['project_id'] != original_id
    assert copied['reused'] is False
    assert len(list(tmp_path.glob('*/project.json'))) == 2


def test_legacy_saved_project_reuses_and_renames_its_existing_folder(robot, tmp_path):
    project = new_project(robot, 'G1 reference')
    project.pop('name_mode')
    project.pop('project_id')
    project.pop('created_at')
    source = project['keyframes'][0]
    project['keyframes'] = [{**source, 'name': name} for name in ('서있기', '손 접근', '상자 파지', '상자 들기')]
    legacy_folder = tmp_path / 'G1_reference_20260917T055155_ec81e3'
    legacy_folder.mkdir()
    legacy_file = legacy_folder / 'project.json'
    legacy_file.write_text(json.dumps(project), encoding='utf-8')

    reopened = json.loads(legacy_file.read_text(encoding='utf-8'))
    prepare_saved_project(reopened, legacy_file.relative_to(tmp_path), legacy_file.stat().st_mtime)
    result = save_bundle(robot, reopened, fps=15, directory=tmp_path)

    assert reopened['name'] == ''
    assert reopened['name_mode'] == 'auto'
    assert datetime.fromisoformat(reopened['created_at']).astimezone(timezone.utc).isoformat().startswith('2026-09-17T05:51:55')
    assert result['display_name'] == '손 접근-상자 파지-상자 들기_20260917'
    assert result['reused'] is True
    assert not legacy_folder.exists()
    created_stamp = datetime.fromisoformat(reopened['created_at']).strftime('%Y%m%dT%H%M%S')
    assert Path(result['directory']).name.startswith(f'손_접근-상자_파지-상자_들기_{created_stamp}_')
    assert len(list(tmp_path.glob('*/project.json'))) == 1
    unchanged = save_bundle(robot, reopened, fps=15, directory=tmp_path)
    assert unchanged['directory'] == result['directory']
    assert len(list(tmp_path.glob('*/project.json'))) == 1


def test_legacy_project_receives_stable_identity_during_validation(robot):
    project = new_project(robot)
    del project['project_id']
    del project['created_at']

    validate_project(robot, project)

    assert len(project['project_id']) == 32
    assert project['created_at']

def test_import_kimodo_npz_and_g1_csv_as_editable_keyframes(robot, tmp_path):
    source = crouch_demo(robot)
    bundle = save_bundle(robot, source, fps=15, directory=tmp_path)
    expected = compile_motion(robot, source, fps=15)['qpos']

    for key, filename in (('npz_file', 'kimodo_walk.npz'), ('csv_file', 'kimodo_walk.csv')):
        imported = project_from_motion_bytes(robot, (tmp_path / bundle[key]).read_bytes(), filename, fps=15)
        clip = imported['keyframes'][0]
        actual = np.asarray(clip['samples'])
        assert imported['name'] == 'kimodo_walk'
        assert len(imported['keyframes']) == 1
        assert clip['pins'] == []
        assert clip['duration'] == pytest.approx((len(expected) - 1) / 15)
        assert np.allclose(actual, expected, atol=2e-6)
        assert imported['current_qpos'] == imported['keyframes'][0]['qpos']
        recompiled = compile_motion(robot, imported, fps=15)
        assert np.allclose(recompiled['qpos'], expected, atol=2e-6)
        assert np.allclose(np.diff(recompiled['time']), 1 / 15)

    resampled = compile_motion(robot, imported, fps=30)
    assert resampled['qpos'].shape == (121, 36)
    assert np.allclose(resampled['qpos'][[0, -1]], expected[[0, -1]], atol=2e-6)

    saved = save_bundle(robot, imported, fps=15, directory=tmp_path)
    reopened = json.loads((tmp_path / saved['project_file']).read_text())
    validate_project(robot, reopened)
    assert len(reopened['keyframes']) == 1
    assert len(reopened['keyframes'][0]['samples']) == len(expected)

    with pytest.raises(ValueError, match='36 qpos'):
        project_from_motion_bytes(robot, b'1,2,3\n', 'invalid.csv', fps=30)


def test_import_motion_api_accepts_csv(robot):
    async def request():
        csv = (','.join(str(value) for value in robot.home) + '\n').encode()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            return await client.post('/api/import-motion', params={'filename': 'single.csv', 'fps': 30},
                                     content=csv, headers={'Content-Type': 'application/octet-stream'})

    response = asyncio.run(request())
    assert response.status_code == 200, response.text
    imported = response.json()
    assert imported['name'] == 'single'
    assert len(imported['keyframes']) == 1
    assert imported['keyframes'][0]['duration'] == pytest.approx(1 / 30)
    assert len(imported['keyframes'][0]['samples']) == 1


def test_saved_projects_support_bundle_folders(tmp_path, monkeypatch):
    bundle = tmp_path / 'motions' / 'walk_001'
    bundle.mkdir(parents=True)
    project = bundle / 'project.json'
    project.write_text('{}')
    monkeypatch.setattr(server, 'ROOT', tmp_path)

    assert server.saved() == ['walk_001/project.json']
    response = server.download('walk_001/project.json')
    assert Path(response.path) == project
    with pytest.raises(HTTPException):
        server.download('../project.json')


def test_open_saved_project_restores_legacy_folder_identity(robot, tmp_path, monkeypatch):
    bundle = tmp_path / 'motions' / 'G1_reference_20260917T055155_ec81e3'
    bundle.mkdir(parents=True)
    project = new_project(robot, 'G1 reference')
    for field in ('name_mode', 'project_id', 'created_at'):
        project.pop(field)
    (bundle / 'project.json').write_text(json.dumps(project), encoding='utf-8')
    monkeypatch.setattr(server, 'ROOT', tmp_path)

    opened = server.open_saved_project('G1_reference_20260917T055155_ec81e3/project.json')

    assert opened['name'] == ''
    assert opened['name_mode'] == 'auto'
    assert datetime.fromisoformat(opened['created_at']).astimezone(timezone.utc).isoformat().startswith('2026-09-17T05:51:55')
    assert len(opened['project_id']) == 32


def test_invalid_projects_and_pin_conflicts(robot):
    p = crouch_demo(robot)
    p['model_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='fingerprint'):
        validate_project(robot, p)
    p = crouch_demo(robot)
    p['keyframes'][1]['qpos'][0] += .1
    with pytest.raises(ValueError, match='pinned'):
        compile_motion(robot, p)
    p = crouch_demo(robot)
    p['keyframes'][1]['qpos'][10] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        validate_project(robot, p)


def test_api_init_solve_and_bad_inputs(robot):
    with TestClient(app) as client:
        assert client.get('/api/health').json()['nq'] == 36
        init = client.get('/api/init').json()
        assert len(init['state']['handles']) == 45
        assert set(init['joint_names']) <= init['state']['handles'].keys()
        target = robot.home[:3].copy(); target[2] -= .1
        payload = {'qpos': robot.home.tolist(), 'anchor': robot.home.tolist(), 'focus': 'pelvis', 'target': target.tolist(), 'pins': list(FEET)}
        response = client.post('/api/solve', json=payload)
        assert response.status_code == 200, response.text
        assert response.json()['solver']['converged']
        payload['focus'] = 'left_foot'
        assert client.post('/api/solve', json=payload).status_code == 422
        assert client.post('/api/pose', json={'qpos': [0]}).status_code == 422
        assert client.get('/api/files/no_such_file.npz').status_code == 404


def _counting_solver(robot, monkeypatch):
    calls = []
    original = robot.solve

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(robot, 'solve', counting)
    return calls


def test_compile_motion_reuses_result_until_the_project_changes(robot, monkeypatch):
    project = crouch_demo(robot)
    calls = _counting_solver(robot, monkeypatch)
    first = compile_motion(robot, json.loads(json.dumps(project)), fps=30)
    solved = len(calls)
    second = compile_motion(robot, json.loads(json.dumps(project)), fps=30)
    assert len(calls) == solved
    np.testing.assert_array_equal(first['qpos'], second['qpos'])
    second['qpos'][:] = 0  # callers get their own copy of the cached arrays
    np.testing.assert_array_equal(compile_motion(robot, json.loads(json.dumps(project)), fps=30)['qpos'], first['qpos'])
    changed = json.loads(json.dumps(project))
    changed['keyframes'][-1]['duration'] += .5
    compile_motion(robot, changed, fps=30)
    assert len(calls) > solved
    compile_motion(robot, json.loads(json.dumps(project)), fps=20)
    assert len(calls) > solved


def test_hold_between_identical_pinned_keyframes_needs_no_ik(robot, monkeypatch):
    project = new_project(robot)
    project['keyframes'].append({**json.loads(json.dumps(project['keyframes'][0])), 'name': 'Hold', 'duration': 1.})
    calls = _counting_solver(robot, monkeypatch)
    motion = compile_motion(robot, project, fps=30)
    assert not calls
    np.testing.assert_allclose(motion['qpos'], np.tile(robot.validate_q(project['keyframes'][0]['qpos']), (len(motion['qpos']), 1)), atol=1e-9)
    assert motion['max_pin_error_mm'] < .01


def test_parallel_segment_compile_matches_sequential_and_reports_errors(robot, monkeypatch):
    from motioncreator import motion
    project = crouch_demo(robot)
    project['keyframes'].append({**json.loads(json.dumps(project['keyframes'][1])), 'name': 'Crouch again', 'duration': 1.})
    monkeypatch.setattr(motion, '_MIN_PARALLEL_SEGMENTS', 1)
    motion._COMPILE_CACHE.clear()
    sequential = compile_motion(robot, json.loads(json.dumps(project)), fps=20)
    assert motion.enable_parallel_compile(robot, workers=2) == 2
    try:
        motion._COMPILE_CACHE.clear()
        parallel = compile_motion(robot, json.loads(json.dumps(project)), fps=20)
        np.testing.assert_array_equal(parallel['qpos'], sequential['qpos'])
        np.testing.assert_array_equal(parallel['time'], sequential['time'])
        assert parallel['max_pin_error_mm'] == sequential['max_pin_error_mm']
        broken = json.loads(json.dumps(project))
        broken['keyframes'][2]['qpos'] = robot.home.tolist()
        broken['keyframes'][2]['qpos'][7] += .3  # left hip pitch moves the pinned left foot
        with pytest.raises(ValueError, match=r'keyframe [12] → [23]'):
            compile_motion(robot, broken, fps=20)
    finally:
        _pool, motion._PARALLEL['pool'], motion._PARALLEL['robot'] = motion._PARALLEL['pool'], None, None
        _pool.shutdown()
        motion._COMPILE_CACHE.clear()
