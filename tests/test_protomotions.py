"""Integration tests against the pinned, unmodified upstream APIs, CPU only."""
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation
from motioncreator.protomotions_bridge import convert_reference, UPSTREAM
from motioncreator.robot import Robot
from motioncreator.motion import save_bundle, new_project

torch = pytest.importorskip('torch', reason='Run scripts/test-protomotions.sh in the separate conversion environment')
sys.path.insert(0, str(UPSTREAM))
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from deployment.motion_utils import MotionPlayer


@pytest.fixture
def source(tmp_path):
    r = Robot(); p = new_project(r)
    p['keyframes'][0]['pins'] = []
    q = r.home.copy(); q[:3] += [.08, .04, .05]
    q[3:7] = Rotation.from_euler('xyz', [.04, .06, .15]).as_quat()[[3,0,1,2]]
    p['keyframes'].append({'name':'Translate and rotate','duration':1.,'qpos':q.tolist(),'pins':[]})
    b = save_bundle(r, p, fps=30, directory=tmp_path)
    return tmp_path/b['files'][1]


def test_native_fk_motionlib_and_deployment_resampling(source):
    result = convert_reference(source)
    report = result['report']; path = source.with_suffix('.motion')
    state = torch.load(path, weights_only=False)
    data = np.load(source, allow_pickle=False)
    assert state['rigid_body_rot'].shape[2] == 4
    assert 'left_rubber_hand' in report['body_names'] and 'head' in report['body_names']
    assert np.allclose(state['rigid_body_pos'][:,0], data['root_pos'], atol=1e-6)
    assert np.allclose(np.abs(np.sum(state['rigid_body_rot'][:,0].numpy()*data['root_quat_wxyz'][:,[1,2,3,0]], axis=-1)), 1, atol=1e-6)
    # Independently compare native ProtoMotions FK against compiled MuJoCo target FK.
    target = UPSTREAM / report['upstream']['target_mjcf']
    tree = ET.parse(target); root=tree.getroot()
    root.find('compiler').set('meshdir', str((target.parent/'../mesh/G1').resolve()))
    # Visual meshes are Git LFS assets; FK only needs body transforms/inertials.
    # Keep the target's primitive collision geoms and exact joints unchanged.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'mesh' or (child.tag == 'geom' and child.get('mesh')):
                parent.remove(child)
    for sensor in root.findall('sensor'): root.remove(sensor)
    world=root.find('worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05')
    pelvis=world.find('body')
    if pelvis.find('freejoint') is None and pelvis.find("joint[@type='free']") is None:
        ET.SubElement(pelvis, 'freejoint')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode')); d=mujoco.MjData(m)
    for frame in [0, 13, 30]:
        d.qpos[:7]=data['qpos'][frame,:7]
        for i,name in enumerate(report['joint_names']): d.qpos[m.joint(name).qposadr[0]]=state['dof_pos'][frame,i]
        mujoco.mj_forward(m,d)
        for i,name in enumerate(report['body_names']):
            bid=m.body(name).id
            assert np.allclose(state['rigid_body_pos'][frame,i], d.xpos[bid], atol=2e-6), name
            q=state['rigid_body_rot'][frame,i].numpy()
            assert abs(np.dot(q,d.xquat[bid][[1,2,3,0]])) > 1-2e-6
    lib=MotionLib(MotionLibConfig(motion_file=str(source.with_suffix('.pt'))), device='cpu')
    assert lib.num_motions() == 1
    assert float(lib.motion_lengths[0]) == 1.
    # Actual deployment helper performs 30 -> 50 Hz reference interpolation.
    player=MotionPlayer(str(path), control_dt=.02)
    assert player.num_dofs==29 and player.num_bodies==len(report['body_names'])
    final=player.get_state_at_frame(player.total_frames-1)
    assert np.allclose(final['dof_pos'], state['dof_pos'][-1], atol=1e-6)
    assert player.get_future_references(0,[1,5])['body_rot'].shape == (2, len(report['body_names']), 4)
    assert not report['validation']['controller_tracking_checked']


def test_name_reordering_and_authored_contacts(source, tmp_path):
    data=dict(np.load(source, allow_pickle=False)); reverse=np.arange(28,-1,-1)
    data['joint_names']=data['joint_names'][reverse]
    data['dof_pos']=data['dof_pos'][:,reverse]
    data['qpos'][:,7:]=data['dof_pos']
    data['contacts'][:]=True
    reordered=tmp_path/'reordered.npz'; np.savez(reordered, **data)
    output=convert_reference(reordered)
    state=torch.load(reordered.with_suffix('.motion'), weights_only=False)
    assert np.allclose(state['dof_pos'], data['dof_pos'][:,reverse])
    assert state['rigid_body_contacts'].sum().item() == len(data['time'])*2
    assert output['report']['joint_permutation_from_source']==reverse.tolist()
    with pytest.raises(ValueError, match='already exists'): convert_reference(reordered)


def test_single_frame_and_wrong_model_are_rejected(source, tmp_path):
    r=Robot(); b=save_bundle(r,new_project(r),directory=tmp_path)
    with pytest.raises(ValueError, match='2프레임'): convert_reference(tmp_path/b['files'][1])
    d=dict(np.load(source,allow_pickle=False)); meta=json.loads(str(d['metadata_json']))
    meta['model_sha256']='unknown'; d['metadata_json']=np.array(json.dumps(meta))
    invalid=tmp_path/'invalid.npz'; np.savez(invalid,**d)
    with pytest.raises(ValueError, match='Source G1 model'): convert_reference(invalid)


def test_previous_known_model_fingerprint_still_loads(source,tmp_path):
    from motioncreator.protomotions_bridge import PROFILE_PATH
    profile=json.loads(PROFILE_PATH.read_text())
    data=dict(np.load(source,allow_pickle=False)); meta=json.loads(str(data['metadata_json']))
    meta['model_sha256']=profile['legacy_source_mjcf_sha256'][0]
    data['metadata_json']=np.array(json.dumps(meta)); legacy=tmp_path/'legacy.npz';np.savez(legacy,**data)
    converted=convert_reference(legacy)
    assert converted['report']['source_model_sha256']==meta['model_sha256']
