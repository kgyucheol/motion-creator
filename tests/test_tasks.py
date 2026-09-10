"""Contracts that prevent perception leakage, false success and stale validation."""
import json
import numpy as np
import mujoco
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation
from motioncreator.tasks import TaskSpec,Pose,TaskFSM,plan_task
from motioncreator.task_physics import build_scene,contact_metrics,hand_surface_offsets
from motioncreator.robot import Robot
from motioncreator.task_generation import write_reference
from motioncreator.reference import load_reference
from motioncreator import task_jobs


def test_planner_uses_perception_only_but_validation_binds_truth_and_start():
    a=TaskSpec(); before=plan_task(a)
    a.actual.size[1]=.70; a.mass_kg=8; a.hand_friction=.2
    after=plan_task(a)
    assert before['phases']==after['phases']
    assert before['task_sha256']!=after['task_sha256']
    a.perceived.size[1]=.51
    assert plan_task(a)['target_surface_gap_m']==.51
    a.squeeze_m=.02
    assert plan_task(a)['target_surface_gap_m']==.49
    q=Robot().home.copy();q[0]+=.1
    assert plan_task(a,q)['task_sha256']!=plan_task(a)['task_sha256']


def test_pose_transforms_rotate_the_grasp_and_standoff():
    s=TaskSpec();s.perceived.pose=Pose(position=[0.,.6,.2],quaternion_xyzw=Rotation.from_euler('z',90,degrees=True).as_quat().tolist())
    p=plan_task(s); g=p['phases'][3]
    difference=np.array(g['surface_targets']['left'])-g['surface_targets']['right']
    np.testing.assert_allclose(difference,[-.49,0,0],atol=1e-8)
    assert abs(g['root_position'][0])<1e-8
    assert np.isclose(g['root_position'][1],.6-s.standoff_m)
    with pytest.raises(ValidationError): Pose(quaternion_xyzw=[0,0,0,2])
    with pytest.raises(ValidationError): TaskSpec(hand_friction=float('nan'))


def test_fsm_waits_for_measured_guard_times_out_and_requires_dwell():
    plan=plan_task(TaskSpec(phase_seconds=.4,timeout_seconds=.5)); f=TaskFSM(plan)
    for _ in range(25):f.update(.02,{})
    assert f.index==0 and f.reference_time==.4 and f.status=='running'
    for _ in range(6):f.update(.02,{'root_arrival':True})
    assert f.index==0
    f.update(.02,{'root_arrival':False})
    assert f.dwell==0
    for _ in range(11):f.update(.02,{'root_arrival':True})
    assert f.index==1
    for _ in range(46):f.update(.02,{})
    assert f.status=='failed' and f.reason=='timeout:pose_arrival'
    f=TaskFSM(plan);f.update(.02,{'fallen':True,'root_arrival':True})
    assert f.status=='failed' and f.index==0


def test_physics_free_box_and_explicit_friction_and_no_robot_asset_mutation():
    r=Robot(); original=r.fingerprint
    spec=TaskSpec(hand_friction=0.,floor_friction=.9)
    m,xml=build_scene(spec)
    assert m.nq==43 and m.nu==29 and m.neq==0
    assert m.jnt_type[m.joint('task_box_free').id]==mujoco.mjtJoint.mjJNT_FREE
    for s in ['left','right']:
        gid=m.geom(s+'_grip').id
        assert m.geom_contype[gid] and m.geom_conaffinity[gid]
        idx=np.where(((m.pair_geom1==gid)&(m.pair_geom2==m.geom('task_box_geom').id))|((m.pair_geom2==gid)&(m.pair_geom1==m.geom('task_box_geom').id)))[0][0]
        np.testing.assert_array_equal(m.pair_friction[idx,:2],[0.,0.])
    assert np.all(m.dof_armature[6:35]>0)
    assert Robot().fingerprint==original
    offsets=hand_surface_offsets(m)
    assert all(np.isfinite(v).all() for v in offsets.values())


def test_contact_wrench_support_has_correct_sign_and_no_fictitious_hand_force():
    s=TaskSpec(mass_kg=2.);s.actual.pose.position=[3.,0.,.20]
    m,_=build_scene(s);d=mujoco.MjData(m)
    for _ in range(500):mujoco.mj_step(m,d)
    result=contact_metrics(m,d)
    assert abs(result['support_n']-2*9.81)<.05
    assert all(h['normal_n']==0 and h['tangent_n']==0 for h in result['hands'].values())
    assert np.all(d.xfrc_applied==0) and np.all(d.qfrc_applied==0)


def test_task_npz_uses_existing_reference_contract(tmp_path):
    r=Robot();t=np.arange(26)/25;q=np.tile(r.home,(26,1))
    write_reference(tmp_path,t,q,'test','hash')
    a,meta=load_reference(tmp_path/'reference.npz')
    assert a['qpos'].shape==(26,36) and a['body_pos'].shape[0]==26
    assert meta['validation_status']=='not_run' and meta['source']=='test'


def test_task_presets_persist_and_stale_generation_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(task_jobs,'TASKS',tmp_path)
    first=task_jobs.save_task(TaskSpec())
    task=task_jobs.get_task(first['id']);assert task==first
    assert len(task_jobs.list_tasks())==1
    rid='abcdef012345';folder=task_jobs.run_folder(first['id'],rid);folder.mkdir(parents=True)
    (folder/'request.json').write_text(json.dumps({'plan':first['plan']}))
    (folder/'reference.npz').touch()
    changed=TaskSpec(hand_friction=.4);task_jobs.save_task(changed,task_id=first['id'])
    with pytest.raises(ValueError,match='변경'):task_jobs.start_run(first['id'],'simulate',reference_run=rid)
    with pytest.raises(ValueError):task_jobs.get_task('../outside')


def test_friction_grasp_fixture_holds_only_with_friction():
    # Isolated two sliding pads, not a G1 success claim. Finite actuator force,
    # 51 cm free box, 49 cm commanded surface gap, and no box attachment.
    def trial(mu):
        xml=f'''<mujoco><option timestep=".002" integrator="implicitfast" cone="elliptic"/>
        <worldbody><geom name="floor" type="plane" size="3 3 .1"/>
        <geom name="task_pallet" type="box" pos="3 0 .1" size=".1 .1 .1"/>
        <body name="task_box" pos="0 0 1"><freejoint/><geom name="task_box_geom" type="box" size=".12 .255 .12" mass="1"/></body>
        <body name="left_wrist_yaw_link" pos="0 .275 1"><joint name="left" type="slide" axis="0 1 0" damping="10"/><geom name="left_grip" type="box" size=".12 .02 .20" mass="1"/></body>
        <body name="right_wrist_yaw_link" pos="0 -.275 1"><joint name="right" type="slide" axis="0 1 0" damping="10"/><geom name="right_grip" type="box" size=".12 .02 .20" mass="1"/></body></worldbody>
        <contact><pair geom1="left_grip" geom2="task_box_geom" condim="3" friction="{mu} {mu} 0 0 0"/>
        <pair geom1="right_grip" geom2="task_box_geom" condim="3" friction="{mu} {mu} 0 0 0"/></contact>
        <actuator><position joint="left" kp="1000" forcelimited="true" forcerange="-20 20"/>
        <position joint="right" kp="1000" forcelimited="true" forcerange="-20 20"/></actuator></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.ctrl[:]=[-.01,.01]
        for _ in range(300):mujoco.mj_step(m,d)
        assert m.neq==0 and np.all(d.xfrc_applied==0)
        return float(d.xpos[m.body('task_box').id,2]),contact_metrics(m,d)
    held,forces=trial(.8);dropped,_=trial(0.)
    assert held>.97 and dropped<.5
    assert forces['hands']['left']['normal_n']>5 and forces['hands']['right']['normal_n']>5
    # The box pushes hands outwards (+Y left / -Y right), and downwards.
    assert forces['hands']['left']['force_world_n'][1]>5
    assert forces['hands']['right']['force_world_n'][1]<-5
    down=sum(h['force_world_n'][2] for h in forces['hands'].values())
    assert abs(down+9.81)<.15


def test_interrupted_worker_is_reconciled_after_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(task_jobs,'TASKS',tmp_path)
    folder=task_jobs.run_folder('abcdef123456','abcdef654321');folder.mkdir(parents=True)
    task_jobs.atomic_json(folder/'status.json',{'id':'abcdef654321','task_id':'abcdef123456','status':'running','worker_pid':99999999})
    status=task_jobs.get_run('abcdef123456','abcdef654321')
    assert status['status']=='interrupted'
    assert json.loads((folder/'status.json').read_text())['status']=='interrupted'
