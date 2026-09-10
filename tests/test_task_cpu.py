"""Actual released CPU models: no mocks in policy/ARDY smoke checks."""
import importlib.util
import numpy as np
import mujoco
import pytest
from motioncreator.robot import Robot,FEET,ROOT
from motioncreator.tasks import TaskSpec
from motioncreator.task_physics import build_scene
from motioncreator.sonic import SonicCPU,Reference,DEFAULT,KP,KD,ORDER,INVERSE

pytestmark=pytest.mark.skipif(importlib.util.find_spec('onnxruntime') is None,reason='Run scripts/test-task-cpu.sh in the separate CPU environment')


def test_sonic_named_mapping_and_reference_hold():
    np.testing.assert_array_equal(np.arange(29)[ORDER][INVERSE],np.arange(29))
    r=Robot();ref=Reference(np.array([0.,1.]),np.tile(r.home,(2,1)),r.model)
    q,v=ref.sample([0.,.5,4.]);np.testing.assert_allclose(q,np.tile(r.home,(3,1)));assert np.max(np.abs(v))==0


def test_actual_sonic_cpu_keeps_free_g1_standing_five_seconds():
    r=Robot();m,_=build_scene(TaskSpec());d=mujoco.MjData(m)
    q=r.home.copy();q[7:]=DEFAULT
    q[2]-=min(r.point(r.data(q),k)[0][2] for k in FEET)
    d.qpos[:36]=q;mujoco.mj_forward(m,d)
    ref=Reference(np.array([0.,6.]),np.tile(q,(2,1)),r.model);p=SonicCPU()
    for i in range(250):
        target=p.action(d.qpos[:36],d.qvel[:35],ref,i*.02)
        for _ in range(10):
            d.ctrl[:]=np.clip(KP*(target-d.qpos[7:36])-KD*d.qvel[6:35],m.jnt_actfrcrange[1:30,0],m.jnt_actfrcrange[1:30,1])
            mujoco.mj_step(m,d)
        assert d.qpos[2]>.65
        assert d.xmat[m.body('pelvis').id].reshape(3,3)[2,2]>.95
    assert np.linalg.norm(d.qpos[:2]-q[:2])<.08
    assert np.all(d.xfrc_applied==0) and m.neq==0


def test_ardy_converter_matches_editor_fk_and_roundtrips():
    from ardy.skeleton import G1Skeleton34
    from ardy.exports.mujoco import MujocoQposConverter
    from motioncreator.task_generation import pose_to_ardy,C
    sk=G1Skeleton34();c=MujocoQposConverter(sk);r=Robot()
    q=r.home.copy();q[7+r.names.index('waist_yaw_joint')]=.23
    q[7+r.names.index('left_shoulder_roll_joint')]=.38
    local,root=pose_to_ardy(q,c)
    np.testing.assert_allclose(c.to_qpos(local[None],root[None]).numpy()[0,0],q,atol=2e-5)
    gr,gp,_=sk.fk(local,root);d=r.data(q)
    for side in ['left','right']:
        i=sk.bone_index[side+'_wrist_yaw_skel'];b=r.model.body(side+'_wrist_yaw_link').id
        np.testing.assert_allclose(gp[0,i].numpy(),C@d.xpos[b],atol=3e-5)
        np.testing.assert_allclose(gr[0,i].numpy(),C@d.xmat[b].reshape(3,3)@C.T,atol=3e-5)


def test_ardy_heading_has_same_sign_as_mujoco_yaw():
    import torch
    from scipy.spatial.transform import Rotation
    from ardy.skeleton import G1Skeleton34
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.motion_rep.tools import compute_heading_angle
    from motioncreator.task_generation import pose_to_ardy
    sk=G1Skeleton34();c=MujocoQposConverter(sk);q=Robot().home.copy()
    q[3:7]=Rotation.from_euler('z',.7).as_quat()[[3,0,1,2]]
    local,root=pose_to_ardy(q,c);_,gp,_=sk.fk(local,root)
    assert torch.allclose(compute_heading_angle(gp[None],sk),torch.tensor([[.7]]),atol=1e-5)
