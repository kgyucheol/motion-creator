"""IK layout preview and actual numeric-constraint ARDY inference, CPU only."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .robot import Robot,ROOT,HANDLES,FEET
from .tasks import TaskSpec
from .task_physics import build_scene,hand_surface_offsets
from .sonic import Reference

C=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=float)

def seed_poses(plan,progress=lambda *a:None):
    robot=Robot(); spec=TaskSpec.model_validate(plan['spec']); model,_=build_scene(spec)
    offsets=hand_surface_offsets(model); q=np.array(plan['start_qpos']); poses=[q.copy()]; diagnostics=[]
    for i,phase in enumerate(plan['phases']):
        progress(i/len(plan['phases']),phase['label'])
        anchor=q.copy()
        yaw=Rotation.from_euler('z',phase['root_yaw'])
        if i in (0,6,10):
            q[:3]=phase['root_position']; q[3:7]=yaw.as_quat()[[3,0,1,2]]; anchor=q.copy()
        rot=Rotation.from_quat(phase['hand_rotation_xyzw']).as_matrix()
        targets={'pelvis':phase['root_position']}
        if i not in (0,10):
            for side in ('left','right'):
                wrist=np.array(phase['surface_targets'][side])-rot@offsets[side]
                targets[side+'_hand']=(wrist+rot@np.array(HANDLES[side+'_hand'][1])).tolist()
        orientations={'pelvis':yaw.as_quat().tolist()}
        if i not in (0,10): orientations.update({side+'_hand':phase['hand_rotation_xyzw'] for side in offsets})
        q,info=robot.solve(q,anchor,pins=FEET,mode='free',resistance=0,selected_targets=targets,
                           orientation_targets=orientations,max_nfev=100,posture_weight=.02)
        poses.append(q.copy()); diagnostics.append({'phase':phase['id'],**info})
    return np.array([0.]+[p['end'] for p in plan['phases']]),np.array(poses),diagnostics,offsets

def write_reference(folder,t,q,source,task_sha):
    robot=Robot(); ref=Reference(t,q,robot.model); v=ref.v
    fps=1/np.diff(t)[0]
    if not np.allclose(np.diff(t),1/fps): raise ValueError('Uniform motion time required')
    d=mujoco.MjData(robot.model); pos=[]; quat=[]; bv=[]; av=[]
    jp,jr=np.zeros((3,robot.model.nv)),np.zeros((3,robot.model.nv))
    for pose,vel in zip(q,v):
        d.qpos[:]=pose; mujoco.mj_forward(robot.model,d); pos.append(d.xpos[1:].copy()); quat.append(d.xquat[1:].copy())
        linear=[]; angular=[]
        for b in range(1,robot.model.nbody):
            mujoco.mj_jacBody(robot.model,d,jp,jr,b); linear.append(jp@vel); angular.append(jr@vel)
        bv.append(linear); av.append(angular)
    limits=robot.model.jnt_range[1:]
    excess=np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1])
    meta={'reference_schema':'motioncreator.reference.v2','source':source,'task_sha256':task_sha,
          'model_sha256':robot.fingerprint,'joint_names':robot.names,'fps':float(fps),'quaternion_order':'wxyz',
          'coordinate_system':'right-handed, +X forward, +Y left, +Z up','units':{'position':'m','angle':'rad','time':'s'},
          'validation_status':'not_run','max_joint_limit_excess_rad':float(max(0,excess.max())),
          'body_pose_frame':'world; body frame origins, not centers of mass','body_velocity_frame':'world',
          'joint_position_semantics':'absolute hinge angles in radians; not policy actions','control_binding':None}
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'reference.npz',time=t,qpos=q,qvel=v,qacc=np.gradient(v,t,axis=0),fps=fps,
                        root_pos=q[:,:3],root_quat_wxyz=q[:,3:7],root_lin_vel_world=v[:,:3],root_ang_vel_world=np.array(av)[:,0],
                        dof_pos=q[:,7:],dof_vel=v[:,6:],joint_names=np.array(robot.names),body_pos=pos,body_quat_wxyz=quat,
                        body_lin_vel_world=bv,body_ang_vel_world=av,body_names=np.array([robot.model.body(i).name for i in range(1,robot.model.nbody)]),
                        body_parent_indices=robot.model.body_parentid[1:]-1,metadata_json=json.dumps(meta))
    (folder/'reference.metadata.json').write_text(json.dumps(meta,indent=2))
    return meta

def ik_preview(plan,folder,progress=lambda *a:None):
    t,q,diag,offsets=seed_poses(plan,progress)
    ref=Reference(t,q,Robot().model); dense_t=np.arange(round(t[-1]*25)+1)/25; dense_q,_=ref.sample(dense_t)
    meta=write_reference(folder,dense_t,dense_q,'IK layout preview (kinematic; navigation legs are not a gait)',plan['task_sha256'])
    (folder/'ik-diagnostics.json').write_text(json.dumps(diag,indent=2))
    return {'source':'ik_preview','reference':meta,'diagnostics':diag}

def pose_to_ardy(q,converter):
    """Inverse of official converter's hinge extraction, including fixed local rotations."""
    import torch
    q=np.atleast_2d(q); n=len(q); sk=converter.skeleton
    local=np.tile(np.eye(3),(n,sk.nbjoints,1,1))
    local[:,0]=C@Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix()@C.T
    for j,idx in enumerate(converter._mujoco_indices_to_ardy_indices.tolist()):
        axis=converter._mujoco_joint_axis_values_ardy_space[j].numpy()
        local[:,idx]=Rotation.from_rotvec(q[:,7+j,None]*axis).as_matrix()
    local=converter._rot_offsets_f2q.numpy().transpose(0,2,1)@local
    return torch.tensor(local,dtype=torch.float32),torch.tensor(q[:,:3]@C.T,dtype=torch.float32)

def generate_ardy(plan,folder,progress=lambda *a:None,cancel=lambda:False):
    import sys
    sys.path.insert(0,str(ROOT/'external/ardy-src'))
    import torch
    from ardy.model import load_model
    from ardy.constraints import FullBodyConstraintSet,EndEffectorConstraintSet,Root2DConstraintSet
    from ardy.exports.mujoco import MujocoQposConverter
    torch.set_num_threads(2); torch.manual_seed(plan['spec']['seed'])
    from .task_provenance import verify_assets
    assets=verify_assets('ardy')
    progress(.02,'ARDY G1 CPU 모델 로딩')
    model=load_model('g1',text_encoder=False,device='cpu',checkpoints_dir=str(ROOT/'external/task-models/ardy'))
    converter=MujocoQposConverter(model.skeleton)
    spec=TaskSpec.model_validate(plan['spec']); fps=model.motion_rep.fps
    if spec.diffusion_steps>model.diffusion.num_base_steps: raise ValueError(f'이 ARDY 모델은 확산 단계를 최대 {model.diffusion.num_base_steps}회 지원합니다.')
    t,seeds,diag,offsets=seed_poses(plan,lambda v,msg:progress(.08+.1*v,msg))
    if cancel(): raise InterruptedError('cancelled')
    local,root=pose_to_ardy(seeds,converter)
    back=converter.to_qpos(local[None],root[None]).numpy()[0]
    if not np.allclose(back,seeds,atol=2e-5): raise ValueError('ARDY/MuJoCo skeleton roundtrip failed')
    grots,gpos,_=model.skeleton.fk(local,root)
    indices=torch.tensor(np.rint(t*fps).astype('int64'))
    constraints=[FullBodyConstraintSet(model.skeleton,indices[:1],gpos[:1],grots[:1])]
    # Exact object targets override the IK seed. Both wrist and terminal hand point shift together.
    for i,phase in enumerate(plan['phases'],start=1):
        gp=gpos[i:i+1].clone(); gr=grots[i:i+1].clone()
        gp[:,0]=torch.tensor(np.array(phase['root_position'])@C.T,dtype=torch.float32)
        rot=Rotation.from_quat(phase['hand_rotation_xyzw']).as_matrix()
        parts=['Hips']
        if phase['id'] not in ('approach','retreat'):
            parts+=['LeftHand','RightHand']
            for side in ('left','right'):
                wrist_idx=model.skeleton.bone_index[side+'_wrist_yaw_skel']; end_idx=model.skeleton.bone_index[side+'_hand_roll_skel']
                wrist=np.array(phase['surface_targets'][side])-rot@offsets[side]
                # Preserve the skeleton's rigid terminal offset at the requested wrist rotation.
                original_rot=gr[0,wrist_idx].numpy(); delta_local=original_rot.T@(gp[0,end_idx]-gp[0,wrist_idx]).numpy()
                desired_rot=C@rot@C.T
                gp[0,wrist_idx]=torch.tensor(C@wrist,dtype=torch.float32)
                gp[0,end_idx]=gp[0,wrist_idx]+torch.tensor(desired_rot@delta_local,dtype=torch.float32)
                gr[0,wrist_idx]=torch.tensor(desired_rot,dtype=torch.float32)
        if phase['id'] in ('crouch','pregrasp','grasp','lift','stand','lower','place','release'): parts+=['LeftFoot','RightFoot']
        constraints.append(EndEffectorConstraintSet(model.skeleton,indices[i:i+1],gp,gr,None,joint_names=parts))
    # Keep the path spatially anchored throughout the sequence; hand/root-height constraints at task landmarks.
    n=round(plan['duration']*fps)+1; dense_t=np.arange(n)/fps
    xy=np.stack([np.interp(dense_t,t,seeds[:,j]) for j in (1,0)],-1)
    # The basis change maps MuJoCo +Z yaw to ARDY +Y heading with the same sign.
    yaws=np.unwrap(np.r_[Rotation.from_quat(seeds[0,[4,5,6,3]]).as_euler('xyz')[2],[p['root_yaw'] for p in plan['phases']]])
    headings=np.interp(dense_t,t,yaws)
    # Endpoint EE constraints already set root2d: avoid duplicate indexed writes there.
    mask=np.ones(n,bool); mask[indices.numpy()]=False; interior=np.where(mask)[0]
    constraints.append(Root2DConstraintSet(model.skeleton,torch.tensor(interior),torch.tensor(xy[interior],dtype=torch.float32),torch.tensor(headings[interior],dtype=torch.float32)))
    lengths=torch.tensor([n]); observed,mmask=model.motion_rep.create_conditions_from_constraints_batched(constraints,lengths,to_normalize=True,device='cpu')
    np.savez_compressed(folder/'ardy-conditions.npz',observed_motion=observed.numpy(),motion_mask=mmask.numpy())
    (folder/'ik-diagnostics.json').write_text(json.dumps(diag,indent=2))
    progress(.22,'ARDY 수치 제약 기반 생성 중 (CPU)')
    completed_steps=0
    total_steps=int(np.ceil(n/model.gen_horizon_len))*spec.diffusion_steps
    def bar(iterable,**kwargs):
        nonlocal completed_steps
        for item in iterable:
            if cancel(): raise InterruptedError('cancelled')
            progress(.22+.65*min(1.,completed_steps/max(total_steps,1)),'ARDY 생성 중')
            completed_steps+=1
            yield item
    with torch.inference_mode():
        motion=model([''],n,num_denoising_steps=spec.diffusion_steps,pad_mask=torch.ones((1,n),dtype=torch.bool),
                     first_heading_angle=torch.tensor([yaws[0]],dtype=torch.float32),motion_mask=mmask,observed_motion=observed,
                     cfg_weight=(0.,2.),text_feat=torch.zeros((1,1,4096)),text_pad_mask=torch.ones((1,1),dtype=torch.bool),
                     crop_history_length=52,progress_bar=bar)
        output=model.motion_rep.inverse(motion,is_normalized=True)
        q=converter.dict_to_qpos(output)[0,:n]
    if not np.isfinite(q).all(): raise ValueError('ARDY returned nonfinite motion')
    progress(.92,'참조 모션과 목표 오차 저장')
    meta=write_reference(folder,dense_t,q,'ARDY-G1-RP-25FPS-Horizon52 / CPU / numeric constraints / no text encoder',plan['task_sha256'])
    # Quantify whether the requested contact surfaces were actually reached; never equate generation with feasibility.
    robot=Robot(); errors=[]
    for phase in plan['phases']:
        if phase['id'] in ('approach','retreat'): continue
        d=robot.data(q[min(n-1,round(phase['end']*fps))]); e={}
        for side in ('left','right'):
            b=robot.ids[side+'_hand']; surface=d.xpos[b]+d.xmat[b].reshape(3,3)@offsets[side]
            e[side+'_surface_error_m']=float(np.linalg.norm(surface-phase['surface_targets'][side]))
        errors.append({'phase':phase['id'],**e})
    (folder/'constraint-errors.json').write_text(json.dumps(errors,indent=2))
    return {'source':'ardy','reference':meta,'constraint_errors':errors,'inference_assets':assets}
