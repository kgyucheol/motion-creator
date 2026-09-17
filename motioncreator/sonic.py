"""CPU ONNX port of NVIDIA's original-release G1 observation path (not teleop).
Source: gear_sonic_deploy/src/g1/g1_deploy_onnx_ref; see integration provenance.
"""
from collections import deque
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .robot import ROOT

PARAMS=json.loads((ROOT/'integrations/sonic-parameters.json').read_text())
ORDER=np.array(PARAMS['mujoco_to_isaaclab'])
INVERSE=np.array(PARAMS['isaaclab_to_mujoco'])
DEFAULT=np.array(PARAMS['default_angles'])
KP=np.array(PARAMS['kp']); KD=np.array(PARAMS['kd'])
# Explicit original-release feature sizes, from C++ observation registry. Unused modes zero-filled.
ENCODER_DIMS=[('encoder_mode_4',4),('motion_joint_positions_10frame_step5',290),
 ('motion_joint_velocities_10frame_step5',290),('motion_root_z_position_10frame_step5',10),
 ('motion_root_z_position',1),('motion_anchor_orientation',6),('motion_anchor_orientation_10frame_step5',60),
 ('motion_joint_positions_lowerbody_10frame_step5',120),('motion_joint_velocities_lowerbody_10frame_step5',120),
 ('vr_3point_local_target',9),('vr_3point_local_orn_target',12),('smpl_joints_10frame_step1',720),
 ('smpl_anchor_orientation_10frame_step1',60),('motion_joint_positions_wrists_10frame_step1',60)]

class Reference:
    def __init__(self,t,q,model):
        import mujoco
        self.t=np.asarray(t); self.q=np.asarray(q)
        if len(t)<2 or self.q.shape!=(len(t),36) or not np.isfinite(q).all() or np.any(np.diff(t)<=0):
            raise ValueError('Finite, increasing reference with at least 2 frames is required')
        self.rot=Slerp(self.t,Rotation.from_quat(self.q[:,[4,5,6,3]]))
        self.v=np.zeros((len(t),35))
        for i in range(len(t)):
            a,b=max(0,i-1),min(len(t)-1,i+1)
            mujoco.mj_differentiatePos(model,self.v[i],t[b]-t[a],q[a],q[b])
    def sample(self,t):
        ts=np.atleast_1d(np.clip(t,self.t[0],self.t[-1]))
        q=np.stack([np.interp(ts,self.t,self.q[:,i]) for i in range(36)],-1)
        q[:,3:7]=self.rot(ts).as_quat()[:,[3,0,1,2]]
        vel=np.stack([np.interp(ts,self.t,self.v[:,i]) for i in range(35)],-1)
        vel[np.atleast_1d(t)>self.t[-1]]=0
        return q,vel

class SonicCPU:
    def __init__(self,threads=2):
        import onnxruntime as ort
        import yaml
        from .task_provenance import verify_assets
        self.asset_info=verify_assets('sonic')
        folder=ROOT/'external/task-models/sonic'
        config=yaml.safe_load((folder/'observation_config.yaml').read_text())
        names=[x['name'] for x in config['encoder']['encoder_observations'] if x.get('enabled',True)]
        if names!=[x[0] for x in ENCODER_DIMS]: raise ValueError('Unsupported SONIC observation configuration; use the pinned original release')
        expected=['token_state']+['his_'+k+'_10frame_step1' for k in ['base_angular_velocity','body_joint_positions','body_joint_velocities','last_actions','gravity_dir']]
        if [x['name'] for x in config['observations'] if x.get('enabled',True)]!=expected: raise ValueError('Unsupported SONIC decoder configuration')
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
        options.log_severity_level=3
        self.encoder=ort.InferenceSession(str(folder/'model_encoder.onnx'),options,providers=['CPUExecutionProvider'])
        self.decoder=ort.InferenceSession(str(folder/'model_decoder.onnx'),options,providers=['CPUExecutionProvider'])
        self.enc_size=sum(n for _,n in ENCODER_DIMS)
        for session,size in ((self.encoder,self.enc_size),(self.decoder,994)):
            if len(session.get_inputs())!=1 or session.get_inputs()[0].shape[-1]!=size:
                raise ValueError(f'Unexpected SONIC ONNX input shape: {session.get_inputs()[0].shape}; expected {size}')
        self.history=deque(maxlen=10); self.last_action=np.zeros(29)
    def reset(self):
        """Reset rollout state while retaining the loaded ONNX sessions."""
        self.history.clear(); self.last_action.fill(0)
    def action(self,q,qvel,ref,time,stop_at=None):
        rot=Rotation.from_quat(q[[4,5,6,3]]).as_matrix()
        # Free-joint rotational qvel is body-local in MuJoCo, matching the pelvis gyro.
        entry=[qvel[3:6].copy(),(q[7:36]-DEFAULT)[ORDER],qvel[6:35][ORDER],self.last_action.copy(),rot.T@np.array([0.,0.,-1.])]
        if not self.history:
            # Match the original StateLogger's zero-quaternion padding exactly.
            for _ in range(9):
                zero=[np.zeros_like(x) for x in entry]; zero[-1]=np.array([0.,0.,1.]); self.history.append(zero)
        self.history.append(entry)
        # Literal step5 at the C++ 50 Hz motion clock: 0, .10, ... .90 s lookahead.
        times=time+np.arange(10)*.10
        if stop_at is not None: times=np.minimum(times,stop_at)
        rq,rv=ref.sample(times)
        if stop_at is not None: rv[times>=stop_at]=0
        rref=Rotation.from_quat(rq[:,[4,5,6,3]]).as_matrix()
        features={'encoder_mode_4':np.array([0.,0,0,0]),'motion_joint_positions_10frame_step5':rq[:,7:36][:,ORDER],
                  'motion_joint_velocities_10frame_step5':rv[:,6:35][:,ORDER],
                  'motion_anchor_orientation_10frame_step5':(rot.T@rref)[:,:,:2]}
        enc=np.concatenate([np.asarray(features.get(k,np.zeros(n))).ravel() for k,n in ENCODER_DIMS]).astype('float32')[None]
        token=self.encoder.run(None,{self.encoder.get_inputs()[0].name:enc})[0].ravel()
        obs=np.concatenate([token]+[np.concatenate([h[i] for h in self.history]) for i in range(5)]).astype('float32')[None]
        action=self.decoder.run(None,{self.decoder.get_inputs()[0].name:obs})[0].ravel()
        if action.shape!=(29,) or not np.isfinite(action).all(): raise ValueError('SONIC returned invalid actions')
        self.last_action=action.copy()
        return DEFAULT+action[INVERSE]*np.array(PARAMS['action_scale'])
