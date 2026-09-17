"""Free-base, torque-actuated MuJoCo box test. No attachments or external support."""
import hashlib, json, time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .grip_geometry import (grip_pad_center, grip_pad_contact_anchor, grip_pad_half_size,
                            grip_pad_quaternion_wxyz)
from .hand_collision import physical_hand_geom_names
from .robot import MODEL_PATH, ROOT, Robot
from .tasks import TaskSpec, TaskFSM
from .sonic import KP,KD,Reference,SonicCPU

def numbers(a): return ' '.join(str(float(x)) for x in a)

def build_scene(spec:TaskSpec):
    root=ET.parse(MODEL_PATH).getroot()
    root.find('compiler').set('meshdir',str(ROOT/'assets/g1/meshes'))
    opt=root.find('option')
    if opt is None: opt=ET.SubElement(root,'option')
    opt.attrib.update(timestep='0.002',integrator='implicitfast',cone='elliptic',iterations='80',gravity='0 0 -9.81')
    # Retain editor inertias/limits. The colored alignment pads are visual-only;
    # explicit object contacts use the actual wrist and hand meshes.
    # Effective motor rotor inertia from the same released SONIC stiffness/armature constants.
    for i,name in enumerate(Robot().names):
        root.find(f".//joint[@name='{name}']").set('armature',str(float(KP[i]/(20*np.pi)**2)))
    contact=root.find('contact')
    if contact is None: contact=ET.SubElement(root,'contact')
    for side in ('left','right'):
        body=root.find(f".//body[@name='{side}_wrist_yaw_link']")
        physical_geoms=physical_hand_geom_names(root,side)
        ET.SubElement(body,'geom',name=side+'_grip',type='box',pos=numbers(grip_pad_center(side)),
                      quat=numbers(grip_pad_quaternion_wxyz(side)), size=numbers(grip_pad_half_size()),
                      contype='0',conaffinity='0',group='3',density='0',
                      rgba='.15 .9 .72 .45' if side=='left' else '1 .62 .25 .45')
        for geom_name in physical_geoms:
            ET.SubElement(contact,'pair',geom1=geom_name,geom2='task_box_geom',condim='3',
                          friction=numbers([spec.hand_friction]*2+[0,0,0]),
                          solref='.01 1',solimp='.95 .99 .001')
    floor=root.find(".//geom[@name='floor']"); floor.set('friction',numbers([spec.floor_friction,.005,.0001])); floor.set('priority','1')
    world=root.find('worldbody')
    box=ET.SubElement(world,'body',name='task_box',pos=numbers(spec.actual.pose.position),
                      quat=numbers(np.array(spec.actual.pose.quaternion_xyzw)[[3,0,1,2]]))
    ET.SubElement(box,'freejoint',name='task_box_free')
    ET.SubElement(box,'geom',name='task_box_geom',type='box',size=numbers(np.array(spec.actual.size)/2),
                  mass=str(spec.mass_kg),rgba='.65 .42 .20 1',friction=numbers([spec.floor_friction,.005,.0001]),
                  contype='0',conaffinity='0')
    pp=np.array(spec.destination.position); pp[2]=spec.pallet_height/2
    yaw=Rotation.from_matrix(spec.destination.rotation()).as_euler('xyz')[2]
    ET.SubElement(world,'geom',name='task_pallet',type='box',size=numbers([.4,.4,spec.pallet_height/2]),
                  pos=numbers(pp),quat=numbers(Rotation.from_euler('z',yaw).as_quat()[[3,0,1,2]]),rgba='.35 .28 .15 1',priority='1',friction=numbers([spec.pallet_friction,.005,.0001]))
    for name,mu in [('floor',spec.floor_friction),('task_pallet',spec.pallet_friction)]:
        ET.SubElement(contact,'pair',geom1=name,geom2='task_box_geom',condim='3',friction=numbers([mu,mu,0,0,0]))
    xml=ET.tostring(root,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml)
    assert model.neq==0 and model.nu==29 and model.nq==43
    return model,xml

def hand_surface_offsets(model):
    """Return the virtual hand-alignment plane centers used by grasp planning."""
    return {side: grip_pad_contact_anchor(side) for side in ('left','right')}

def contact_metrics(m,d):
    box=m.geom('task_box_geom').id
    out={s:{'normal_n':0.,'tangent_n':0.,'slip_m_s':0.,'force_world_n':np.zeros(3),'wrist_contact_torque_world_nm':np.zeros(3),'contacts':0} for s in ('left','right')}
    hand_geoms={}
    for side in out:
        ids={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'{side}_physical_hand_{index}')
             for index in range(2)}-{-1}
        if not ids:
            ids={m.geom(f'{side}_grip').id}
        hand_geoms[side]=ids
    support=0.; pallet_support=0.; penetration=0.
    for i in range(d.ncon):
        c=d.contact[i]
        if box not in (c.geom1,c.geom2): continue
        f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f)
        other=c.geom1 if c.geom2==box else c.geom2
        sign=1 if c.geom2==box else -1
        force=c.frame.reshape(3,3).T@f[:3]*sign
        penetration=max(penetration,-float(c.dist))
        if other in (m.geom('floor').id,m.geom('task_pallet').id):
            support+=max(0.,force[2])
            if other==m.geom('task_pallet').id: pallet_support+=max(0.,force[2])
        for side in out:
            if other not in hand_geoms[side]: continue
            h=out[side]; h['normal_n']+=max(0.,f[0]); h['tangent_n']+=np.linalg.norm(f[1:3]); h['contacts']+=1
            # External box force/wrench ON the hand, expressed in world axes at wrist origin.
            h['force_world_n']-=force
            wrist=m.body(side+'_wrist_yaw_link').id
            h['wrist_contact_torque_world_nm']+=np.cross(c.pos-d.xpos[wrist],-force)-c.frame.reshape(3,3).T@f[3:]*sign
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv)); mujoco.mj_jac(m,d,jp,jr,c.pos,m.geom_bodyid[box]); bv=jp@d.qvel
            mujoco.mj_jac(m,d,jp,jr,c.pos,wrist); rel=bv-jp@d.qvel
            normal=c.frame[:3]; tang=rel-normal*np.dot(normal,rel)
            h['slip_m_s']=max(h['slip_m_s'],float(np.linalg.norm(tang)))
    for h in out.values():
        for k,v in h.items():
            if isinstance(v,np.ndarray): h[k]=v.tolist()
            elif isinstance(v,np.generic): h[k]=v.item()
    return {'hands':out,'support_n':support,'pallet_support_n':pallet_support,'penetration_m':penetration}

def simulate(plan,t,q,folder,controller='sonic',progress=lambda *a:None,cancel=lambda:False, duration_limit=None):
    spec=TaskSpec.model_validate(plan['spec']); m,xml=build_scene(spec); d=mujoco.MjData(m)
    robot=Robot(); ref=Reference(t,q,robot.model); fsm=TaskFSM(plan)
    policy=SonicCPU() if controller=='sonic' else None
    if controller not in ('sonic','pd_diagnostic'): raise ValueError('Unknown controller')
    d.qpos[:36]=q[0]; mujoco.mj_forward(m,d)
    folder.mkdir(parents=True,exist_ok=True); (folder/'scene.xml').write_text(xml)
    jid=np.array([m.joint(name).id for name in robot.names]); qa=m.jnt_qposadr[jid]; va=m.jnt_dofadr[jid]
    limit=np.min(np.abs(m.jnt_actfrcrange[jid]),axis=1)
    boxid=m.body('task_box').id; boxva=m.joint('task_box_free').dofadr[0]
    start_box_z=float(d.xpos[boxid,2]); was_lifted=False
    trace=[]; logs=[]; start=time.perf_counter(); maxsat=0.; maxforce=0.
    steps=0; maxseconds=plan['duration']+len(plan['phases'])*spec.timeout_seconds
    if duration_limit is not None: maxseconds=min(maxseconds,duration_limit)
    while fsm.status=='running' and d.time<maxseconds:
        if cancel(): fsm.status='cancelled'; break
        phase=plan['phases'][fsm.index]
        target=policy.action(d.qpos[:36],d.qvel[:35],ref,fsm.reference_time,phase['end']) if policy else ref.sample(fsm.reference_time)[0][0,7:]
        sat=0.
        for _ in range(10):
            requested=KP*(target-d.qpos[qa])-KD*d.qvel[va]
            sat=max(sat,float(np.mean(np.abs(requested)>limit)))
            d.ctrl[:]=np.clip(requested,-limit,limit)
            mujoco.mj_step(m,d)
        steps+=1
        if any(d.warning[k].number for k in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL)):
            fsm.fail('numerical_instability'); break
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all(): fsm.fail('nonfinite_state'); break
        cm=contact_metrics(m,d); hands=cm['hands']; bp=d.xpos[boxid].copy()
        rot=d.xmat[m.body('pelvis').id].reshape(3,3)
        tilted=rot[2,2]<np.cos(np.deg2rad(45))
        box_axis=d.xmat[boxid].reshape(3,3)[:,1]
        bilateral=all(h['normal_n']>2 for h in hands.values()) and np.dot(hands['left']['force_world_n'],box_axis)>.5 and np.dot(hands['right']['force_world_n'],box_axis)<-.5
        was_lifted=was_lifted or bp[2]>start_box_z+.06
        rq=ref.sample(phase['end'])[0][0]
        root_error=np.linalg.norm(d.qpos[:2]-phase['root_position'][:2])
        pose_error=float(np.sqrt(np.mean((d.qpos[qa]-rq[7:])**2)))
        yaw_error=abs(float(Rotation.from_matrix(rot.T@Rotation.from_euler('z',phase['root_yaw']).as_matrix()).as_rotvec()[2]))
        box_angle_error=float(Rotation.from_matrix(spec.destination.rotation().T@d.xmat[boxid].reshape(3,3)).magnitude())
        settled=bool(cm['pallet_support_n']>spec.mass_kg*9.81*.5 and np.linalg.norm(d.qvel[boxva:boxva+3])<.06 and np.linalg.norm(d.qvel[boxva+3:boxva+6])<.2 and np.linalg.norm(bp-spec.destination.position)<.10 and box_angle_error<.25)
        released=all(h['normal_n']<1 for h in hands.values())
        root_arrived=bool(root_error<.12 and yaw_error<.25 and abs(d.qpos[2]-phase['root_position'][2])<.10 and np.linalg.norm(d.qvel[:2])<.20)
        if phase['id']=='retreat': root_arrived=root_arrived and settled and released
        metrics={'fallen': bool(d.qpos[2]<.25 or tilted), 'dropped':bool(was_lifted and not bilateral and bp[2]<start_box_z+.03),
                 'root_arrival':root_arrived,
                 'pose_arrival':bool(pose_error<.22 and abs(d.qpos[2]-rq[2])<.10),
                 'bilateral_contact':bilateral and max(h['slip_m_s'] for h in hands.values())<.08,
                 'box_lifted':bool(bp[2]>start_box_z+.06 and bilateral and cm['support_n']<spec.mass_kg*9.81*.2),
                 'box_arrival':bool(np.linalg.norm(bp-np.array(phase['box_target']))<.10 and bilateral),
                 'supported_and_still':settled,
                 'hands_released':bool(released and settled)}
        fsm.update(.02,metrics)
        maxsat=max(maxsat,sat); maxforce=max(maxforce,*(h['normal_n'] for h in hands.values()))
        row={'time':float(d.time),'phase':phase['id'],'reference_time':fsm.reference_time,'root_error_m':float(root_error),
             'joint_rmse_rad':pose_error,'box_orientation_error_rad':box_angle_error,'torque_saturation_fraction':sat,**cm,**metrics}
        logs.append(row)
        if steps%2==0:
            trace.append({'time':float(d.time),'qpos':d.qpos[:36].tolist(),'qvel':d.qvel[:35].tolist(),
                          'joint_target':target.tolist(),'joint_torque_nm':d.qfrc_actuator[va].tolist(),
                          'box_linear_velocity_world':d.qvel[boxva:boxva+3].tolist(),'box_position':bp.tolist(),
                          'box_quaternion_xyzw':d.xquat[boxid][[1,2,3,0]].tolist(),'metrics':row})
        if steps%25==0: progress(min(.99,float(d.time)/maxseconds),phase['label'])
    if fsm.status=='running': fsm.status='incomplete'; fsm.reason='duration_limit'
    report={'format':'motioncreator.task-validation.v1','task_sha256':plan['task_sha256'],
            'controller':controller,'device':'cpu','status':fsm.status,'reason':fsm.reason,
            'validated':fsm.status=='succeeded' and controller=='sonic','events':fsm.events,
            'inference_assets':policy.asset_info if policy else [],
            'reference_sha256':hashlib.sha256(np.ascontiguousarray(q).tobytes()+np.ascontiguousarray(t).tobytes()).hexdigest(),
            'controller_parameters':{'kp':KP.tolist(),'kd':KD.tolist(),'torque_limits_nm':limit.tolist(),'rotor_armature':m.dof_armature[6:35].tolist()},
            'minimum_static_normal_per_hand_n':spec.mass_kg*9.81/(2*spec.hand_friction) if spec.hand_friction else None,
            'sim_seconds':float(d.time),'wall_seconds':time.perf_counter()-start,'physics_hz':500,'policy_hz':50,
            'scene_sha256':hashlib.sha256(xml.encode()).hexdigest(),'max_torque_saturation_fraction':maxsat,
            'max_hand_normal_n':maxforce,'attachments':False,'external_support_forces':False,
            'wrench_semantics':'box contact wrench on real hand meshes, at wrist origin, world axes; not total wrist load',
            'limitations':['Rigid box and convex hand/wrist collision meshes; no cardboard deformation','A successful trial is specific to this reference, box and controller configuration']}
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (folder/'contact-log.json').write_text(json.dumps(logs,allow_nan=False))
    (folder/'replay.json').write_text(json.dumps(trace,allow_nan=False))
    if trace:
        arrays={key:np.asarray([x[key] for x in trace]) for key in ('time','qpos','qvel','joint_target','joint_torque_nm','box_position','box_quaternion_xyzw','box_linear_velocity_world')}
        np.savez_compressed(folder/'simulation.npz',**arrays,joint_names=np.array(robot.names),
                            metadata_json=json.dumps({'format':'motioncreator.simulation.v1','robot_root_quaternion':'wxyz','box_quaternion':'xyzw',
                            'coordinate_system':'right-handed, +X forward, +Y left, +Z up','units':{'position':'m','angle':'rad','time':'s','torque':'N m'},
                            'robot_qvel':'MuJoCo free joint: root linear world, root angular local, hinge rad/s',
                            'reference_sha256':report['reference_sha256'],'scene_sha256':report['scene_sha256']}))
    return report
