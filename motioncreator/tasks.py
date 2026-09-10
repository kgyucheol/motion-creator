"""Object-relative task contracts. Planning never consumes ground-truth box dimensions."""
from __future__ import annotations
import hashlib, json, math
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation
from .robot import Robot, HANDLES, FEET

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Pose(Strict):
    position: list[float] = Field(default_factory=lambda: [.60, 0., .20], min_length=3, max_length=3)
    quaternion_xyzw: list[float] = Field(default_factory=lambda: [0., 0., 0., 1.], min_length=4, max_length=4)
    @model_validator(mode='after')
    def check(self):
        if max(abs(x) for x in self.position)>4: raise ValueError('작업 위치는 원점에서 각 축 ±4 m 이내여야 합니다.')
        if abs(np.linalg.norm(self.quaternion_xyzw)-1)>1e-4: raise ValueError('정규화된 xyzw quaternion이 필요합니다.')
        return self
    def rotation(self): return Rotation.from_quat(self.quaternion_xyzw).as_matrix()

class BoxSpec(Strict):
    pose: Pose = Field(default_factory=Pose)
    size: list[float] = Field(default_factory=lambda: [.40,.51,.40], min_length=3, max_length=3)
    @model_validator(mode='after')
    def check(self):
        if not all(.03<=v<=1.2 for v in self.size): raise ValueError('상자 크기는 각 축 0.03–1.2 m입니다.')
        return self

class TaskSpec(Strict):
    format: Literal['motioncreator.box-task.v1'] = 'motioncreator.box-task.v1'
    name: str = Field('Box pick and place', min_length=1, max_length=80)
    actual: BoxSpec = Field(default_factory=BoxSpec)
    perceived: BoxSpec = Field(default_factory=lambda: BoxSpec(size=[.40,.49,.40]))
    destination: Pose = Field(default_factory=lambda: Pose(position=[1.25,0.,.32]))
    mass_kg: float = Field(1., gt=0, le=20)
    hand_friction: float = Field(.8, ge=0, le=2)
    floor_friction: float = Field(.7, ge=0, le=2)
    pallet_friction: float = Field(.7, ge=0, le=2)
    pallet_height: float = Field(.12, ge=.01, le=.5)
    squeeze_m: float = Field(0., ge=0, le=.06, description='Additional total closure, independent of perception error')
    pregrasp_clearance_m: float = Field(.06, ge=.01, le=.25)
    standoff_m: float = Field(.42, ge=.25, le=.8)
    carry_height_m: float = Field(.65, ge=.35, le=.95)
    phase_seconds: float = Field(2., ge=.4, le=6)
    timeout_seconds: float = Field(3., ge=.5, le=10)
    seed: int = Field(42, ge=0, le=2147483647)
    diffusion_steps: int = Field(10, ge=1, le=10)
    @model_validator(mode='after')
    def check(self):
        if self.perceived.size[1]-self.squeeze_m < .03: raise ValueError('파지 간격이 너무 작습니다.')
        # First PoC is upright boxes on horizontal supports; retain a genuine 6D input contract.
        for p in (self.actual.pose,self.perceived.pose,self.destination):
            if p.rotation()[2,2] < math.cos(math.radians(10)):
                raise ValueError('첫 PoC는 수평 받침 위의 상자만 지원합니다. roll/pitch는 ±10° 이내로 설정하세요.')
        return self

def digest(spec):
    data=spec.model_dump() if isinstance(spec,TaskSpec) else spec
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

PHASES = ['approach','crouch','pregrasp','grasp','lift','stand','carry','lower','place','release','retreat']
LABELS = ['상자 접근','웅크리기','양손 접근','압착 파지','들어 올리기','일어서기','목표로 운반','내려가기','안착','손 떼기','물러나기']

def plan_task(spec: TaskSpec, start_q=None):
    """World-space surface targets; perception-only planning, physical box separate."""
    robot=Robot(); q=robot.validate_q(robot.home if start_q is None else start_q)
    source=np.array(spec.perceived.pose.position); dest=np.array(spec.destination.position)
    sr=spec.perceived.pose.rotation(); dr=spec.destination.rotation()
    approach=source-sr[:,0]*spec.standoff_m; approach[2]=q[2]
    finish=dest-dr[:,0]*spec.standoff_m; finish[2]=q[2]
    grasp_gap=spec.perceived.size[1]-spec.squeeze_m
    phases=[]
    for i,(name,label) in enumerate(zip(PHASES,LABELS)):
        place=i>=7
        center=(dest if place else source).copy(); rot=dr if i>=6 else sr
        if i in (4,): center[2]+=.10
        elif i in (5,6): center[2]=spec.carry_height_m
        if i==6: center[:2]=dest[:2]
        base=(finish if i>=6 else approach).copy()
        if i in (1,2,3,4,7,8,9): base[2]=max(.38,min(q[2],center[2]+.30))
        if i==10: base-=dr[:,0]*.25
        clearance=spec.pregrasp_clearance_m if i in (0,1,2,9,10) else 0.
        gap=grasp_gap+2*clearance
        hands={s:(center+rot[:,1]*(gap/2)*(1 if s=='left' else -1)).tolist() for s in ('left','right')}
        phases.append({'id':name,'label':label,'start':i*spec.phase_seconds,'end':(i+1)*spec.phase_seconds,
                       'root_position':base.tolist(),'root_yaw':float(math.atan2(rot[1,0],rot[0,0])),
                       'surface_targets':hands,'hand_rotation_xyzw':Rotation.from_matrix(rot).as_quat().tolist(),
                       'box_target':center.tolist(),'controller':'sonic',
                       'guard': {'approach':'root_arrival','grasp':'bilateral_contact','lift':'box_lifted',
                                 'carry':'box_arrival','place':'supported_and_still','release':'hands_released',
                                 'retreat':'root_arrival'}.get(name,'pose_arrival')})
    return {'format':'motioncreator.task-plan.v1','task_sha256':digest({'spec':spec.model_dump(),'start_qpos':q.tolist(),'model_sha256':robot.fingerprint,'planner_version':2}),'spec':spec.model_dump(),
            'start_qpos':q.tolist(),'phases':phases,'duration':len(phases)*spec.phase_seconds,
            'coordinate_system':'right-handed, +X forward, +Y left, +Z up; metres; xyzw rotations',
            'target_surface_gap_m':grasp_gap,'validation_status':'not_run',
            'navigation_contract':{'production':'external navigation supplies arrival; hand off whole-body ownership to SONIC',
                                   'poc':'SONIC follows generated approach/carry/retreat reference; no SLAM or SDK navigation'} }

class TaskFSM:
    """Reference time pauses at phase boundaries until measured guards hold."""
    def __init__(self,plan):
        self.plan=plan; self.index=0; self.clock=0.; self.elapsed=0.; self.reference_time=0.; self.dwell=0.; self.status='running'; self.reason=''; self.events=[]
    def update(self,dt,metrics):
        if self.status!='running': return
        self.clock+=dt; self.elapsed+=dt
        p=self.plan['phases'][self.index]
        self.reference_time=min(p['end'],self.reference_time+dt)
        if metrics.get('fallen') or metrics.get('nonfinite'):
            self.fail('robot_fall' if metrics.get('fallen') else 'nonfinite_state'); return
        if self.index in (5,6,7) and metrics.get('dropped'):
            self.fail('box_dropped'); return
        guard=p['guard']; passed=bool(metrics.get(guard,False))
        self.dwell=self.dwell+dt if passed and self.reference_time>=p['end']-1e-8 else 0.
        if self.dwell>=.20:
            self.events.append({'phase':p['id'],'time':self.elapsed,'phase_time':self.clock,'event':'guard_passed'})
            self.index+=1; self.dwell=0.; self.clock=0.
            if self.index==len(self.plan['phases']): self.status='succeeded'
        elif self.clock > (p['end']-p['start'])+self.plan['spec']['timeout_seconds']:
            self.fail('timeout:'+guard)
    def fail(self,reason):
        self.status='failed'; self.reason=reason
        self.events.append({'phase':self.plan['phases'][self.index]['id'],'time':self.elapsed,'phase_time':self.clock,'event':reason})
