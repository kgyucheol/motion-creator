from pathlib import Path
import json
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict
from typing import Literal
from .tasks import TaskSpec,plan_task
from . import task_jobs as jobs
from .robot import Robot
from .reference import load_reference

router=APIRouter(prefix='/api/tasks')
class SaveInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    spec:TaskSpec
    start_qpos:list[float]|None=None
    id:str|None=None
class RunInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['ardy','ik_preview','simulate']
    reference_run:str|None=None
    reference_file:str|None=None
    controller:Literal['sonic','pd_diagnostic']='sonic'

def checked(fn):
    try: return fn()
    except FileNotFoundError as exc: raise HTTPException(404,'파일을 찾을 수 없습니다.') from exc
    except (ValueError,KeyError,TypeError) as exc: raise HTTPException(422,str(exc)) from exc

@router.get('/defaults')
def defaults(): return TaskSpec().model_dump()
@router.get('/runtime')
def runtime(): return jobs.runtime_status()
@router.get('')
def listing(): return checked(jobs.list_tasks)
@router.post('')
def save(payload:SaveInput): return checked(lambda:jobs.save_task(payload.spec,payload.start_qpos,payload.id))
@router.post('/plan')
def plan(payload:SaveInput): return checked(lambda:plan_task(payload.spec,payload.start_qpos))
@router.get('/{tid}')
def get(tid:str): return checked(lambda:jobs.get_task(tid))
@router.get('/{tid}/runs')
def runs(tid:str): return checked(lambda:jobs.list_runs(tid))
@router.post('/{tid}/runs')
def run(tid:str,payload:RunInput): return checked(lambda:jobs.start_run(tid,**payload.model_dump()))
@router.get('/{tid}/runs/{rid}')
def status(tid:str,rid:str): return checked(lambda:jobs.get_run(tid,rid))
@router.post('/{tid}/runs/{rid}/cancel')
def cancel(tid:str,rid:str):
    def work():
        folder=jobs.run_folder(tid,rid)
        if not folder.is_dir(): raise FileNotFoundError()
        (folder/'cancel').touch(); return {'status':'cancellation_requested'}
    return checked(work)
@router.get('/{tid}/runs/{rid}/files/{name}')
def file(tid:str,rid:str,name:str):
    if Path(name).name!=name or Path(name).suffix not in ('.json','.npz','.xml','.log'): raise HTTPException(404)
    def work():
        p=jobs.run_folder(tid,rid)/name
        if not p.is_file(): raise FileNotFoundError()
        return FileResponse(p,filename=p.name)
    return checked(work)
@router.get('/{tid}/runs/{rid}/replay')
def replay(tid:str,rid:str):
    def work():
        folder=jobs.run_folder(tid,rid); robot=Robot(); request=json.loads((folder/'request.json').read_text())
        if (folder/'replay.json').exists():
            frames=json.loads((folder/'replay.json').read_text())
        else:
            data,_=load_reference(folder/'reference.npz')
            # 12.5 FPS preview keeps browser memory bounded; full-rate NPZ is retained.
            frames=[{'time':float(t),'qpos':q.tolist()} for t,q in zip(data['time'][::2],data['qpos'][::2])]
        for frame in frames: frame['state']=robot.state(frame.pop('qpos'))
        return {'frames':frames,'plan':request['plan'],'physics':(folder/'replay.json').exists()}
    return checked(work)
