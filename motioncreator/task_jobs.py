"""Immutable run snapshots and isolated, cancellable CPU workers."""
import json,os,re,subprocess,uuid,threading,time
from datetime import datetime,timezone
from pathlib import Path
from .robot import ROOT,Robot
from .tasks import TaskSpec,plan_task,digest

TASKS=ROOT/'tasks'

def atomic_json(path,value):
    temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp'); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)); temp.replace(path)
def identifier(value):
    if not re.fullmatch('[0-9a-f]{12}',value): raise ValueError('Invalid task/run ID')
    return value

def runtime_status():
    folder=ROOT/'external/task-models'
    missing=[]
    for p in [ROOT/'.conda-policy/bin/python',ROOT/'external/ardy-src/ardy/model/load_model.py',
              folder/'ardy/ARDY-G1-RP-25FPS-Horizon52/denoiser.safetensors',folder/'ardy/ARDY-G1-RP-25FPS-Horizon52/tokenizer.safetensors',
              folder/'sonic/model_encoder.onnx',folder/'sonic/model_decoder.onnx']:
        if not p.is_file(): missing.append(str(p.relative_to(ROOT)))
    return {'device':'cpu','installed_assets':not missing,'missing':missing,'execution':'offline; 500 Hz physics / 50 Hz policy, wall-clock speed may be slower',
            'ardy':'ARDY G1 Horizon52, no text encoder','sonic':'original release ONNX, G1 reference mode',
            'reference_files':[p.name for p in sorted((ROOT/'motions').glob('*.npz'))]}

def save_task(spec,start_q=None,task_id=None):
    plan=plan_task(spec,start_q); tid=identifier(task_id) if task_id else uuid.uuid4().hex[:12]
    folder=TASKS/tid; folder.mkdir(parents=True,exist_ok=True)
    record={'id':tid,'updated_at':datetime.now(timezone.utc).isoformat(),'plan':plan}
    atomic_json(folder/'task.json',record); return record

def get_task(tid): return json.loads((TASKS/identifier(tid)/'task.json').read_text())
def list_tasks():
    result=[]
    for p in TASKS.glob('*/task.json'):
        record=json.loads(p.read_text()); result.append({'id':record['id'],'name':record['plan']['spec']['name'],'updated_at':record['updated_at']})
    return sorted(result,key=lambda x:x['updated_at'],reverse=True)
def run_folder(tid,rid): return TASKS/identifier(tid)/'runs'/identifier(rid)
def get_run(tid,rid):
    folder=run_folder(tid,rid); status=json.loads((folder/'status.json').read_text())
    if status['status'] in ('starting','running'):
        pid=status.get('worker_pid')
        gone=False
        if pid:
            try:
                command=Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
                gone=b'motioncreator.task_worker' not in command
            except (OSError,ValueError): gone=True
        elif time.time()-(folder/'status.json').stat().st_mtime>60: gone=True
        if gone:
            latest=json.loads((folder/'status.json').read_text())
            if latest['status'] in ('starting','running'):
                latest.update(status='interrupted',message='CPU 작업 프로세스가 종료되었습니다. 실행 로그를 확인하고 다시 실행하세요.')
                atomic_json(folder/'status.json',latest)
            status=latest
    if (folder/'result.json').is_file(): status['result']=json.loads((folder/'result.json').read_text())
    status['files']=sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.npz','.xml','.log'))
    return status

def list_runs(tid):
    return [get_run(tid,p.name) for p in sorted((TASKS/identifier(tid)/'runs').glob('*'),key=lambda p:p.stat().st_mtime,reverse=True) if (p/'status.json').is_file()]

def start_run(tid,kind,reference_run=None,reference_file=None,controller='sonic'):
    if kind not in ('ardy','ik_preview','simulate'): raise ValueError('Unknown run kind')
    if controller not in ('sonic','pd_diagnostic'): raise ValueError('Unknown controller')
    task=get_task(tid); plan=task['plan']
    # Check active CPU worker before allocating another immutable run.
    import fcntl
    TASKS.mkdir(exist_ok=True)
    with (TASKS/'.worker.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('CPU 작업이 실행 중입니다. 완료하거나 취소한 뒤 다시 실행하세요.')
    source=None
    if kind=='simulate':
        if reference_run:
            previous=run_folder(tid,reference_run)
            previous_plan=json.loads((previous/'request.json').read_text())['plan']
            if previous_plan['task_sha256']!=plan['task_sha256']: raise ValueError('상자/작업 설정이 변경되었습니다. 현재 설정으로 모션을 다시 생성하세요.')
            source=previous/'reference.npz'
        elif reference_file:
            if Path(reference_file).name!=reference_file or not reference_file.endswith('.npz'): raise ValueError('Invalid reference filename')
            source=ROOT/'motions'/reference_file
        else: raise ValueError('생성한 모션 또는 기존 NPZ를 선택하세요.')
        if not source.is_file(): raise ValueError('참조 모션 파일이 없습니다.')
    folder=TASKS/identifier(tid)/'runs'/uuid.uuid4().hex[:12]; folder.mkdir(parents=True)
    if source:
        from .reference import load_reference,joint_permutation
        data,meta=load_reference(source); robot=Robot()
        if meta.get('model_sha256')!=robot.fingerprint: raise ValueError('Reference model fingerprint mismatch')
        if data['joint_names'].tolist()!=robot.names: raise ValueError('Reference joint order mismatch')
        import shutil
        shutil.copyfile(source,folder/'reference.npz')
        source_metadata = source.with_suffix('.metadata.json')
        if source_metadata.is_file():
            shutil.copyfile(source_metadata, folder/'reference.metadata.json')
        else:
            atomic_json(folder/'reference.metadata.json', meta)
    request={'kind':kind,'plan':plan,'controller':controller}
    atomic_json(folder/'request.json',request)
    atomic_json(folder/'status.json',{'id':folder.name,'task_id':tid,'kind':kind,'status':'starting','progress':0.,'message':'CPU 작업 시작 중'})
    env=os.environ.copy(); env.update(PYTHONPATH=str(ROOT),PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    with (folder/'worker.log').open('w') as log:
        try:
            process=subprocess.Popen([str(ROOT/'.conda-policy/bin/python'),'-m','motioncreator.task_worker',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError as exc:
            atomic_json(folder/'status.json',{'id':folder.name,'task_id':tid,'kind':kind,'status':'failed','progress':0.,'message':str(exc)})
            raise ValueError('CPU 환경 실행 실패. scripts/setup-task-cpu.sh를 확인하세요.') from exc
    def watch():
        code=process.wait()
        saved=json.loads((folder/'status.json').read_text())
        if saved['status'] in ('starting','running'):
            saved.update(status='failed',message=f'CPU worker exited unexpectedly ({code}); see worker.log')
            atomic_json(folder/'status.json',saved)
    threading.Thread(target=watch,daemon=True).start()
    return get_run(tid,folder.name)
