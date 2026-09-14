import fcntl,json,sys,traceback,os,time
from pathlib import Path
from .task_jobs import atomic_json,TASKS

def main(folder):
    started=time.perf_counter()
    status=json.loads((folder/'status.json').read_text()); status['worker_pid']=os.getpid(); request=json.loads((folder/'request.json').read_text())
    def progress(value,message):
        status.update(status='running',progress=float(value),message=message); atomic_json(folder/'status.json',status)
    def cancelled(): return (folder/'cancel').exists()
    try:
        with (TASKS/'.worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            progress(.01,'CPU 작업 준비')
            if cancelled(): raise InterruptedError('cancelled')
            if request['kind']=='simulate':
                from .reference import load_reference
                from .task_physics import simulate
                data,_=load_reference(folder/'reference.npz'); t=data['time']; q=data['qpos']
                result=simulate(request['plan'],t,q,folder,request['controller'],progress,cancelled)
            else:
                from .task_generation import generate_ardy,ik_preview
                result=generate_ardy(request['plan'],folder,progress,cancelled) if request['kind']=='ardy' else ik_preview(request['plan'],folder,progress)
            result['worker_wall_seconds']=time.perf_counter()-started
            atomic_json(folder/'result.json',result)
            status.update(status='cancelled' if cancelled() else 'completed',progress=1.,message='작업 취소됨' if cancelled() else '결과 저장 완료')
    except InterruptedError: status.update(status='cancelled',message='작업 취소됨')
    except Exception as exc:
        traceback.print_exc(); status.update(status='failed',message=f'{type(exc).__name__}: {exc}')
    atomic_json(folder/'status.json',status)

if __name__=='__main__': main(Path(sys.argv[1]))
