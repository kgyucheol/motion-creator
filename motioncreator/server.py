import json
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .robot import Robot, ROOT, FEET, HANDLES, ROTATABLE, ANGLE_LOCKABLE
from .motion import new_project, validate_project, compile_motion, project_from_motion_bytes, prepare_saved_project, save_bundle
from .presets import GroupStore

robot = Robot()
groups = GroupStore()
app = FastAPI(title='G1 Motion Creator', version='1.0.0')
from .task_api import router as task_router
app.include_router(task_router)
from .decoupled_api import router as decoupled_router
app.include_router(decoupled_router)
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:3000', 'http://localhost:3000'], allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])


class PoseInput(BaseModel):
    qpos: list[float]


class ModelInput(BaseModel):
    model_id: Literal['g1', 'g1-tools']


@app.post('/api/model')
def select_model(payload: ModelInput):
    global robot
    from .policy_preview import jobs
    from .decoupled_wbc import sessions
    with sessions.lock, jobs.lock:
        if sessions.sessions or any(job['status'] == 'running' for job in jobs.jobs.values()):
            raise HTTPException(409, 'WBC 세션을 닫고 물리 계산을 완료한 뒤 모델을 변경하세요.')
        if robot.model_id == payload.model_id:
            return {'model_id': robot.model_id}
        try:
            candidate = Robot(payload.model_id)
            candidate.export_visual(ROOT / f'assets/g1/robot-{candidate.model_id}.glb')
        except (ValueError, OSError) as exc:
            raise HTTPException(422, detail=f'모델을 불러오지 못했습니다: {exc}') from exc
        if jobs.sonic_process is not None:
            jobs.sonic_process.terminate()
            jobs.sonic_process.wait(timeout=5)
            jobs.sonic_process = None
        os.environ['MOTIONCREATOR_MODEL'] = candidate.model_id
        robot = candidate
        sessions.robot = None
    return {'model_id': robot.model_id}


class SolveInput(PoseInput):
    anchor: list[float]
    focus: str
    target: list[float]
    pins: list[str] = Field(default_factory=lambda: list(FEET))
    resistance: float = Field(1., ge=0, le=5)
    mode: Literal['elastic', 'free'] = 'elastic'


class ProjectInput(BaseModel):
    project: dict
    fps: int = Field(30, ge=1, le=120)
    protomotions: bool = False


class PhysicsPreviewInput(BaseModel):
    project: dict
    controller: Literal['pd', 'gear-sonic'] = 'gear-sonic'
    start_frame_index: int = Field(0, ge=0)


class GroupSolveInput(PoseInput):
    model_config = {'extra': 'forbid'}
    anchor: list[float]
    targets: dict[str, list[float]] = Field(default_factory=dict, max_length=len(HANDLES))
    orientations: dict[str, list[float]] = Field(default_factory=dict, max_length=len(ROTATABLE))
    joints: dict[str, float] = Field(default_factory=dict, max_length=29)
    pins: list[str] = Field(default_factory=lambda: list(FEET))
    angle_pins: list[str] = Field(default_factory=list, max_length=len(ANGLE_LOCKABLE))
    resistance: float = Field(1., ge=0, le=5)
    mode: Literal['elastic', 'free'] = 'elastic'


class GroupPresetInput(BaseModel):
    id: str | None = None
    name: str
    members: list[str]


class GraspFitInput(PoseInput):
    model_config = {'extra': 'forbid'}
    pins: list[str] = Field(default_factory=lambda: list(FEET))
    angle_pins: list[str] = Field(default_factory=list, max_length=len(ANGLE_LOCKABLE))
    object: dict
    grasp: dict


@app.get('/api/groups')
def list_groups():
    return checked(groups.list)


@app.post('/api/groups')
def save_group(payload: GroupPresetInput):
    return checked(lambda: groups.save(payload.name, payload.members, payload.id))


@app.delete('/api/groups/{group_id}')
def delete_group(group_id: str):
    return checked(lambda: groups.delete(group_id))


def checked(fn):
    try:
        return fn()
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/api/health')
def health():
    return {'status': 'ok', 'model': 'G1 29 DoF', 'model_id': robot.model_id, 'nq': robot.model.nq, 'nv': robot.model.nv}


@app.get('/api/init')
def initialize():
    return {'model_id': robot.model_id, 'visual_revision': robot.fingerprint[:16],
            'state': robot.state(robot.home), 'project': new_project(robot), 'joint_names': robot.names,
            'limits': robot.model.jnt_range[1:].tolist()}


def visual_response(model_id: str):
    if model_id != robot.model_id:
        raise HTTPException(409, detail=f'현재 선택된 로봇은 {robot.model_id}입니다. 화면을 새로고침하세요.')
    path = ROOT / f'assets/g1/robot-{model_id}.glb'
    if not path.is_file():
        robot.export_visual(path)
    return FileResponse(path, media_type='model/gltf-binary', headers={
        'Cache-Control': 'no-store, max-age=0',
        'Pragma': 'no-cache',
    })


@app.get('/api/robot/{model_id}.glb')
def model_visual(model_id: Literal['g1', 'g1-tools']):
    return visual_response(model_id)


@app.get('/api/robot.glb')
def visual():
    return visual_response(robot.model_id)


@app.post('/api/pose')
def pose(payload: PoseInput):
    return checked(lambda: robot.state(robot.validate_q(payload.qpos)))


@app.post('/api/solve')
def solve(payload: SolveInput):
    def run():
        q, info = robot.solve(payload.qpos, payload.anchor, payload.focus, payload.target,
                             payload.pins, payload.resistance, payload.mode)
        return {'state': robot.state(q), 'solver': info}
    return checked(run)


@app.post('/api/validate')
def validate(payload: ProjectInput):
    return checked(lambda: validate_project(robot, payload.project))


@app.post('/api/import-motion')
async def import_motion(request: Request, filename: str, fps: int = Query(30, ge=1, le=120)):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in ('.npz', '.csv'):
        raise HTTPException(status_code=422, detail='Only .npz and .csv motion files are supported')
    content = await request.body()
    if not content:
        raise HTTPException(status_code=422, detail='Motion file is empty')
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Motion file must be 100 MB or smaller')
    return checked(lambda: project_from_motion_bytes(robot, content, filename, fps))


@app.post('/api/scene-assets/import')
async def import_scene_model(request: Request, filename: str):
    from .scene_assets import import_scene_asset
    content = await request.body()
    return checked(lambda: import_scene_asset(content, filename))


@app.get('/api/scene-assets/{identifier}.glb')
def scene_model(identifier: str):
    from .scene_assets import asset_path
    try:
        path = asset_path(identifier)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, detail='3D model asset not found')
    return FileResponse(path, media_type='model/gltf-binary', headers={'Cache-Control': 'public, max-age=31536000, immutable'})


@app.get('/api/scene-assets/{identifier}/parts/{part_id}.glb')
def scene_model_part(identifier: str, part_id: str):
    from .scene_assets import asset_part_path
    try:
        path = asset_part_path(identifier, part_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, detail='3D model asset part not found')
    return FileResponse(path, media_type='model/gltf-binary', headers={'Cache-Control': 'public, max-age=31536000, immutable'})


@app.post('/api/solve-group')
def solve_group(payload: GroupSolveInput):
    def run():
        if not any((payload.targets, payload.orientations, payload.joints)):
            raise ValueError('At least one position, rotation or joint target is required')
        q, info = robot.solve(payload.qpos, payload.anchor, pins=payload.pins,
                             resistance=payload.resistance, mode=payload.mode, selected_targets=payload.targets,
                             orientation_targets=payload.orientations, joint_targets=payload.joints,
                             angle_pins=payload.angle_pins)
        return {'state': robot.state(q), 'solver': info}
    return checked(run)


@app.post('/api/grasp-fit')
def grasp_fit(payload: GraspFitInput):
    from .grasp import fit_two_hand_grasp
    return checked(lambda: fit_two_hand_grasp(robot, payload.qpos, payload.pins, payload.object,
                                              payload.grasp, payload.angle_pins))


@app.post('/api/preview')
def preview(payload: ProjectInput):
    def run():
        motion = compile_motion(robot, payload.project, min(payload.fps, 30))
        return {'time': motion['time'].tolist(), 'states': [robot.state(q) for q in motion['qpos']],
                'max_pin_error_mm': motion['max_pin_error_mm']}
    return checked(run)


@app.get('/api/policy-preview/runtime')
def policy_runtime():
    from .policy_preview import runtime
    return runtime()


@app.post('/api/policy-preview')
def policy_preview(payload: PhysicsPreviewInput):
    from .policy_preview import jobs
    return checked(lambda: jobs.start(payload.project, payload.controller, payload.start_frame_index))


@app.get('/api/policy-preview/{identifier}')
def policy_status(identifier: str):
    from .policy_preview import jobs
    return checked(lambda: jobs.status(identifier))


@app.get('/api/policy-preview/{identifier}/result')
def policy_result(identifier: str):
    from .policy_preview import jobs
    return checked(lambda: jobs.result(identifier))


@app.post('/api/policy-preview/{identifier}/cancel')
def policy_cancel(identifier: str):
    from .policy_preview import jobs
    return checked(lambda: jobs.cancel(identifier))


@app.post('/api/save')
def save(payload: ProjectInput):
    return checked(lambda: save_bundle(robot, payload.project, payload.fps, protomotions=payload.protomotions))


@app.post('/api/save-as')
def save_as(payload: ProjectInput):
    return checked(lambda: save_bundle(robot, payload.project, payload.fps,
                                       protomotions=payload.protomotions, save_as=True))


@app.get('/api/files/{name:path}')
def download(name: str):
    relative = Path(name)
    if (relative.is_absolute() or '..' in relative.parts
            or relative.suffix not in ('.json', '.npz', '.csv', '.motion', '.pt')):
        raise HTTPException(404)
    path = ROOT / 'motions' / relative
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=path.name)


@app.get('/api/project/{name:path}')
def open_saved_project(name: str):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.json':
        raise HTTPException(404)
    path = ROOT / 'motions' / relative
    if not path.is_file():
        raise HTTPException(404)
    def load():
        project = json.loads(path.read_text(encoding='utf-8'))
        prepare_saved_project(project, relative, path.stat().st_mtime)
        return validate_project(robot, project)
    return checked(load)


@app.get('/api/saved')
def saved():
    folder = ROOT / 'motions'
    projects = [*folder.glob('*/project.json'),
                *(path for path in folder.glob('*.json') if not path.name.endswith(('.metadata.json', '.protomotions.json')))]
    return [str(path.relative_to(folder)) for path in sorted(projects, key=lambda path: path.stat().st_mtime, reverse=True)]


@app.get('/api/demo')
def demo():
    from .demo import crouch_demo
    return checked(lambda: crouch_demo(robot))


def main():
    import argparse
    import json
    import urllib.error
    import urllib.request
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    address = f'http://127.0.0.1:{args.port}'
    # Reuse a running editor; never stop another process occupying the port.
    # Ignore proxy environment variables for this loopback-only check.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'{address}/api/health', timeout=1) as response:
            running_health = json.load(response)
        with opener.open(f'{address}/openapi.json', timeout=1) as response:
            running_schema = json.load(response)
        if (isinstance(running_health, dict) and isinstance(running_schema, dict)
                and running_health.get('status') == 'ok'
                and running_health.get('model') == 'G1 29 DoF'
                and running_schema.get('info', {}).get('title') == 'G1 Motion Creator'):
            if running_health.get('model_id', 'g1') != robot.model_id:
                print('다른 로봇 모델의 편집기가 이 포트에서 실행 중입니다. --port 8766 등 다른 포트를 사용하세요.')
            else:
                print(f'G1 Motion Creator가 이미 실행 중입니다.\n브라우저에서 열기: {address}/')
            return
    except (urllib.error.URLError, OSError, ValueError):
        pass
    robot.export_visual(ROOT / f'assets/g1/robot-{robot.model_id}.glb')
    frontend = ROOT / 'frontend/dist'
    if (frontend / 'index.html').exists():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='editor')
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
