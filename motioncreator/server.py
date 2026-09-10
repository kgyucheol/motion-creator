from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .robot import Robot, ROOT, FEET, HANDLES, ROTATABLE
from .motion import new_project, validate_project, compile_motion, save_bundle
from .presets import GroupStore

robot = Robot()
groups = GroupStore()
app = FastAPI(title='G1 Motion Creator', version='1.0.0')
from .task_api import router as task_router
app.include_router(task_router)
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:3000', 'http://localhost:3000'], allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])


class PoseInput(BaseModel):
    qpos: list[float]


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


class GroupSolveInput(PoseInput):
    model_config = {'extra': 'forbid'}
    anchor: list[float]
    targets: dict[str, list[float]] = Field(default_factory=dict, max_length=len(HANDLES))
    orientations: dict[str, list[float]] = Field(default_factory=dict, max_length=len(ROTATABLE))
    joints: dict[str, float] = Field(default_factory=dict, max_length=29)
    pins: list[str] = Field(default_factory=lambda: list(FEET))
    resistance: float = Field(1., ge=0, le=5)
    mode: Literal['elastic', 'free'] = 'elastic'


class GroupPresetInput(BaseModel):
    id: str | None = None
    name: str
    members: list[str]


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
    return {'status': 'ok', 'model': 'G1 29 DoF', 'nq': robot.model.nq, 'nv': robot.model.nv}


@app.get('/api/init')
def initialize():
    return {'state': robot.state(robot.home), 'project': new_project(robot), 'joint_names': robot.names,
            'limits': robot.model.jnt_range[1:].tolist()}


@app.get('/api/robot.glb')
def visual():
    path = ROOT / 'assets/g1/robot.glb'
    return FileResponse(path, media_type='model/gltf-binary')


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


@app.post('/api/solve-group')
def solve_group(payload: GroupSolveInput):
    def run():
        if not any((payload.targets, payload.orientations, payload.joints)):
            raise ValueError('At least one position, rotation or joint target is required')
        q, info = robot.solve(payload.qpos, payload.anchor, pins=payload.pins,
                             resistance=payload.resistance, mode=payload.mode, selected_targets=payload.targets,
                             orientation_targets=payload.orientations, joint_targets=payload.joints)
        return {'state': robot.state(q), 'solver': info}
    return checked(run)


@app.post('/api/preview')
def preview(payload: ProjectInput):
    def run():
        motion = compile_motion(robot, payload.project, min(payload.fps, 30))
        return {'time': motion['time'].tolist(), 'states': [robot.state(q) for q in motion['qpos']],
                'max_pin_error_mm': motion['max_pin_error_mm']}
    return checked(run)


@app.post('/api/save')
def save(payload: ProjectInput):
    return checked(lambda: save_bundle(robot, payload.project, payload.fps, protomotions=payload.protomotions))


@app.get('/api/files/{name}')
def download(name: str):
    if Path(name).name != name or Path(name).suffix not in ('.json', '.npz', '.motion', '.pt'):
        raise HTTPException(404)
    path = ROOT / 'motions' / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=name)


@app.get('/api/saved')
def saved():
    return [p.name for p in sorted((ROOT / 'motions').glob('*.json'), reverse=True)
            if not p.name.endswith(('.metadata.json', '.protomotions.json'))]


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
            print(f'G1 Motion Creator가 이미 실행 중입니다.\n브라우저에서 열기: {address}/')
            return
    except (urllib.error.URLError, OSError, ValueError):
        pass
    robot.export_visual(ROOT / 'assets/g1/robot.glb')
    frontend = ROOT / 'frontend/dist'
    if (frontend / 'index.html').exists():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='editor')
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
