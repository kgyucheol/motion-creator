import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .robot import Robot, ROOT, FEET, HANDLES, ROTATABLE, ANGLE_LOCKABLE
from .motion import new_project, validate_project, compile_motion, rename_project, enable_parallel_compile, project_from_motion_bytes, prepare_saved_project, save_bundle
from .presets import GroupStore

SELECTION_PATH = ROOT / '.motioncreator-server.json'


def _saved_selection():
    try:
        value = json.loads(SELECTION_PATH.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _preferred_tools():
    from .tool_model import parse_tools, DEFAULT_TOOLS
    value = os.environ.get('MOTIONCREATOR_TOOLS') or _saved_selection().get('tools')
    if isinstance(value, (list, tuple)):
        if not all(isinstance(tool, str) for tool in value):
            return DEFAULT_TOOLS
        value = ':'.join(value)
    try:
        return parse_tools(value)
    except ValueError:
        return DEFAULT_TOOLS


def _initial_robot():
    model_id = os.environ.get('MOTIONCREATOR_MODEL') or _saved_selection().get('model_id', 'g1')
    if model_id not in ('g1', 'g1-tools'):
        model_id = 'g1'
    return Robot(model_id, _preferred_tools() if model_id == 'g1-tools' else None)


def _save_selection(candidate):
    pair = candidate.tools or robot.tools or _preferred_tools()
    value = {'model_id': candidate.model_id, 'tools': list(pair)}
    temporary = SELECTION_PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, SELECTION_PATH)


robot = _initial_robot()
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


def _replace_robot(candidate):
    """Swap the server's robot; refuse while WBC sessions or physics jobs use the current one."""
    global robot
    from .policy_preview import jobs
    from .decoupled_wbc import sessions
    with sessions.lock, jobs.lock:
        if sessions.sessions or any(job['status'] == 'running' for job in jobs.jobs.values()):
            raise HTTPException(409, 'WBC 세션을 닫고 물리 계산을 완료한 뒤 모델을 변경하세요.')
        try:
            candidate.export_visual(ROOT / f'assets/g1/robot-{candidate.model_id}.glb')
        except (ValueError, OSError) as exc:
            raise HTTPException(422, detail=f'모델을 불러오지 못했습니다: {exc}') from exc
        try:
            _save_selection(candidate)
        except OSError as exc:
            raise HTTPException(500, detail=f'장착 툴 선택을 저장하지 못했습니다: {exc}') from exc
        if jobs.sonic_process is not None:
            jobs.sonic_process.terminate()
            jobs.sonic_process.wait(timeout=5)
            jobs.sonic_process = None
        os.environ['MOTIONCREATOR_MODEL'] = candidate.model_id
        if candidate.tools is not None:
            os.environ['MOTIONCREATOR_TOOLS'] = ':'.join(candidate.tools)
        robot = candidate
        sessions.robot = None


@app.post('/api/model')
def select_model(payload: ModelInput):
    if robot.model_id == payload.model_id:
        return {'model_id': robot.model_id}
    try:
        candidate = Robot(payload.model_id, _preferred_tools() if payload.model_id == 'g1-tools' else None)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, detail=f'모델을 불러오지 못했습니다: {exc}') from exc
    _replace_robot(candidate)
    return {'model_id': robot.model_id}


def tool_options():
    from .tool_model import TOOLS
    return [{'id': key, 'label': spec['label']} for key, spec in TOOLS.items()]


class ToolsInput(BaseModel):
    model_config = {'extra': 'forbid'}
    left: str
    right: str


@app.get('/api/tools')
def get_tools():
    return {'model_id': robot.model_id, 'tools': list(robot.tools) if robot.tools else None, 'options': tool_options()}


@app.post('/api/tools')
def select_tools(payload: ToolsInput):
    if robot.model_id != 'g1-tools':
        raise HTTPException(409, '툴 교환은 G1 그리퍼 모델에서만 사용할 수 있습니다.')
    from .tool_model import parse_tools
    try:
        tools = parse_tools(f'{payload.left}:{payload.right}')
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if tools != robot.tools:
        try:
            candidate = Robot('g1-tools', tools)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, detail=f'툴을 불러오지 못했습니다: {exc}') from exc
        _replace_robot(candidate)
    return {'model_id': robot.model_id, 'tools': list(robot.tools)}


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
    controller: Literal['pd', 'gear-sonic', 'objects'] = 'gear-sonic'
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
    retry_from_presets: bool = False


class LabPathTarget(BaseModel):
    model_config = {'extra': 'forbid'}
    targets: dict[str, list[float]]
    orientations: dict[str, list[float]]


class LabPathSolveInput(PoseInput):
    model_config = {'extra': 'forbid'}
    tools: tuple[str, str]
    pins: list[str] = Field(default_factory=lambda: list(FEET))
    angle_pins: list[str] = Field(default_factory=list)
    waypoints: list[LabPathTarget] = Field(min_length=1, max_length=100)


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


class RamenSequenceSettings(BaseModel):
    model_config = {'extra': 'forbid'}
    approach_clearance_m: float = Field(.08, ge=.01, le=.30)
    lift_height_m: float = Field(.18, ge=.02, le=.60)
    extraction_distance_m: float = Field(.20, ge=.02, le=.80)
    extraction_cycles: int = Field(3, ge=2, le=6)
    carry_tilt_deg: float = Field(18., ge=0., le=45.)
    phase_seconds: float = Field(2., ge=.2, le=10.)
    insertion_seconds: float = Field(3., ge=.2, le=15.)
    hold_seconds: float = Field(1., ge=.1, le=10.)
    force_limit_n: float = Field(35., ge=1., le=200.)
    lateral_stiffness_n_per_m: float = Field(420., ge=1., le=3000.)
    insertion_stiffness_n_per_m: float = Field(90., ge=0., le=1000.)
    translation_damping_ns_per_m: float = Field(28., ge=0., le=300.)
    orientation_stiffness_nm_per_rad: float = Field(38., ge=0., le=300.)
    orientation_damping_nms_per_rad: float = Field(4.5, ge=0., le=50.)
    maximum_feedback_torque_fraction: float = Field(.35, ge=.05, le=.8)


class SavedPoseInput(PoseInput):
    model_config = {'extra': 'forbid'}
    pins: list[str] = Field(default_factory=list, max_length=len(HANDLES))
    angle_pins: list[str] = Field(default_factory=list, max_length=len(ANGLE_LOCKABLE))


class RamenSequenceInput(PoseInput):
    model_config = {'extra': 'forbid'}
    pins: list[str] = Field(default_factory=lambda: list(FEET))
    angle_pins: list[str] = Field(default_factory=list, max_length=len(ANGLE_LOCKABLE))
    object: dict
    settings: RamenSequenceSettings = Field(default_factory=RamenSequenceSettings)
    attention_pose: SavedPoseInput | None = None


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
    return {'model_id': robot.model_id, 'visual_revision': robot.visual_revision,
            'state': robot.state(robot.home), 'project': new_project(robot), 'joint_names': robot.names,
            'tools': list(robot.tools) if robot.tools else None, 'tool_options': tool_options(),
            'limits': robot.model.jnt_range[1:].tolist()}


@app.get('/api/attention')
def attention():
    return {'state': robot.state(robot.attention)}


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


@app.post('/api/solve-group')
def solve_group(payload: GroupSolveInput):
    def run():
        if not any((payload.targets, payload.orientations, payload.joints)):
            raise ValueError('At least one position, rotation or joint target is required')
        def attempt(start):
            return robot.solve(start, payload.anchor, pins=payload.pins,
                               resistance=payload.resistance, mode=payload.mode, selected_targets=payload.targets,
                               orientation_targets=payload.orientations, joint_targets=payload.joints,
                               angle_pins=payload.angle_pins)
        q, info = attempt(payload.qpos)
        info['restart'] = 'current'
        if payload.retry_from_presets and not info['converged']:
            # A local solve can get stuck from an awkward pose (e.g. raised arms). Retry from
            # the default stand and attention poses, keeping the current base, and keep the best.
            current = robot.validate_q(payload.qpos)
            for label, preset in (('default_stand', robot.home), ('attention', robot.attention)):
                start = preset.copy()
                start[:7] = current[:7]
                candidate = attempt(start)
                candidate[1]['restart'] = label
                score = lambda result: (not result[1]['converged'], result[1]['rejected'],
                                        result[1]['target_error_mm'] + result[1]['angle_error_deg'])
                if score(candidate) < score((q, info)):
                    q, info = candidate
                if info['converged']:
                    break
        return {'state': robot.state(q), 'solver': info}
    return checked(run)


@app.post('/api/grasp-fit')
def grasp_fit(payload: GraspFitInput):
    from .grasp import fit_two_hand_grasp
    return checked(lambda: fit_two_hand_grasp(robot, payload.qpos, payload.pins, payload.object,
                                              payload.grasp, payload.angle_pins))


@app.post('/api/ramen-sequence')
def ramen_sequence(payload: RamenSequenceInput):
    from .ramen_sequence import plan_ramen_sequence
    return checked(lambda: plan_ramen_sequence(robot, payload.qpos, payload.pins, payload.angle_pins,
                                               payload.object, payload.settings.model_dump(),
                                               payload.attention_pose.model_dump() if payload.attention_pose else None))


@app.post('/api/preview')
def preview(payload: ProjectInput):
    def run():
        motion = compile_motion(robot, payload.project, min(payload.fps, 30))
        return {'time': motion['time'].tolist(), 'states': [robot.state(q) for q in motion['qpos']],
                'max_pin_error_mm': motion['max_pin_error_mm']}
    return checked(run)


@app.websocket('/api/preview/stream')
async def preview_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        project = payload['project']
        fps = min(int(payload.get('fps', 30)), 30)
        if not 1 <= fps <= 30:
            raise ValueError('재생 FPS는 1–30이어야 합니다.')
    except (ValueError, KeyError, TypeError) as exc:
        await websocket.send_json({'type': 'error', 'message': str(exc)})
        await websocket.close()
        return
    active_robot = robot
    loop = asyncio.get_running_loop()
    packets = asyncio.Queue()
    cancelled = threading.Event()

    def emit(times, poses):
        if cancelled.is_set():
            raise InterruptedError('키프레임 스트리밍 중단')
        for offset in range(0, len(times), 10):
            if cancelled.is_set():
                raise InterruptedError('키프레임 스트리밍 중단')
            end = offset + 10
            packet = {'type': 'batch', 'time': times[offset:end],
                      'states': [active_robot.state(pose) for pose in poses[offset:end]]}
            asyncio.run_coroutine_threadsafe(packets.put(packet), loop).result()

    def compile_preview():
        try:
            motion = compile_motion(active_robot, project, fps, on_segment=emit)
            packet = {'type': 'done', 'max_pin_error_mm': float(motion['max_pin_error_mm'])}
        except InterruptedError:
            return
        except Exception as exc:
            packet = {'type': 'error', 'message': str(exc)}
        asyncio.run_coroutine_threadsafe(packets.put(packet), loop).result()

    async def watch_disconnect():
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            cancelled.set()
            await packets.put({'type': 'closed'})

    worker = asyncio.create_task(asyncio.to_thread(compile_preview))
    disconnect = asyncio.create_task(watch_disconnect())
    try:
        while True:
            packet = await packets.get()
            if packet['type'] == 'closed':
                break
            await websocket.send_json(packet)
            if packet['type'] in ('done', 'error'):
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        cancelled.set()
        disconnect.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass


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


@app.websocket('/api/policy-preview/{identifier}/stream')
async def policy_stream(websocket: WebSocket, identifier: str):
    from .policy_preview import jobs
    try:
        path = jobs.stream_path(identifier)
    except KeyError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    source = None
    last_progress = -1.
    stream_origin = None
    try:
        while True:
            if source is None and path.is_file():
                source = path.open(encoding='utf-8')
            if source is not None:
                while True:
                    position = source.tell()
                    line = source.readline()
                    if not line.endswith('\n'):
                        source.seek(position)
                        break
                    frame = json.loads(line)
                    if stream_origin is None:
                        stream_origin = time.monotonic() - frame['time']
                    delay = stream_origin + frame['time'] - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await websocket.send_text(line)
            status = jobs.status(identifier)
            if status['progress'] != last_progress or status['status'] != 'running':
                await websocket.send_json({'type': 'status', **status})
                last_progress = status['progress']
            if status['status'] != 'running':
                break
            await asyncio.sleep(.05)
    except (WebSocketDisconnect, OSError, KeyError):
        pass
    finally:
        if source is not None:
            source.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.post('/api/policy-preview/{identifier}/cancel')
def policy_cancel(identifier: str):
    from .policy_preview import jobs
    return checked(lambda: jobs.cancel(identifier))


@app.post('/api/save')
def save(payload: ProjectInput):
    return checked(lambda: save_bundle(robot, payload.project, payload.fps, protomotions=payload.protomotions))


class RenameInput(BaseModel):
    model_config = {'extra': 'forbid'}
    project_id: str
    name: str = Field(max_length=80)


@app.post('/api/rename-project')
def rename_project_endpoint(payload: RenameInput):
    return checked(lambda: rename_project(payload.project_id, payload.name))


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
        validation_robot = robot if not project.get('mounted_tools') or robot.model_id == 'g1-tools' else Robot('g1-tools', _preferred_tools())
        return validate_project(validation_robot, project)
    return checked(load)


class CylinderSceneInput(BaseModel):
    project: dict
    target_id: str | None = None


class CylinderPlanInput(BaseModel):
    scene: dict
    positions: dict[str, list[float]] = Field(default_factory=dict)
    params: dict[str, float] = Field(default_factory=dict)


@app.post('/api/cylinder-lab/scene')
def cylinder_lab_scene(payload: CylinderSceneInput):
    from .cylinder_extraction import DEFAULT_PARAMS, LIMITS, scene_2d
    def run():
        scene = scene_2d(payload.project, payload.target_id)
        return {'scene': scene, 'defaults': DEFAULT_PARAMS, 'limits': LIMITS,
                'targets': [{'id': c['id'], 'name': c['name']} for c in scene['circles']]}
    return checked(run)


@app.post('/api/cylinder-lab/solve-path')
def cylinder_lab_solve_path(payload: LabPathSolveInput):
    def run():
        active_robot = robot
        if active_robot.model_id != 'g1-tools':
            raise ValueError('장착 툴 경로는 G1 그리퍼 모델에서만 생성할 수 있습니다.')
        if tuple(active_robot.tools or ()) != payload.tools:
            raise ValueError('장착 툴이 변경되었습니다. 메인 에디터를 새로고침한 뒤 다시 시도하세요.')
        start = active_robot.validate_q(payload.qpos)
        seed = start.copy()
        # Keep the authored stance while the hands follow the path. Otherwise the
        # whole-body solve can move the pelvis far enough to make a reachable-looking
        # hand target produce an implausible full-body pose.
        stance_pins = tuple(dict.fromkeys([*payload.pins, 'pelvis']))
        frames = []
        diagnostics = []
        hands = {'left_hand', 'right_hand'}
        for index, waypoint in enumerate(payload.waypoints, 1):
            active_hands = set(waypoint.targets)
            if not active_hands or not active_hands <= hands or set(waypoint.orientations) != active_hands:
                raise ValueError('각 경로 지점에는 경로가 있는 손의 TCP 위치와 방향이 필요합니다.')

            def attempt(initial, posture_weight, evaluations):
                return active_robot.solve(
                    initial, start, pins=stance_pins, angle_pins=payload.angle_pins,
                    mode='free', resistance=0., selected_targets=waypoint.targets,
                    orientation_targets=waypoint.orientations, posture_reference=seed,
                    posture_weight=posture_weight, max_nfev=evaluations,
                )

            solved, info = attempt(seed, .8, 180)
            if not info['converged']:
                # Follow the reached posture into the next local solve, then relax the preference
                # for the previous joints. TCP and stance tolerances remain unchanged.
                for weight, evaluations in ((.3, 280), (.08, 400)):
                    candidate, candidate_info = attempt(solved if not info['rejected'] else seed, weight, evaluations)
                    if (candidate_info['converged'] or
                            (not info['converged'] and not candidate_info['rejected'] and
                             candidate_info['target_error_mm'] < info['target_error_mm'])):
                        solved, info = candidate, candidate_info
                    if info['converged']:
                        break
            if not info['converged'] or info['rejected']:
                errors = ' · '.join(f'{"왼손" if hand == "left_hand" else "오른손"} {error:.1f} mm'
                                    for hand, error in sorted(info['target_errors_mm'].items()))
                raise ValueError(f'파지법 {index}/{len(payload.waypoints)}번째 지점 IK 실패: '
                                 f'위치 {errors} · 방향 {info["angle_error_deg"]:.1f}°. '
                                 '골반과 고정된 부위를 유지할 수 없어 키프레임을 추가하지 않았습니다. '
                                 '해당 그린 점을 로봇 쪽으로 옮기거나 시작 자세를 조정해 주세요.')
            frames.append(solved.tolist())
            diagnostics.append(info)
            seed = solved
        if robot is not active_robot:
            raise ValueError('경로 계산 중 장착 툴이 변경되었습니다. 다시 시도하세요.')
        return {'qpos': frames, 'first_state': active_robot.state(frames[0]), 'diagnostics': diagnostics}
    return checked(run)


@app.get('/api/cylinder-lab/tool-profile')
def cylinder_lab_tool_profile(side: Literal['left', 'right'] = Query(...)):
    from .tool_section import tool_section
    return checked(lambda: tool_section(robot, side))


@app.post('/api/cylinder-lab/plan')
def cylinder_lab_plan(payload: CylinderPlanInput):
    from .cylinder_extraction import plan_scoop, with_positions
    def run():
        scene = with_positions(payload.scene, payload.positions) if payload.positions else payload.scene
        return {'scene': scene, 'path': plan_scoop(scene, payload.params)}
    return checked(run)


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
    # Fork the compile workers now, while the process is still single-threaded.
    enable_parallel_compile(robot)
    frontend = ROOT / 'frontend/dist'
    if (frontend / 'index.html').exists():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='editor')
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
