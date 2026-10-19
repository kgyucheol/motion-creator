import asyncio
import json
import numpy as np
import os
import struct
import threading
import time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .robot import Robot, ROOT, FEET, HANDLES, ROTATABLE, ANGLE_LOCKABLE
from .motion import new_project, validate_project, compile_motion, rename_project, enable_parallel_compile, project_from_motion_bytes, prepare_saved_project, save_bundle
from .presets import GroupStore
from .head_camera import detected_objects, head_camera, point_cloud

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
from .controller_api import router as controller_router
app.include_router(controller_router)
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


@app.get('/api/head-camera/status')
def head_camera_status():
    return head_camera.status()


def streaming_frames(source, wrap):
    """수신 스레드를 구독해 최신 프레임만 `wrap`으로 감싸 내보내는 StreamingResponse 본문."""
    try:
        source.acquire()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def frames():
        sequence = 0
        try:
            while True:
                packet = await asyncio.to_thread(source.wait_frame, sequence, 1.0)
                if packet is None:
                    continue
                sequence, payload = packet
                yield wrap(payload)
        finally:
            source.release()

    return frames()


@app.get('/api/head-camera/stream')
async def head_camera_stream():
    wrap = lambda jpeg: (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg)) + jpeg + b'\r\n'
    return StreamingResponse(streaming_frames(head_camera, wrap), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})


@app.get('/api/head-camera/cloud/status')
def point_cloud_status():
    return point_cloud.status()


@app.get('/api/head-camera/cloud/stream')
async def point_cloud_stream():
    """포인트 클라우드 메시지를 `uint32 길이 + 본문` 단위로 이어 보낸다."""
    wrap = lambda message: struct.pack('<I', len(message)) + message
    return StreamingResponse(streaming_frames(point_cloud, wrap), media_type='application/octet-stream', headers={'Cache-Control': 'no-store'})


@app.get('/api/head-camera/objects/status')
def detected_objects_status():
    return detected_objects.status()


@app.get('/api/head-camera/objects/stream')
async def detected_objects_stream():
    """로봇이 인식한 용기 줄(JSON)을 `uint32 길이 + 본문` 단위로 이어 보낸다."""
    wrap = lambda message: struct.pack('<I', len(message)) + message
    return StreamingResponse(streaming_frames(detected_objects, wrap), media_type='application/octet-stream', headers={'Cache-Control': 'no-store'})


@app.get('/api/init')
def initialize():
    return {'model_id': robot.model_id, 'visual_revision': robot.visual_revision,
            'state': robot.state(robot.home), 'arm_geoms': robot.arm_geom_ids, 'project': new_project(robot), 'joint_names': robot.names,
            'tools': list(robot.tools) if robot.tools else None, 'tool_options': tool_options(),
            'limits': robot.model.jnt_range[1:].tolist()}


@app.get('/api/attention')
def attention():
    return {'state': robot.state(robot.attention)}


@app.get('/api/wbc-stand')
def wbc_stand():
    return {'state': robot.state(robot.wbc_stand)}


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
async def import_motion(request: Request, filename: str, fps: int = Query(30, ge=1, le=120),
                        source: str = Query('actions', pattern='^(actions|states)$')):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in ('.npz', '.csv', '.json'):
        raise HTTPException(status_code=422, detail='Only .npz, .csv and LeRobot .json motion files are supported')
    content = await request.body()
    if not content:
        raise HTTPException(status_code=422, detail='Motion file is empty')
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Motion file must be 100 MB or smaller')
    return checked(lambda: project_from_motion_bytes(robot, content, filename, fps, source))


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


class DynamicResolveInput(BaseModel):
    project: dict
    index: int = Field(ge=0)


@app.post('/api/dynamic-resolve')
def dynamic_resolve(payload: DynamicResolveInput):
    """The samples a dynamic clip plays with the scene object where it is now, for baking them into poses."""
    def run():
        from .dynamic_clip import resolve_project_dynamics
        validate_project(robot, payload.project)
        frames = payload.project['keyframes']
        if payload.index >= len(frames) or not frames[payload.index].get('dynamic'):
            raise ValueError('Selected keyframe is not a dynamic clip')
        resolved, reports = resolve_project_dynamics(robot, payload.project)
        return {'samples': resolved['keyframes'][payload.index]['samples'], 'report': reports[payload.index]}
    return checked(run)


# ---- robot replay (Unitree default policy, arms over rt/arm_sdk) -------------------------------------------------------------
_replay = None


def replay_session():
    global _replay
    if _replay is None:
        from .replay.session import ReplaySession
        _replay = ReplaySession(lambda: robot, read_only=os.environ.get('MOTIONCREATOR_REPLAY_READ_ONLY') == '1')
        import atexit
        atexit.register(_replay.shutdown)       # hand the arms back if the server is stopped mid-replay
    return _replay


def replay_checked(fn):
    from .replay.backends import CommandError
    try:
        return fn()
    except CommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ReplayConnectInput(BaseModel):
    interface: str | None = None
    mock: bool = False


class ReplayPrepareInput(BaseModel):
    project: dict
    fps: int = Field(30, ge=1, le=60)


class ReplayStartInput(BaseModel):
    speed: float = 1.0
    auto_return: bool = True
    allow_fast: bool = False
    confirm: bool = False


@app.get('/api/replay/status')
def replay_status():
    return replay_checked(lambda: replay_session().status())


@app.get('/api/replay/interfaces')
def replay_interfaces():
    return replay_session().interfaces()


@app.post('/api/replay/connect')
def replay_connect(payload: ReplayConnectInput):
    def run():
        replay_session().connect(payload.interface, payload.mock)
        return replay_session().status()
    return replay_checked(run)


@app.post('/api/replay/disconnect')
def replay_disconnect():
    def run():
        replay_session().disconnect()
        return replay_session().status()
    return replay_checked(run)


@app.post('/api/replay/prepare')
def replay_prepare(payload: ReplayPrepareInput):
    return replay_checked(lambda: replay_session().prepare(payload.project, payload.fps))


@app.post('/api/replay/start')
def replay_start(payload: ReplayStartInput):
    return replay_checked(lambda: replay_session().start(payload.speed, payload.auto_return, payload.allow_fast, payload.confirm))


@app.post('/api/replay/pause')
def replay_pause(on: bool = True):
    return replay_checked(lambda: replay_session().pause(on))


@app.post('/api/replay/stop')
def replay_stop():
    return replay_checked(lambda: replay_session().stop())


@app.post('/api/replay/home')
def replay_home():
    return replay_checked(lambda: replay_session().home())


@app.post('/api/replay/speed')
def replay_speed(speed: float):
    return replay_checked(lambda: replay_session().set_speed(speed))


@app.post('/api/export-lerobot')
def export_lerobot(payload: ProjectInput):
    """The compiled motion as a LeRobot data.json (arms and waist) for robot-studio or unitree_lerobot."""
    def run():
        from .lerobot_format import qpos_to_lerobot_episode
        from .replay.plan import resample
        motion = compile_motion(robot, payload.project, payload.fps)
        qpos = resample(motion['time'], motion['qpos'], payload.fps)
        return qpos_to_lerobot_episode(qpos, payload.fps, payload.project.get('name', ''))
    return checked(run)


class DynamicOffsetInput(BaseModel):
    project: dict
    index: int = Field(ge=0)
    sample: int = Field(ge=0)
    qpos: list[float]


@app.post('/api/dynamic-offset')
def dynamic_offset(payload: DynamicOffsetInput):
    """The TCP offset that carries a dynamic clip's grip to where the edited pose has its hands."""
    def run():
        from .dynamic_clip import capture_tcp_offset
        validate_project(robot, payload.project)
        frames = payload.project['keyframes']
        if payload.index >= len(frames) or not frames[payload.index].get('dynamic'):
            raise ValueError('Selected keyframe is not a dynamic clip')
        return capture_tcp_offset(robot, payload.project, payload.index, payload.sample, payload.qpos)
    return checked(run)


def _original_states(motion, active_robot=None):
    """Per frame the pose without the dynamic grasp correction, or None where it equals the corrected one."""
    original = motion.get('original_qpos')
    if original is None:
        return None
    active_robot = active_robot or robot
    return [None if np.allclose(before, after, atol=1e-6) else active_robot.state(before)
            for before, after in zip(original, motion['qpos'])]


class GraspTemplateInput(BaseModel):
    object_position: list[float] = Field(min_length=3, max_length=3)
    object_yaw_deg: float = 0.
    carry_target: list[float] | None = Field(default=None, min_length=3, max_length=3)
    fps: int = Field(30, ge=1, le=120)


@app.post('/api/grasp-template/generate')
def generate_grasp_template_motion(payload: GraspTemplateInput):
    """The symmetric two-hand pick motion for an object on the body's centre line: one clip per phase plus a report each."""
    from .grasp_template import generate_motion, load_template
    return checked(lambda: dict(zip(('keyframes', 'reports'), generate_motion(
        robot, load_template(), payload.object_position, payload.object_yaw_deg, payload.fps, payload.carry_target))))


class KeyframesToClipInput(BaseModel):
    project: dict
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    fps: int = Field(30, ge=1, le=30)


@app.post('/api/keyframes-to-clip')
def keyframes_to_clip(payload: KeyframesToClipInput):
    """Join cards `start`…`end` into one motion clip as they were recorded (motion modes left out).

    The caller gives the clip the motion of the last card, so the end of the joined clip is retargeted like that card was.
    """
    def run():
        frames = payload.project['keyframes']
        if payload.end >= len(frames) or payload.end <= payload.start:
            raise ValueError('Pick two or more cards in order')
        recorded = {**payload.project, 'keyframes': [{key: value for key, value in frame.items() if key not in ('motion', 'dynamic')}
                                                     for frame in frames]}
        motion = compile_motion(robot, recorded, payload.fps)
        times = np.asarray(motion['time'])
        starts = motion['keyframe_times']
        begin = starts[payload.start]
        last = frames[payload.end]
        finish = starts[payload.end] + (float(last['duration']) if last.get('samples') is not None else 0.)
        first_index = int(np.searchsorted(times, begin - 1e-6))
        last_index = int(np.searchsorted(times, finish + 1e-6, side='right')) - 1
        samples = np.asarray(motion['qpos'])[first_index:last_index + 1]
        if len(samples) < 2:
            raise ValueError('The picked cards are too short to become a clip')
        entry = frames[payload.start]
        clip = {'name': entry['name'] if payload.start == payload.end - 1 else f"{entry['name']} → {last['name']}",
                'duration': (len(samples) - 1) / payload.fps, 'qpos': samples[0].tolist(), 'pins': [], 'samples': samples.tolist()}
        arrival = entry.get('transition') if entry.get('samples') is not None else entry.get('duration')
        if payload.start > 0 and arrival is not None:
            clip['transition'] = float(arrival)
        return clip
    return checked(run)


class CarryEditInput(BaseModel):
    project: dict
    index: int = Field(ge=0)
    qpos: list[float]


@app.post('/api/carry-edit')
def carry_edit_endpoint(payload: CarryEditInput):
    """Store an edited pose and move the cards linked after it (앞 카드 이동 연결) so they continue from its new end."""
    def run():
        from .card_motion import carry_edit
        keyframes, reports = carry_edit(robot, payload.project, payload.index, payload.qpos)
        return {'keyframes': keyframes, 'reports': {str(key): value for key, value in reports.items()}}
    return checked(run)


class HandsOnRowInput(BaseModel):
    qpos: list[float]
    object: dict
    placement: dict[str, list[float]] | None = None
    frame: str = 'object'


@app.post('/api/hands-on-row')
def hands_on_row_endpoint(payload: HandsOnRowInput):
    """Read (or set) where each hand is relative to its end of a lying row object."""
    def run():
        from .card_motion import hands_on_row
        return hands_on_row(robot, payload.qpos, payload.object, payload.placement, payload.frame)
    return checked(run)


@app.post('/api/preview')
def preview(payload: ProjectInput):
    def run():
        motion = compile_motion(robot, payload.project, min(payload.fps, 30))
        return {'time': motion['time'].tolist(), 'states': [robot.state(q) for q in motion['qpos']],
                'max_pin_error_mm': motion['max_pin_error_mm'], 'keyframe_times': motion.get('keyframe_times'),
                'dynamic_reports': motion.get('dynamic_reports', []), 'motion_reports': motion.get('motion_reports', []),
                'original_states': _original_states(motion)}
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
            packet = {'type': 'done', 'max_pin_error_mm': float(motion['max_pin_error_mm']), 'keyframe_times': motion.get('keyframe_times'),
                      'dynamic_reports': motion.get('dynamic_reports', []), 'motion_reports': motion.get('motion_reports', []),
                      'original_states': _original_states(motion, active_robot)}
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
    parser.add_argument('--read-only', action='store_true', help='Never connect to a real robot for replay (the simulated robot still works)')
    args = parser.parse_args()
    if args.read_only:
        os.environ['MOTIONCREATOR_REPLAY_READ_ONLY'] = '1'
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
