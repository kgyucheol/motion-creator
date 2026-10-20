'use client';
import NumericInput from './numeric-input';
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import SceneObjectControls from './scene-object-controls';
import GraspControls, { type GraspPickMode } from './grasp-controls';
import RamenSequenceControls, { type RamenSequenceSettings } from './ramen-sequence-controls';
import BodyControls from './body-controls';
import { allNodes, nodeMembers, selectMembers, controlKey, controlSelection, groupForControl, visibleTreeHandles, expandVirtualControls } from '../lib/body-groups';
import { Play, Pause, Plus, Save, FolderOpen, RotateCcw, Undo2, Redo2, LockKeyhole, MousePointer2, Move3d, ChevronLeft, ChevronRight, ChevronDown, Trash2, Download, Check, AlertCircle, Boxes, PersonStanding, ArrowDownToLine, Scissors } from 'lucide-react';
import { ReplayPanel, RobotStatusChip } from './ReplayPanel';
import { RobotScene, canRotateSelection, isJointHandle, HIP_HANDLES, ANKLE_HANDLES, COMBINED_JOINTS, type PoseState, type TransformMode } from '../lib/robot-scene';
import { eulerDegrees, fromLocalAxes, toLocalAxes, quaternionFromDegrees, rotatedGroupTargets, incrementRotation, canMirrorSelection, counterpart, isJointPairSelection, mirrorFrameHandle, pairedJointTargets, translatedTargets } from '../lib/pose-transforms';
import type { WorldRow } from '../lib/detected-objects';
import { candidateRamenGraspPoints, createImportedSceneObjects, createSceneObject, createSceneObjectGroup, DEFAULT_RAMEN_GRASP_COEFFICIENTS, groundedSceneObject, isRamenBundle, normalizedObjectSize, objectVerticalHalfExtent, objectsFromProject, placeSceneObject, removeSceneObjectSubtree, reparentSceneObjects, savedRamenGraspCoefficients, sceneObjectDescendantIds, scenePlacementOptions, transformSceneObjectChildren, transformSceneObjectGroup, withScenePlacement, type ObjectTransformMode, type RamenGraspCoefficients, type SceneAssetImport, type SceneObject, type SceneObjectGroup, type SceneObjectPose, type SceneObjectShape, type ScenePlacementOptions } from '../lib/scene-objects';
import { duplicateKeyframeKeepingTimes, insertPoseAfterClip, insertSmoothTransitionFrames, keyframeStartTimes, moveKeyframe, outgoingSeconds, mergeIntoNextPose, removeKeyframe, retimeSmoothTransition, type Keyframe, type TwoHandGrasp, clipSampleAtTime, clipSampleIndexAtTime, convertClipToPoses, defaultDynamicWindow, setDynamicWindowValue, DYNAMIC_WINDOW_KEYS, type DynamicGrasp, type DynamicReport, type CardMotion, type MotionReport, mergeMotionClips, middleSample, splitMotionClip } from '../lib/keyframes';
import { buildLabPathPreview, readLabToolPaths, toolPoseAtKeyPoint, withLabLateralOffsets, type LabLateralOffsets, type LabPathPreview, type HandSide } from '../lib/lab-path-apply';
import { authoredSignature, labGroupRange, labSourceSignature, legacyLabRange } from '../lib/lab-grasp-group';
import { restingScene, type Scene as CylinderScene } from '../lib/cylinder-sim';

type SavedPose = { qpos: number[]; pins: string[]; angle_pins: string[] };
type ObjectTarget = Extract<NonNullable<NonNullable<Keyframe['motion']>['target']>, { kind: 'object' }>;
const SCENE_FORMAT = 'motioncreator.scene.v1';
/** 인식한 라면 용기 한 줄: 한 묶음 약 800g, 안이 빈 얇은 벽(3mm). 충돌은 벽이 손가락을 막지 않도록 축 위의 가는 선(6mm)으로 둔다. */
const ROW_MASS_KG = .8, ROW_WALL_M = .003, ROW_LINE_M = .006;
type SceneFile = { format: typeof SCENE_FORMAT; scene_objects: SceneObject[]; scene_groups?: SceneObjectGroup[] };
type Project = { format: string; name: string; name_mode?: 'auto' | 'manual'; display_name?: string; project_id?: string; created_at?: string; model_sha256: string; mounted_tools?: [string, string] | null; joint_names: string[]; coordinate_system: string; units: Record<string, string>; keyframes: Keyframe[]; current_qpos?: number[]; pins?: string[]; angle_pins?: string[]; attention_pose?: SavedPose; scene_objects?: SceneObject[]; scene_groups?: SceneObjectGroup[]; lab_lateral_offsets?: Record<string, LabLateralOffsets>; box?: { position: number[]; size: number[]; visible: boolean } };
type GraspSummary = { object_id: string; bilateral_contact: boolean; target_reached: boolean; force_limit_exceeded: boolean; target_force_n: number; max_force_n: number; max_penetration_m: number; left: { max_normal_n: number; contact_samples: number }; right: { max_normal_n: number; contact_samples: number } };
type InteractionSummary = { active: boolean; max_tcp_error_mm: number; max_orientation_error_deg: number; max_feedback_torque_nm: number; max_contact_force_n: Record<'left' | 'right', number>; force_limit_exceeded: boolean };
type Preview = { time: number[]; keyframe_times?: number[] | null; dynamic_reports?: DynamicReport[]; motion_reports?: MotionReport[]; original_states?: (PoseState | null)[] | null; states: PoseState[]; object_states?: Record<string, SceneObjectPose>[]; max_pin_error_mm: number; physics?: boolean; controller?: 'pd' | 'gear-sonic' | 'objects'; summary?: { reason: string; joint_rmse_rad: number; sim_seconds: number; reference_seconds: number; start_frame_index?: number; start_frame_name?: string; grasp?: GraspSummary | null; interaction?: InteractionSummary | null } };
type PolicyJob = { id: string; status: 'running' | 'completed' | 'cancelled' | 'failed'; progress: number; message?: string };
type PhysicsStreamFrame = { type: 'frame'; time: number; state: PoseState; objects: Record<string, SceneObjectPose> };
type SolveInfo = { target_error_mm: number; pin_error_mm: number; rejected: boolean; converged: boolean; target_errors_mm?: Record<string, number>; angle_error_deg?: number };
type GroupPreset = { id: string; name: string; members: string[] };
type EditorSnapshot = { qpos: number[]; pins: string[]; anglePins: string[]; objects: SceneObject[]; objectGroups: SceneObjectGroup[]; keyframes: Keyframe[]; frameIndex: number; poseDirty: boolean };
const feet = ['left_foot', 'right_foot'];
function dynamicStatusText(report: DynamicReport | null) {
  if (!report) return '계산 중…';
  const error = report.max_position_error_mm !== undefined ? ` · ${report.max_position_error_mm.toFixed(1)}mm` : '';
  switch (report.status) {
    case 'unchanged': return '기준 위치 그대로';
    case 'ok': return `도달${error}`;
    case 'approx': return `근사${error}`;
    case 'unreachable': return `도달 불가${error}`;
    case 'too_far': return `너무 멂 · ${(report.shift_mm ?? 0).toFixed(0)}mm (한도 ${(report.max_shift_mm ?? 0).toFixed(0)}mm)`;
    default: return '물체를 찾을 수 없음';
  }
}
const formatClock = (seconds: number) => `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
function cardOutgoingText(keyframes: Keyframe[], index: number) {
  const out = outgoingSeconds(keyframes, index);
  if (out.kind === 'none') return '마지막 카드';
  if (out.kind === 'joined') return '다음 클립과 이어짐';
  return `다음까지 ${out.seconds === undefined ? '자동' : `${out.seconds.toFixed(1)}s`}`;
}
const GRASP_GHOST_ID = '__grasp_keyframe_ghost__';
const GRASP_COEFFICIENTS_STORAGE_KEY = 'motioncreator-editor-ramen-grasp-v1';
const labLateralKey = (targetId: string, mounted: [string, string]) => `${targetId}|${mounted.join('|')}`;
const labLateralOffsets = (project: Project | null, targetId: string, mounted: [string, string]): LabLateralOffsets =>
  project?.lab_lateral_offsets?.[labLateralKey(targetId, mounted)] ?? {};
// Keep the experimental planner available in code while users author each phase manually.
const SHOW_AUTOMATIC_RAMEN_SEQUENCE = false;
const genericFrameNames = new Set(['stand', 'standing', 'pose', 'frame', 'keyframe', 'start', 'start pose', 'imported motion clip', '서기', '서있기', '기본 서기', '기본 서기 자세', '자세', '키프레임', '시작', '시작 자세']);
function objectAtGraspPose(frame: Keyframe | undefined, objects: SceneObject[]) {
  const grasp = frame?.grasp;
  const object = objects.find(value => value.id === grasp?.object_id);
  if (!grasp || !object) return undefined;
  return grasp.object_pose ? { ...object, position: [...grasp.object_pose.position],
    quaternion_xyzw: [...grasp.object_pose.quaternion_xyzw] } : object;
}
function graspGhost(frame: Keyframe | undefined, objects: SceneObject[]) {
  const target = objectAtGraspPose(frame, objects);
  if (!target || !frame?.grasp?.object_pose) return undefined;
  return { ...target, id: GRASP_GHOST_ID, name: `${target.name} · 키프레임 고스트`, color: '#56d8e4',
    opacity: .24, visible: true, placement: { prevent_overlap: false, surface_snap: false, ground_lock: false }, ghost: true };
}
function withLegacyGraspGhosts(project: Project, objects: SceneObject[]) {
  return { ...project, keyframes: project.keyframes.map(frame => {
    const grasp = frame.grasp;
    const object = objects.find(value => value.id === grasp?.object_id);
    if (!grasp?.object_signature || grasp.object_pose || !object) return frame;
    return { ...frame, grasp: { ...grasp, object_pose: { position: [...object.position],
      quaternion_xyzw: [...object.quaternion_xyzw] } } };
  }) };
}
function automaticProjectName(project: Project) {
  const names: string[] = [];
  for (const frame of project.keyframes) {
    const name = frame.name.trim().replace(/\s+/g, ' ');
    if (!name || genericFrameNames.has(name.toLocaleLowerCase()) || names.some(value => value.toLocaleLowerCase() === name.toLocaleLowerCase())) continue;
    names.push(name);
    if (names.length === 3) break;
  }
  const core = (names.join('-') || 'G1_모션').replace(/[\\/:*?"<>|]+/g, '_').replace(/^[ ._-]+|[ ._-]+$/g, '').slice(0, 60) || 'G1_모션';
  const date = project.created_at?.slice(0, 10).replaceAll('-', '') || new Date().toISOString().slice(0, 10).replaceAll('-', '');
  return `${core}_${date}`;
}
function selectionCenter(pose: PoseState, members: string[]) {
  const controls = controlSelection(members);
  return [0, 1, 2].map(i => controls.reduce((sum, key) => sum + pose.handles[key].position[i], 0) / controls.length);
}
async function api<T>(path: string, body?: unknown, method = body === undefined ? 'GET' : 'POST'): Promise<T> {
  const response = await fetch(`/api/${path}`, { method, ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { detail?: unknown };
    throw new Error(typeof error.detail === 'string' ? error.detail : `요청 실패 (${response.status})`);
  }
  return response.json();
}
const fileApiPath = (name: string) => name.split('/').map(encodeURIComponent).join('/');

export default function Editor() {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<RobotScene | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const addClipInput = useRef<HTMLInputElement>(null);
  const assetFile = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<PoseState | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [modelId, setModelId] = useState('g1');
  const [tools, setTools] = useState<[string, string] | null>(null);
  const [toolOptions, setToolOptions] = useState<{ id: string; label: string }[]>([]);
  const draftKey = useRef('g1-motion-draft-v1');
  // Swapping a hand's tool keeps the draft, like switching models: the server rebuilds the robot and the page reloads.
  async function switchTool(side: 'left' | 'right', next: string) {
    if (!tools || !project || !state || tools[side === 'left' ? 0 : 1] === next) return;
    setBusy(true);
    try {
      const nextTools: [string, string] = side === 'left' ? [next, tools[1]] : [tools[0], next];
      await api('tools', { left: nextTools[0], right: nextTools[1] });
      localStorage.setItem(draftKey.current, JSON.stringify({ ...project, mounted_tools: nextTools,
        current_qpos: state.qpos, pins, angle_pins: anglePins, scene_objects: objects, scene_groups: objectGroups }));
      location.reload();
    } catch (error) {
      setError(error instanceof Error ? error.message : '툴 교환 실패');
      setBusy(false);
    }
  }
  async function switchModel(next: string) {
    if (next === modelId || !project || !state) return;
    setBusy(true);
    try {
      const draft = JSON.stringify({ ...project, mounted_tools: tools, current_qpos: state.qpos, pins, angle_pins: anglePins, scene_objects: objects, scene_groups: objectGroups });
      localStorage.setItem(draftKey.current, draft);
      await api('model', { model_id: next });
      location.reload();
    } catch (error) {
      setError(error instanceof Error ? error.message : '모델 변경 실패');
      setBusy(false);
    }
  }
  const [selected, setSelected] = useState('pelvis');
  const [members, setMembers] = useState<string[]>(['pelvis']);
  const [expanded, setExpanded] = useState<string[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('g1-body-tree-expanded') ?? 'null');
      if (Array.isArray(saved)) return saved.filter(id => allNodes().some(node => node.id === id));
    } catch { /* Use initial accordion state when storage is unavailable. */ }
    return ['shoulders', 'elbows', 'hands', 'waist_group', 'legs', 'feet'];
  });
  const [groups, setGroups] = useState<GroupPreset[]>([]);
  const [groupName, setGroupName] = useState('');
  const [groupId, setGroupId] = useState('');
  const [pins, setPins] = useState<string[]>(feet);
  const [anglePins, setAnglePins] = useState<string[]>([]);
  const [mode, setMode] = useState('elastic');
  const [transformMode, setTransformMode] = useState<TransformMode>('translate');
  const [mirror, setMirror] = useState(false);
  const [showMotionAdvanced, setShowMotionAdvanced] = useState(false);
  const [showPaths, setShowPaths] = useState(false);
  const [space, setSpace] = useState<'world' | 'local'>('world');
  const [rotation, setRotation] = useState([0, 0, 0]);
  const [jointDraft, setJointDraft] = useState<number[]>([]);
  const [limits, setLimits] = useState<number[][]>([]);
  const [resistance, setResistance] = useState(1);
  const [target, setTarget] = useState([0, 0, 0]);
  const [info, setInfo] = useState<SolveInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [solving, setSolving] = useState(false);
  const [message, setMessage] = useState('G1 모델을 불러오는 중…');
  const [error, setError] = useState('');
  const [frameIndex, setFrameIndex] = useState(0);
  const [transitionSelection, setTransitionSelection] = useState<number[]>([]);
  const [poseDirty, setPoseDirty] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewStreaming, setPreviewStreaming] = useState(false);
  const previewStream = useRef<{ socket: WebSocket; cancelled: boolean; sourceState: PoseState | null } | null>(null);
  const [sample, setSample] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const playbackSpeedRef = useRef(1);
  playbackSpeedRef.current = playbackSpeed;
  const [physicsEnabled, setPhysicsEnabled] = useState(false);
  const [objectPhysicsEnabled, setObjectPhysicsEnabled] = useState(false);
  const [physicsAvailable, setPhysicsAvailable] = useState(false);
  const [policyEnabled, setPolicyEnabled] = useState(false);
  const [policyAvailable, setPolicyAvailable] = useState(false);
  const [policyJob, setPolicyJob] = useState<PolicyJob | null>(null);
  const [streamTime, setStreamTime] = useState<number | null>(null);
  const policyRequest = useRef<{ id?: string; cancelled: boolean; socket?: WebSocket } | null>(null);
  const policySource = useRef<{ state: PoseState; pins: string[]; anglePins: string[]; frameIndex: number; dirty: boolean } | null>(null);
  const [fps, setFps] = useState(30);
  const [exportProto, setExportProto] = useState(false);
  const [showHandles, setShowHandles] = useState(true);
  const [bakeCount, setBakeCount] = useState(2);
  const [showCollisions, setShowCollisions] = useState(false);
  const [objects, setObjects] = useState<SceneObject[]>(() => objectsFromProject({}));
  const [objectGroups, setObjectGroups] = useState<SceneObjectGroup[]>([]);
  const [selectedObjectGroupId, setSelectedObjectGroupId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [graspPreviewId, setGraspPreviewId] = useState<string | null>(null);
  const [labPathPreview, setLabPathPreview] = useState<LabPathPreview | null>(null);
  const [labStorageRevision, setLabStorageRevision] = useState(0);
  const [pendingLabReplacement, setPendingLabReplacement] = useState<{ project: Project; start: number; end: number;
    frames: Keyframe[]; firstState: PoseState; editedNames: string[]; targetId: string } | null>(null);
  const [collapsedLabGroups, setCollapsedLabGroups] = useState<string[]>([]);
  useEffect(() => {
    const changed = (event: StorageEvent) => {
      if (['motioncreator-cylinder-lab-v2', 'motioncreator-cylinder-lab-tool-settings-v1',
        'motioncreator-cylinder-lab-tool-grasps-v1'].includes(event.key ?? '')) setLabStorageRevision(value => value + 1);
    };
    window.addEventListener('storage', changed);
    return () => window.removeEventListener('storage', changed);
  }, []);
  const [graspCoefficients, setGraspCoefficients] = useState<RamenGraspCoefficients>(DEFAULT_RAMEN_GRASP_COEFFICIENTS);
  const [graspCoefficientsLoaded, setGraspCoefficientsLoaded] = useState(false);
  useEffect(() => {
    try {
      const saved = savedRamenGraspCoefficients(JSON.parse(localStorage.getItem(GRASP_COEFFICIENTS_STORAGE_KEY) ?? 'null'));
      if (saved) setGraspCoefficients(saved);
    } catch { /* 손상된 에디터 설정은 기본 계수로 시작합니다. */ }
    setGraspCoefficientsLoaded(true);
    // The cylinder lab edits the same shared coefficients in another tab.
    const follow = (event: StorageEvent) => {
      if (event.key !== GRASP_COEFFICIENTS_STORAGE_KEY) return;
      try { const next = savedRamenGraspCoefficients(JSON.parse(event.newValue ?? 'null')); if (next) setGraspCoefficients(next); }
      catch { /* 다른 탭의 손상된 값은 무시합니다. */ }
    };
    window.addEventListener('storage', follow);
    return () => window.removeEventListener('storage', follow);
  }, []);
  useEffect(() => {
    if (!graspCoefficientsLoaded) return;
    try { localStorage.setItem(GRASP_COEFFICIENTS_STORAGE_KEY, JSON.stringify(graspCoefficients)); }
    catch { /* 저장 공간이 없어도 파지점 편집은 계속합니다. */ }
  }, [graspCoefficients, graspCoefficientsLoaded]);
  const [objectTransformMode, setObjectTransformMode] = useState<ObjectTransformMode>('translate');
  const [preventObjectOverlap, setPreventObjectOverlap] = useState(true);
  const [objectSurfaceSnap, setObjectSurfaceSnap] = useState(false);
  const [objectGroundLock, setObjectGroundLock] = useState(true);
  const [graspPickMode, setGraspPickMode] = useState<GraspPickMode>(null);
  const graspPickModeRef = useRef<GraspPickMode>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [saved, setSaved] = useState<string[]>([]);
  const [savedChoice, setSavedChoice] = useState('');
  const [saveMenuOpen, setSaveMenuOpen] = useState(false);
  const [robotOpen, setRobotOpen] = useState(false);
  /** 로봇 패널에서 점검했을 때의 프로젝트·장면·툴. 이 중 하나라도 바뀌면(물체 재인식, 카드 수정 등) 다시 점검할 때까지 시작을 막는다. */
  const [checkedFor, setCheckedFor] = useState<{ project: unknown; objects: unknown; groups: unknown; tools: unknown } | null>(null);
  const [historyCount, setHistoryCount] = useState(0);
  const [futureCount, setFutureCount] = useState(0);
  const current = useRef({ state, project, pins, anglePins, mode, resistance, selected, members, transformMode, mirror, busy, playing, solving, objects, objectGroups, selectedObjectId, objectTransformMode, preventObjectOverlap, objectSurfaceSnap, objectGroundLock, poseDirty, frameIndex });
  current.current = { state, project, pins, anglePins, mode, resistance, selected, members, transformMode, mirror, busy, playing, solving, objects, objectGroups, selectedObjectId, objectTransformMode, preventObjectOverlap, objectSurfaceSnap, objectGroundLock, poseDirty, frameIndex };
  const history = useRef<EditorSnapshot[]>([]);
  const future = useRef<EditorSnapshot[]>([]);
  const objectDragCheckpointed = useRef(false);
  const sceneObjectSelection = useRef<string | null>(null);
  const graspFollowTimer = useRef<number | null>(null);
  const graspFollowPending = useRef<{ object: SceneObject; frameIndex: number } | null>(null);
  const graspFollowInFlight = useRef(false);
  const anchor = useRef<number[]>([]);
  const anchorPositions = useRef<Record<string, number[]>>({});
  const anchorCenter = useRef<number[]>([0, 0, 0]);
  const anchorOrientations = useRef<Record<string, number[]>>({});
  const anchorQuaternion = useRef([0, 0, 0, 1]);
  const anchorMirror = useRef<{ active: string; rootQuaternion: number[] } | undefined>(undefined);
  const anchorHinges = useRef<PoseState['hinges']>({});
  const pending = useRef<{ kind: TransformMode | 'joint'; key: string; target: number[]; joints?: Record<string, number> } | null>(null);
  const inFlight = useRef(false);
  const dragActive = useRef(false);
  const alive = useRef(true);
  const actions = useRef({ select: (_key: string, _additive: boolean, _hover: boolean) => {}, begin: () => {}, move: (_key: string, _target: number[]) => {}, rotate: (_key: string, _quaternion: number[]) => {}, jointAngle: (_key: string, _angle: number) => {}, transformMode: (_mode: TransformMode) => {}, history: (_redo: boolean) => {}, pickSurface: (_id: string, _point: number[], _normal: number[]) => {}, end: () => {} });

  /** 지금 선택 부위의 방향(로컬 축). 위치 입력과 1cm 이동이 로컬 축을 쓸 때의 기준이다. */
  const localFrame = state && members.length ? state.handles[controlKey(members, selected)]?.quaternion : undefined;
  /** 선택 위치를 축 하나(로컬 축 모드면 선택 부위의 축)로 `step`(m)만큼 옮긴 목표. */
  function nudgedPosition(axis: number, step: number) {
    const position = selectionPosition(state!, members, selected);
    const offset = [0, 0, 0].map((_, j) => j === axis ? step : 0);
    const move = space === 'local' && localFrame ? fromLocalAxes(offset, localFrame) : offset;
    return position.map((n, j) => n + move[j]);
  }
  function selectionPosition(pose: PoseState, items: string[], active: string) {
    const c = current.current;
    return c.mirror && c.transformMode === 'translate' && canMirrorSelection(controlSelection(items))
      ? [...pose.handles[controlKey(items, active)].position] : selectionCenter(pose, items);
  }

  function applyState(next: PoseState) {
    current.current.state = next;
    setState(next);
    scene.current?.update(next);
    setJointDraft(next.qpos.slice(7).map(v => v * 180 / Math.PI));
    if (!dragActive.current) {
      setTarget(selectionPosition(next, current.current.members, current.current.selected));
      setRotation(eulerDegrees(next.handles[controlKey(current.current.members, current.current.selected)].quaternion));
    }
  }
  function editorSnapshot(): EditorSnapshot | null {
    const value = current.current;
    return value.state && value.project ? { qpos: [...value.state.qpos], pins: [...value.pins], anglePins: [...value.anglePins], objects: structuredClone(value.objects), objectGroups: structuredClone(value.objectGroups), keyframes: structuredClone(value.project.keyframes), frameIndex: value.frameIndex, poseDirty: value.poseDirty } : null;
  }
  function checkpoint() {
    const snapshot = editorSnapshot();
    if (!snapshot) return;
    history.current.push(snapshot);
    if (history.current.length > 60) history.current.shift();
    future.current = [];
    setHistoryCount(history.current.length); setFutureCount(0);
  }
  function invalidate() { setPreview(null); setSample(0); setPlaying(false); }
  function commitObjects(next: SceneObject[], ground = true) {
    setLabPathPreview(null);
    const grounded = ground ? next.map(object => object.parent_id ? object : groundedSceneObject(object)) : next;
    current.current.objects = grounded;
    setObjects(grounded);
    const activeProject = current.current.project;
    if (activeProject) {
      const nextProject = { ...activeProject, scene_objects: grounded };
      current.current.project = nextProject;
      setProject(nextProject);
    }
    invalidate();
    return grounded;
  }
  function commitObjectGroups(next: SceneObjectGroup[]) {
    current.current.objectGroups = next;
    setObjectGroups(next);
    const activeProject = current.current.project;
    if (activeProject) {
      const nextProject = { ...activeProject, scene_groups: next };
      current.current.project = nextProject;
      setProject(nextProject);
    }
    invalidate();
  }
  function scheduleGraspFollow(object: SceneObject) {
    const c = current.current;
    const grasp = c.project?.keyframes[c.frameIndex]?.grasp;
    if (!grasp?.object_signature || grasp.object_id !== object.id || grasp.follow_object === false) return;
    graspFollowPending.current = { object: structuredClone(object), frameIndex: c.frameIndex };
    if (graspFollowTimer.current !== null) window.clearTimeout(graspFollowTimer.current);
    graspFollowTimer.current = window.setTimeout(() => { graspFollowTimer.current = null; void drainGraspFollow(); }, 70);
  }
  async function drainGraspFollow() {
    if (graspFollowInFlight.current) return;
    const pendingFollow = graspFollowPending.current;
    if (!pendingFollow) return;
    graspFollowPending.current = null;
    const c = current.current;
    const frame = c.project?.keyframes[pendingFollow.frameIndex];
    if (!frame?.grasp || c.frameIndex !== pendingFollow.frameIndex || frame.grasp.object_id !== pendingFollow.object.id) return;
    graspFollowInFlight.current = true; setSolving(true); setError('');
    try {
      const result = await api<{ state: PoseState; grasp: TwoHandGrasp; solver: { contact: SolveInfo } }>('grasp-fit', {
        qpos: c.state?.qpos ?? frame.qpos, pins: c.pins, angle_pins: c.anglePins,
        object: pendingFollow.object, grasp: { ...frame.grasp, follow_object: true },
      });
      if (!alive.current || current.current.frameIndex !== pendingFollow.frameIndex || graspFollowPending.current) return;
      applyState(result.state); setInfo(result.solver.contact); setPoseDirty(false);
      const activeProject = current.current.project;
      if (activeProject) {
        const nextProject = { ...activeProject, keyframes: activeProject.keyframes.map((item, index) => index === pendingFollow.frameIndex
          ? { ...item, qpos: [...result.state.qpos], grasp: { ...result.grasp, follow_object: true }, interaction: undefined } : item) };
        current.current.project = nextProject; setProject(nextProject);
      }
      invalidate();
      setMessage(`키프레임 고스트 박스를 따라 양손 자세를 갱신했습니다 · 오차 ${result.solver.contact.target_error_mm.toFixed(2)} mm`);
    } catch (failure) {
      setError(`박스 파지점을 따라갈 수 없습니다: ${(failure as Error).message}`);
    } finally {
      graspFollowInFlight.current = false; setSolving(false);
      if (graspFollowPending.current) {
        graspFollowTimer.current = window.setTimeout(() => { graspFollowTimer.current = null; void drainGraspFollow(); }, 0);
      }
    }
  }
  function changeObject(id: string, patch: Partial<SceneObject>, recordHistory = true) {
    if (recordHistory) checkpoint();
    const before = current.current.objects.find(object => object.id === id);
    if (!before) return;
    const normalized = { ...before, ...patch, size: normalizedObjectSize(patch.shape ?? before.shape, patch.size ?? before.size) };
    const transformChanged = Object.keys(patch).some(key => ['position', 'quaternion_xyzw', 'size', 'shape'].includes(key));
    const descendants = sceneObjectDescendantIds(current.current.objects, id);
    const next = transformChanged
      ? placeSceneObject(normalized, current.current.objects.filter(other => other.id !== id && !descendants.has(other.id)), scenePlacementOptions(before), before)
      : normalized;
    const updated = current.current.objects.map(object => object.id === id ? next : object);
    commitObjects(transformChanged ? transformSceneObjectChildren(updated, before, next) : updated, false);
  }
  function changeObjectParent(ids: string[], parentId: string | null) {
    const before = current.current.objects;
    const after = reparentSceneObjects(before, ids, parentId);
    if (after === before || before.every((object, index) => object.parent_id === after[index].parent_id)) return;
    checkpoint(); commitObjects(after, false);
    setMessage(parentId ? '부모-자식 관계를 만들었습니다. 부모의 이동·회전에 자식이 함께 따라갑니다.' : '부모 연결을 해제했습니다. 현재 월드 위치는 유지됩니다.');
  }
  function changeGraspGhost(patch: Partial<SceneObject>) {
    const c = current.current;
    const frame = c.project?.keyframes[c.frameIndex];
    const grasp = frame?.grasp;
    const target = objectAtGraspPose(frame, c.objects);
    if (!c.project || !frame || !grasp?.object_signature || !target) return;
    const nextTarget = { ...target,
      position: patch.position ? [...patch.position] : target.position,
      quaternion_xyzw: patch.quaternion_xyzw ? [...patch.quaternion_xyzw] : target.quaternion_xyzw };
    const nextGrasp = { ...grasp, object_pose: { position: [...nextTarget.position],
      quaternion_xyzw: [...nextTarget.quaternion_xyzw] } };
    const nextProject = { ...c.project, keyframes: c.project.keyframes.map((item, index) => index === c.frameIndex
      ? { ...item, grasp: nextGrasp } : item) };
    current.current.project = nextProject; setProject(nextProject); invalidate();
    scheduleGraspFollow(nextTarget);
  }
  function editGraspGhost() {
    const c = current.current;
    const frame = c.project?.keyframes[c.frameIndex];
    const ghost = graspGhost(frame, c.objects);
    if (!ghost || !frame?.grasp) return;
    setSelectedObjectId(null); current.current.selectedObjectId = null;
    sceneObjectSelection.current = GRASP_GHOST_ID;
    setObjectTransformMode('translate'); current.current.objectTransformMode = 'translate';
    scene.current?.selectSceneObject(GRASP_GHOST_ID, 'translate');
    setMessage('반투명 고스트 박스를 이동하거나 E로 회전하세요. 현재 키프레임의 양손만 따라갑니다.');
  }
  function selectObject(id: string | null) {
    if (id === GRASP_GHOST_ID) {
      const grasp = current.current.project?.keyframes[current.current.frameIndex]?.grasp;
      if (!grasp) return;
      setSelectedObjectId(null); current.current.selectedObjectId = null;
      sceneObjectSelection.current = id;
      scene.current?.selectSceneObject(id, current.current.objectTransformMode);
      setMessage('키프레임 고스트 박스를 선택했습니다. 실제 장면 박스 위치는 바뀌지 않습니다.');
      return;
    }
    setSelectedObjectId(id); current.current.selectedObjectId = id;
    sceneObjectSelection.current = id;
    const selectedObject = current.current.objects.find(object => object.id === id);
    if (selectedObject) {
      const placement = scenePlacementOptions(selectedObject);
      current.current.preventObjectOverlap = placement.preventOverlap; setPreventObjectOverlap(placement.preventOverlap);
      current.current.objectSurfaceSnap = placement.surfaceSnap; setObjectSurfaceSnap(placement.surfaceSnap);
      current.current.objectGroundLock = placement.groundLock; setObjectGroundLock(placement.groundLock);
    }
    scene.current?.selectSceneObject(id, current.current.objectTransformMode);
  }
  function changeObjectMode(mode: ObjectTransformMode) {
    setObjectTransformMode(mode); current.current.objectTransformMode = mode;
    scene.current?.selectSceneObject(sceneObjectSelection.current, mode);
  }
  function addObject(shape: SceneObjectShape) {
    checkpoint();
    const created = createSceneObject(current.current.objects.length + 1, shape);
    const object = placeSceneObject(created, current.current.objects, scenePlacementOptions(created));
    commitObjects([...current.current.objects, object]);
    selectObject(object.id);
  }
  function addDetectedRows(rows: WorldRow[]) {
    if (!rows.length) { setError('인식된 줄이 없습니다. 용기 인식을 켜고 줄이 보일 때 다시 눌러 주세요.'); return; }
    const existing = current.current.objects;
    if (existing.length + rows.length > 32) { setError(`장면 오브젝트는 최대 32개입니다. ${rows.length}개를 추가할 수 없습니다.`); return; }
    checkpoint();
    const created = rows.map((row, index) => ({
      ...createSceneObject(existing.length + index + 1, 'cylinder'),
      name: `인식 줄 ${existing.length + index + 1}${row.clipped ? ' (잘림)' : ''}`,
      position: row.position, quaternion_xyzw: row.quaternion_xyzw, size: row.size,
      mass_kg: ROW_MASS_KG, wall_thickness_m: ROW_WALL_M,
      collision_shape: 'cylinder' as const, collision_size: [ROW_LINE_M, ROW_LINE_M, row.size[2]],
      placement: { prevent_overlap: false, surface_snap: false, ground_lock: false },
    }));
    commitObjects([...existing, ...created], false);
    selectObject(created[0].id);
    const clipped = rows.filter(row => row.clipped).length;
    setMessage(`인식된 줄 ${rows.length}개를 원통으로 추가했습니다(지름 ${(rows[0].size[0] * 100).toFixed(1)}cm, 800g, 충돌은 축 선).${clipped ? ` ${clipped}개는 화각 밖으로 이어져 실제 길이보다 짧을 수 있습니다.` : ''}`);
  }
  function removeObject(id: string) {
    const { objects: remaining, removedIds } = removeSceneObjectSubtree(current.current.objects, id);
    if (!removedIds.size) return;
    checkpoint();
    commitObjects(remaining, false);
    commitObjectGroups(current.current.objectGroups.map(group => ({ ...group, member_ids: group.member_ids.filter(member => !removedIds.has(member)) }))
      .filter(group => group.member_ids.length >= 2));
    const activeProject = current.current.project;
    if (activeProject) {
      const nextProject = { ...activeProject, keyframes: activeProject.keyframes.map(frame => ({ ...frame,
        ...(frame.grasp && removedIds.has(frame.grasp.object_id) ? { grasp: undefined } : {}),
        ...(frame.interaction && removedIds.has(frame.interaction.object_id) ? { interaction: undefined } : {}),
      })) };
      current.current.project = nextProject; setProject(nextProject);
    }
    selectObject(null);
    setMessage(removedIds.size > 1 ? `부모와 하위 오브젝트 ${removedIds.size}개를 삭제했습니다. 실행 취소로 복원할 수 있습니다.` : '오브젝트를 삭제했습니다. 실행 취소로 복원할 수 있습니다.');
  }
  async function importSceneAsset(source: File) {
    const extension = source.name.slice(source.name.lastIndexOf('.')).toLowerCase();
    if (!['.blend', '.glb'].includes(extension)) throw new Error('.blend 또는 .glb 파일만 가져올 수 있습니다.');
    const response = await fetch(`/api/scene-assets/import?filename=${encodeURIComponent(source.name)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: source,
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({})) as { detail?: unknown };
      throw new Error(typeof failure.detail === 'string' ? failure.detail : `3D 모델 가져오기 실패 (${response.status})`);
    }
    const imported = await response.json() as SceneAssetImport;
    const created = createImportedSceneObjects(imported, current.current.objects.length + 1);
    if (current.current.objects.length + created.length > 32) throw new Error(`이 파일은 ${created.length}개 오브젝트를 포함합니다. 장면의 최대 32개 제한을 초과합니다.`);
    checkpoint();
    if (created.length === 1) {
      const placed = placeSceneObject(created[0], current.current.objects, scenePlacementOptions(created[0]));
      commitObjects([...current.current.objects, placed]); selectObject(placed.id);
    } else {
      commitObjects([...current.current.objects, ...created], false);
      const group = { ...createSceneObjectGroup(created, created.map(object => object.id), current.current.objectGroups.length + 1), name: imported.name };
      commitObjectGroups([...current.current.objectGroups, group]);
      setSelectedObjectGroupId(null); selectObject(created[0].id);
    }
    setMessage(`${source.name}에서 ${created.length}개 장면 오브젝트를 가져왔습니다. ${extension === '.blend' ? '편집용 BLEND를 런타임 GLB로 변환했습니다.' : 'GLB 좌표축을 URDF 기준으로 변환했습니다.'}`);
  }
  function createObjectGroup(memberIds: string[], name: string) {
    const available = memberIds.filter(id => current.current.objects.some(object => object.id === id)
      && !current.current.objectGroups.some(group => group.member_ids.includes(id)));
    if (available.length < 2) { setError('그룹에는 아직 다른 그룹에 속하지 않은 물체가 2개 이상 필요합니다.'); return; }
    checkpoint();
    const group = { ...createSceneObjectGroup(current.current.objects, available, current.current.objectGroups.length + 1), ...(name.trim() ? { name: name.trim() } : {}) };
    commitObjects(current.current.objects.map(object => available.includes(object.id)
      ? withScenePlacement(object, { preventOverlap: false, surfaceSnap: false, groundLock: false }) : object), false);
    commitObjectGroups([...current.current.objectGroups, group]);
    setSelectedObjectGroupId(group.id); selectObject(null); setError('');
  }
  function changeObjectGroup(id: string, patch: Partial<SceneObjectGroup>) {
    const previous = current.current.objectGroups.find(group => group.id === id);
    if (!previous) return;
    checkpoint();
    const next = { ...previous, ...patch };
    const transformChanged = !!patch.position || !!patch.quaternion_xyzw;
    if (transformChanged) commitObjects(transformSceneObjectGroup(current.current.objects, previous, next), false);
    commitObjectGroups(current.current.objectGroups.map(group => group.id === id ? next : group));
  }
  function removeObjectGroup(id: string) {
    checkpoint(); commitObjectGroups(current.current.objectGroups.filter(group => group.id !== id));
    setSelectedObjectGroupId(null);
  }
  function changeObjectPlacement(patch: Partial<ScenePlacementOptions>) {
    const id = current.current.selectedObjectId;
    const selectedObject = current.current.objects.find(object => object.id === id);
    if (!selectedObject) return;
    checkpoint();
    const placement = { ...scenePlacementOptions(selectedObject), ...patch };
    current.current.preventObjectOverlap = placement.preventOverlap; setPreventObjectOverlap(placement.preventOverlap);
    current.current.objectSurfaceSnap = placement.surfaceSnap; setObjectSurfaceSnap(placement.surfaceSnap);
    current.current.objectGroundLock = placement.groundLock; setObjectGroundLock(placement.groundLock);
    let updated = withScenePlacement(selectedObject, placement);
    if (placement.groundLock) {
      updated = { ...updated, position: [updated.position[0], updated.position[1], objectVerticalHalfExtent(updated)] };
    }
    commitObjects(current.current.objects.map(object => object.id === id ? updated : object));
  }
  function changeGraspPickMode(mode: GraspPickMode) {
    graspPickModeRef.current = mode; setGraspPickMode(mode); scene.current?.setSurfacePickMode(!!mode);
  }
  function pickGripSurface(id: string, localPoint: number[], localNormal: number[]) {
    const hand = graspPickModeRef.current;
    const frame = project?.keyframes[frameIndex];
    const grasp = frame?.grasp;
    if (!hand || !grasp) return;
    if (id !== grasp.object_id) { setError('파지 설정에서 선택한 박스의 표면을 클릭하세요.'); return; }
    const expected = hand === 'left' ? 1 : -1;
    if (localNormal[1] * expected < .5) { setError(`${hand === 'left' ? '왼손은 박스 +Y 면' : '오른손은 박스 −Y 면'}을 클릭하세요.`); return; }
    const snapped = [localPoint[0] * 2, localPoint[2] * 2].map(value => {
      const clamped = Math.max(-.9, Math.min(.9, value));
      return Math.abs(clamped) < .12 ? 0 : Math.round(clamped * 10) / 10;
    });
    const patch = hand === 'left' ? { left_surface_uv: snapped } : { right_surface_uv: snapped };
    checkpoint();
    editFrame(frameIndex, { grasp: { ...grasp, ...patch, closure_qpos: undefined, object_signature: undefined,
      contact_points_world: undefined, hand_twist_deg: undefined } });
    setError(''); changeGraspPickMode(null);
    setMessage(`${hand === 'left' ? '왼손' : '오른손'} 접촉점을 박스 표면에 배치했습니다.`);
  }
  async function run(work: () => Promise<void>) {
    if (current.current.busy || inFlight.current) return;
    setBusy(true); current.current.busy = true; setError(''); setPlaying(false);
    try { await work(); } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); current.current.busy = false; }
  }
  function applySelection(next: string[], active: string, hover = false) {
    next = expandVirtualControls(next);
    if (!next.includes(active)) active = expandVirtualControls([active])[0];
    current.current.selected = active;
    current.current.members = next;
    setSelected(active); setMembers(next); setInfo(null);
    setSelectedObjectId(null); current.current.selectedObjectId = null; scene.current?.selectSceneObject(null);
    sceneObjectSelection.current = null;
    if (scene.current && !hover) scene.current.selectionLocked = true;
    const previousMode = current.current.transformMode;
    const controls = controlSelection(next);
    const nextMode = controls.length === 1 && isJointHandle(controls[0]) && !hover ? 'rotate' : previousMode === 'rotate' && !canRotateSelection(controls) ? 'translate' : previousMode;
    current.current.transformMode = nextMode; setTransformMode(nextMode);
    scene.current?.select(active, current.current.pins, next, current.current.anglePins);
    scene.current?.setTransformMode(nextMode, space);
    const pose = current.current.state;
    if (pose) {
      setTarget(selectionPosition(pose, next, active));
      setRotation(eulerDegrees(pose.handles[controlKey(next, active)].quaternion));
    }
  }
  function select(key: string, additive = false, hover = false) {
    selectBatch([key], additive, hover);
  }
  function selectBatch(incoming: string[], additive = false, hover = false) {
    if (current.current.busy || current.current.playing || inFlight.current || dragActive.current) return;
    const next = selectMembers(current.current.members, incoming, additive);
    if (!next.length) return;
    applySelection(next, next.includes(incoming[0]) ? incoming[0] : next[next.length - 1], hover);
  }
  function toggleExpanded(id: string) {
    setExpanded(previous => {
      const next = previous.includes(id) ? previous.filter(key => key !== id) : [...previous, id];
      try { localStorage.setItem('g1-body-tree-expanded', JSON.stringify(next)); } catch { /* Expansion still works without storage. */ }
      return next;
    });
  }
  function begin() {
    if (!current.current.state) return;
    checkpoint(); anchor.current = [...current.current.state.qpos];
    const controls = controlSelection(current.current.members);
    anchorPositions.current = Object.fromEntries(controls.map(k => [k, [...current.current.state!.handles[k].position]]));
    anchorCenter.current = selectionPosition(current.current.state, current.current.members, current.current.selected);
    anchorMirror.current = current.current.mirror && current.current.transformMode === 'translate' && canMirrorSelection(controls)
      ? { active: controlKey(current.current.members, current.current.selected), rootQuaternion: [...current.current.state.handles[mirrorFrameHandle(controls)].quaternion] } : undefined;
    anchorOrientations.current = Object.fromEntries(controls.map(k => [k, [...current.current.state!.handles[k].quaternion]]));
    anchorQuaternion.current = [...current.current.state.handles[controlKey(current.current.members, current.current.selected)].quaternion];
    anchorHinges.current = structuredClone(current.current.state.hinges);
    dragActive.current = true;
    invalidate(); setError('');
  }
  async function drain() {
    if (inFlight.current || !pending.current || !current.current.state) return;
    inFlight.current = true; setSolving(true);
    try {
      while (pending.current && alive.current) {
        const next = pending.current; pending.current = null;
        const c = current.current;
        let goals: Record<string, unknown>;
        if (next.kind === 'rotate') goals = rotatedGroupTargets(anchorPositions.current, anchorOrientations.current, anchorCenter.current, anchorQuaternion.current, next.target, c.pins);
        else if (next.kind === 'joint') goals = { joints: next.joints };
        else {
          const delta = next.target.map((value, i) => value - anchorCenter.current[i]);
          goals = { targets: translatedTargets(anchorPositions.current, delta, anchorMirror.current) };
        }
        const result = await api<{ state: PoseState; solver: SolveInfo }>('solve-group', {
          qpos: c.state!.qpos, anchor: anchor.current, ...goals,
          pins: c.pins, angle_pins: c.anglePins, mode: c.mode, resistance: c.resistance,
        });
        if (!alive.current) return;
        applyState(result.state); setInfo(result.solver); setPoseDirty(true);
        setMessage(result.solver.rejected ? '고정 조건을 유지할 수 없어 이전 자세를 유지했습니다.' : result.solver.converged ? '편집 목표에 도달했습니다. 자세를 키프레임에 반영하세요.' : '관절 범위 또는 고정 조건 때문에 목표에 완전히 도달하지 못했습니다.');
      }
    } catch (e) { pending.current = null; setError((e as Error).message); }
    finally { inFlight.current = false; setSolving(false); }
  }
  function move(key: string, value: number[]) { pending.current = { kind: 'translate', key, target: value }; setTarget(value); void drain(); }
  function rotate(key: string, quaternion: number[]) { pending.current = { kind: 'rotate', key, target: quaternion }; setRotation(eulerDegrees(quaternion)); void drain(); }
  function jointAngle(key: string, angle: number) {
    const hinge = current.current.state?.hinges?.[key];
    if (!hinge || !Number.isFinite(angle)) return;
    const value = Math.max(hinge.limits[0], Math.min(hinge.limits[1], angle));
    const joints = { [hinge.joint_name]: value } as Record<string, number>;
    const controls = controlSelection(current.current.members);
    if (isJointPairSelection(controls) && controls.includes(key)) {
      const otherKey = counterpart(key)!;
      const other = current.current.state?.hinges?.[otherKey];
      const base = anchorHinges.current[key];
      const otherBase = anchorHinges.current[otherKey];
      if (other && base && otherBase) Object.assign(joints, pairedJointTargets(base, otherBase, value, current.current.mirror, current.current.state!.handles[mirrorFrameHandle([key, otherKey])].quaternion));
    }
    pending.current = { kind: 'joint', key, target: [], joints };
    void drain();
  }
  function applyHingeAngle(angle: number) {
    if (!Number.isFinite(angle) || busy || solving || playing) return;
    begin(); dragActive.current = false; jointAngle(selected, angle);
  }
  function applyPartAngles(names: string[]) {
    if (!state || !project || busy || solving || playing) return;
    const joints = Object.fromEntries(names.map(name => [name, jointDraft[project.joint_names.indexOf(name)] * Math.PI / 180]));
    if (Object.values(joints).some(value => !Number.isFinite(value))) return;
    begin(); dragActive.current = false;
    pending.current = { kind: 'joint', key: selected, target: [], joints };
    void drain();
  }
  function changeTransformMode(next: TransformMode) {
    const c = current.current;
    if (!c.state || c.busy || c.playing || inFlight.current || dragActive.current) return;
    if (next === 'rotate' && !canRotateSelection(controlSelection(c.members))) {
      setMessage('관절축 회전은 한 관절씩 선택하세요. 여러 관절을 함께 선택한 경우 W로 IK 이동할 수 있습니다.');
      return;
    }
    current.current.transformMode = next;
    setTransformMode(next);
    scene.current?.setTransformMode(next, space);
  }
  actions.current = { select: (key, additive, hover) => {
    const group = groupForControl(key);
    if (group && (!expanded.includes(group.id) || !nodeMembers(group).includes(key))) selectBatch(nodeMembers(group), additive, hover);
    else select(key, additive, hover);
  }, begin, move, rotate, jointAngle, transformMode: changeTransformMode, history: changeHistory,
  pickSurface: pickGripSurface, end: () => {
    dragActive.current = false;
    if (current.current.state) applyState(current.current.state);
  } };

  useEffect(() => {
    alive.current = true;
    let viewer: RobotScene | null = null;
    const createViewer = (modelId: string, visualRevision: string) => {
      try {
        viewer = new RobotScene(host.current!, {
          select: (key, additive, hover) => actions.current.select(key, additive, hover), begin: () => actions.current.begin(),
          move: (key, value) => actions.current.move(key, value), rotate: (key, value) => actions.current.rotate(key, value), end: () => actions.current.end(), error: setError,
          transformMode: mode => actions.current.transformMode(mode),
          jointAngle: (key, angle) => actions.current.jointAngle(key, angle),
          history: redo => actions.current.history(redo),
          selectObject,
          addDetectedRows,
          deleteObject: removeObject,
          objectTransformBegin: () => { objectDragCheckpointed.current = false; },
          transformObject: (id, patch) => {
            if (!objectDragCheckpointed.current) { checkpoint(); objectDragCheckpointed.current = true; }
            if (id === GRASP_GHOST_ID) changeGraspGhost(patch);
            else changeObject(id, patch, false);
          },
          objectTransformEnd: () => { objectDragCheckpointed.current = false; },
          objectTransformMode: changeObjectMode,
          pickObjectSurface: (id, point, normal) => actions.current.pickSurface(id, point, normal),
        }, modelId, visualRevision);
        scene.current = viewer;
      } catch { setError('WebGL을 시작하지 못했습니다. 브라우저의 하드웨어 가속 설정을 확인하세요.'); }
    };
    api<{ model_id: string; visual_revision: string; state: PoseState; arm_geoms?: string[]; project: Project; limits: number[][]; tools?: [string, string] | null; tool_options?: { id: string; label: string }[] }>('init').then(async init => {
      if (!alive.current) return;
      createViewer(init.model_id, init.visual_revision);
      scene.current?.setReferenceOnly(init.arm_geoms ?? null);
      setModelId(init.model_id);
      setTools(init.tools ?? null); setToolOptions(init.tool_options ?? []);
      draftKey.current = init.model_id === 'g1' ? 'g1-motion-draft-v1' : `g1-motion-draft-v1-${init.model_id}`;
      setLimits(init.limits.map(range => range.map(v => v * 180 / Math.PI)));
      let nextProject = init.project;
      let initialState = init.state;
      let initialPins = feet;
      let initialAnglePins: string[] = [];
      try {
        const draft = localStorage.getItem(draftKey.current);
        if (draft) {
          nextProject = await api<Project>('validate', { project: JSON.parse(draft) });
          initialState = await api<PoseState>('pose', { qpos: nextProject.current_qpos ?? nextProject.keyframes[0].qpos });
          initialPins = nextProject.pins ?? feet;
          initialAnglePins = nextProject.angle_pins ?? nextProject.keyframes[0].angle_pins ?? [];
        }
      } catch { setMessage('이전 자동 저장을 복원하지 못해 기본 자세로 시작했습니다.'); nextProject = init.project; }
      if (!alive.current) return;
      const nextObjects = objectsFromProject(nextProject);
      nextProject = withLegacyGraspGhosts({ ...nextProject, scene_objects: nextObjects }, nextObjects);
      setObjects(nextObjects); current.current.objects = nextObjects;
      const nextObjectGroups = nextProject.scene_groups ?? [];
      setObjectGroups(nextObjectGroups); current.current.objectGroups = nextObjectGroups;
      setProject(nextProject); current.current.project = nextProject;
      setPins(initialPins); current.current.pins = initialPins;
      setAnglePins(initialAnglePins); current.current.anglePins = initialAnglePins;
      applyState(initialState); setTarget(initialState.handles.pelvis.position);
      setMessage(init.model_id === 'g1-tools' ? `G1 그리퍼 모델: ${(init.tool_options ?? []).find(option => option.id === init.tools?.[0])?.label ?? '왼손 툴'} (왼손) · ${(init.tool_options ?? []).find(option => option.id === init.tools?.[1])?.label ?? '오른손 툴'} (오른손). 양쪽 조작점은 그리퍼 TCP이며 Physics 재생을 지원합니다. 위쪽 선택창에서 손마다 툴을 바꿀 수 있습니다.` : '부위에 마우스를 올리고 축을 드래그하세요. 양발은 고정되어 있습니다.');
    }).catch(e => setError(`계산 서버에 연결할 수 없습니다: ${e.message}`));
    api<string[]>('saved').then(setSaved).catch(() => {});
    api<{ available: boolean; physics_available?: boolean }>('policy-preview/runtime').then(result => {
      if (alive.current) { setPolicyAvailable(result.available); setPhysicsAvailable(!!result.physics_available); }
    }).catch(() => {});
    api<GroupPreset[]>('groups').then(setGroups).catch(e => setError(`그룹 프리셋 불러오기 실패: ${e.message}`));
    return () => {
      alive.current = false; viewer?.dispose(); scene.current = null;
      if (graspFollowTimer.current !== null) window.clearTimeout(graspFollowTimer.current);
      const request = policyRequest.current;
      if (request) {
        request.cancelled = true;
        request.socket?.close();
        if (request.id) void api(`policy-preview/${request.id}/cancel`, {}).catch(() => {});
      }
      if (previewStream.current) {
        previewStream.current.cancelled = true;
        previewStream.current.socket.close();
      }
    };
  }, []);
  useEffect(() => { scene.current?.select(selected, pins, members, anglePins); }, [selected, pins, anglePins, members]);
  useEffect(() => {
    scene.current?.setTransformMode(transformMode, space);
    const pose = current.current.state;
    if (pose) setTarget(selectionPosition(pose, members, selected));
  }, [transformMode, space, mirror]);
  useEffect(() => { if (scene.current) { scene.current.setEditable(!busy && !playing && !preview?.physics); scene.current.keyboardEnabled = !preview?.physics; } }, [busy, playing, preview]);
  useEffect(() => { scene.current?.showHandles(showHandles); }, [showHandles]);
  useEffect(() => {
    // 경로 표시: 미리보기 전체 동안 두 손(TCP)이 지나는 길을 2px 선으로 그린다(분홍 왼손, 파랑 오른손).
    if (!showPaths || !preview?.states.length) { scene.current?.setComparePaths(null); return; }
    const path = (side: 'left' | 'right') => preview.states.map(state => state.handles[`${side}_hand`].position);
    scene.current?.setComparePaths({ original: { left: [], right: [] }, resolved: { left: path('left'), right: path('right') } });
  }, [preview, showPaths]);
  useEffect(() => {
    // 경로 표시 중에는 물체 연결 카드가 따라가는 물체(파지할 물체)의 재질을 파란빛으로 강조한다.
    const ids = showPaths ? [...new Set((project?.keyframes ?? []).flatMap(frame =>
      frame.motion?.mode === 'cartesian' && frame.motion.target?.kind === 'object' ? [frame.motion.target.object_id] : []))] : [];
    scene.current?.setHighlightedObjects(ids);
  }, [project, showPaths, objects]);
  useEffect(() => { scene.current?.setCollisionProxiesVisible(showCollisions); }, [showCollisions]);
  useEffect(() => { scene.current?.setVisibleHandles(visibleTreeHandles(expanded)); }, [expanded]);
  useEffect(() => { scene.current?.setMirrorTranslation(mirror); }, [mirror]);
  useEffect(() => {
    const ghost = !playing && !preview?.physics ? graspGhost(project?.keyframes[frameIndex], objects) : undefined;
    scene.current?.setSceneObjects(ghost ? [...objects, ghost] : objects);
    // Rebuilding the meshes resets them to their authored poses; when playback stops keep the replayed ones.
    if (preview?.object_states?.[sample]) scene.current?.setObjectPoses(preview.object_states[sample]);
    if (!ghost && sceneObjectSelection.current === GRASP_GHOST_ID) sceneObjectSelection.current = null;
  }, [objects, project, frameIndex, playing, preview]); // eslint-disable-line react-hooks/exhaustive-deps -- sample is read only when these change
  useEffect(() => { current.current.project = project; }, [project]);
  useEffect(() => {
    const grasp = project?.keyframes[frameIndex]?.grasp;
    scene.current?.setGripMarkers(grasp?.object_pose ? GRASP_GHOST_ID : grasp?.object_id ?? null,
      grasp?.left_surface_uv, grasp?.right_surface_uv);
  }, [project, frameIndex, objects]);
  const graspPreview = useMemo(() => {
    if (!graspPreviewId || graspPreviewId !== selectedObjectId || playing || preview?.physics) return null;
    const object = objects.find(value => value.id === graspPreviewId);
    const pelvis = state?.handles.pelvis;
    return object && pelvis && isRamenBundle(object)
      ? candidateRamenGraspPoints(object, pelvis.position, pelvis.quaternion, graspCoefficients) : null;
  }, [graspPreviewId, selectedObjectId, objects, state, playing, preview?.physics, graspCoefficients]);
  useEffect(() => { scene.current?.setCandidateGraspMarkers(labPathPreview ? null : graspPreview); }, [graspPreview, labPathPreview]);
  useEffect(() => {
    scene.current?.setLabPathPreview(labPathPreview?.targetId === selectedObjectId ? labPathPreview : null);
  }, [labPathPreview, selectedObjectId]);
  // 현재 재생(또는 탐색) 위치가 속한 키프레임·클립과 그 구간의 진행률. 정지·탐색 중에도 위치를 보여 준다.
  // 카드에 표시하는 시작 시각. 서버가 계산한 값이 있으면 그것을, 없으면 같은 규칙으로 로컬에서 계산한다.
  const cardStarts = useMemo(() => project ? keyframeStartTimes(project.keyframes) : [], [project]);
  const stage = useMemo(() => {
    if (!preview || !project || !(playing || sample > 0)) return null;
    const starts = preview.keyframe_times ?? cardStarts;
    const offset = preview.keyframe_times ? 0 : starts[Math.min(preview.summary?.start_frame_index ?? 0, starts.length - 1)] ?? 0;
    const clock = preview.time[Math.min(sample, preview.time.length - 1)] ?? 0;
    const time = clock + offset + 1e-6;
    let index = 0;
    while (index + 1 < starts.length && starts[index + 1] <= time) index++;
    const next = starts[index + 1];
    // The fill is drawn one sample ahead: the last sample before the next card is one frame short of the boundary, which would leave the bar just under 100% when the highlight moves on.
    const frame = preview.time.length > 1 ? preview.time[1] - preview.time[0] : 0;
    const progress = next === undefined ? 1 : Math.max(0, Math.min(1, (time + frame - starts[index]) / Math.max(1e-6, next - starts[index])));
    const total = preview.summary?.reference_seconds ?? preview.time[preview.time.length - 1] ?? 0;
    return { index, progress, overall: total > 0 ? Math.max(0, Math.min(1, clock / total)) : 0 };
  }, [preview, project, cardStarts, playing, sample]);
  const playbackFrame = stage?.index ?? null;
  useEffect(() => {
    if (playbackFrame !== null) document.querySelector(`.frame-track .frame-card[data-frame-index="${playbackFrame}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [playbackFrame]);
  function applyCandidateGraspToFrame() {
    const frame = project?.keyframes[frameIndex];
    if (!frame || !graspPreview || !selectedObjectId || busy || solving || playing || preview?.physics || frame.samples) return;
    const target = structuredClone(graspPreview);
    const selectedFrameIndex = frameIndex;
    void run(async () => {
      const result = await api<{ state: PoseState; solver: SolveInfo }>('solve-group', {
        qpos: frame.qpos, anchor: frame.qpos,
        targets: { left_hand: target.left, right_hand: target.right },
        orientations: { left_hand: target.orientation.left, right_hand: target.orientation.right },
        pins: frame.pins, angle_pins: frame.angle_pins ?? [], mode, resistance, retry_from_presets: true,
      });
      if (!alive.current || current.current.frameIndex !== selectedFrameIndex) return;
      setInfo(result.solver);
      if (!result.solver.converged || result.solver.rejected) {
        throw new Error(`양손 TCP가 파지점에 도달하지 못했습니다 · 위치 ${result.solver.target_error_mm.toFixed(1)} mm / 방향 ${(result.solver.angle_error_deg ?? 0).toFixed(1)}°. 키프레임은 변경하지 않았습니다.`);
      }
      checkpoint();
      applyState(result.state);
      setPins([...frame.pins]); setAnglePins([...(frame.angle_pins ?? [])]);
      editFrame(selectedFrameIndex, { qpos: [...result.state.qpos] });
      setPoseDirty(false);
      setMessage(`선택 키프레임의 양손 TCP를 파지점에 적용했습니다 · 위치 ${result.solver.target_error_mm.toFixed(1)} mm / 방향 ${(result.solver.angle_error_deg ?? 0).toFixed(1)}°`);
    });
  }
  async function prepareLabGrasp() {
    const target = objects.find(object => object.id === selectedObjectId);
    const pelvis = state?.handles.pelvis;
    if (!target || !pelvis || !tools || modelId !== 'g1-tools') return;
    await run(async () => {
      const mounted = await api<{ model_id: string; tools: [string, string] | null }>('tools');
      if (mounted.model_id !== modelId || !mounted.tools || mounted.tools.some((tool, index) => tool !== tools[index]))
        throw new Error('Cylinder Lab에서 장착 툴이 변경되었습니다. 메인 에디터를 새로고침한 뒤 파지법을 적용해 주세요.');
      const paths = readLabToolPaths(target.id, tools, graspCoefficients);
      // Drawings saved before source metadata was added can be tied to their original saved scene.
      const legacy = Object.values(paths).filter(path => path && !path.source);
      for (const path of legacy) {
        if (!path || path.projectName.startsWith('데모:')) continue;
        try {
          const savedProject = await api<Project>(`project/${fileApiPath(path.projectName)}`);
          const response = await api<{ scene: CylinderScene & { plane_y: number } }>('cylinder-lab/scene',
            { project: savedProject, target_id: target.id });
          const settled = restingScene(response.scene);
          path.source = { center: [...settled.target.center], radius: settled.target.radius, planeY: response.scene.plane_y };
        } catch {
          throw new Error(`${path.toolId}의 예전 경로 기준 위치를 찾지 못했습니다. Cylinder Lab에서 그 툴의 경로를 다시 열어 저장해 주세요.`);
        }
      }
      const next = withLabLateralOffsets(buildLabPathPreview(target, pelvis.position, pelvis.quaternion, paths,
        100 - (project?.keyframes.length ?? 0)), labLateralOffsets(project, target.id, tools));
      setLabPathPreview(next);
      setMessage(`${target.name}의 파지법을 3D 장면에 표시했습니다 · 왼손 ${next.tracks.left?.drawnCount ?? 0}점 / 오른손 ${next.tracks.right?.drawnCount ?? 0}점 · 생성 ${next.times.length}키프레임`);
    });
  }
  function changeLabLateral(targetId: string, side: HandSide, baseLateral: number, lateral: number) {
    if (!project || !tools || !Number.isFinite(lateral)) return;
    const key = labLateralKey(targetId, tools);
    const offsets = { ...labLateralOffsets(project, targetId, tools),
      [side]: Number((Math.max(0, lateral) - baseLateral).toFixed(6)) };
    if (!offsets[side]) delete offsets[side];
    const all = { ...project.lab_lateral_offsets };
    if (Object.keys(offsets).length) all[key] = offsets;
    else delete all[key];
    const nextProject = { ...project, lab_lateral_offsets: all };
    current.current.project = nextProject; setProject(nextProject);
    setLabPathPreview(previous => previous?.targetId === targetId
      ? withLabLateralOffsets(previous, offsets) : previous);
    setPendingLabReplacement(null);
  }
  async function generateLabGraspKeyframes() {
    const preview = labPathPreview, source = project?.keyframes[frameIndex];
    if (!preview || !project || !source || !tools || preview.targetId !== selectedObjectId || poseDirty || source.samples) return;
    await run(async () => {
      if (project.keyframes.length + preview.times.length > 100) throw new Error('키프레임 최대 100개를 넘습니다. 다른 키프레임을 줄인 뒤 파지법을 다시 미리보세요.');
      const activeSides = (['left', 'right'] as HandSide[]).filter(side => !!preview.tracks[side]);
      for (const side of activeSides) {
        if (source.pins.includes(`${side}_hand`) || source.angle_pins?.includes(`${side}_hand`))
          throw new Error(`${side === 'left' ? '왼손' : '오른손'} TCP 고정을 해제한 뒤 다시 시도하세요.`);
      }
      const waypoints = preview.times.map((_, index) => {
        const targets: Record<string, number[]> = {};
        const orientations: Record<string, number[]> = {};
        for (const side of activeSides) {
          const pose = toolPoseAtKeyPoint(preview.tracks[side]!, index);
          targets[`${side}_hand`] = pose.position;
          orientations[`${side}_hand`] = pose.quaternion_xyzw;
        }
        return { targets, orientations };
      });
      setMessage(`파지법 계산 중 · 웨이포인트 ${waypoints.length}개`);
      const solved = await api<{ qpos: number[][]; first_state: PoseState }>('cylinder-lab/solve-path', {
        qpos: source.qpos, tools, pins: source.pins, angle_pins: source.angle_pins ?? [], waypoints,
      });
      if (solved.qpos.length !== preview.times.length) throw new Error('계산된 키프레임 개수가 웨이포인트와 다릅니다.');
      const generated: Keyframe[] = solved.qpos.map((qpos, index) => ({
        name: index === 0 ? '그린 점 1 · 파지점' : `그린 점 ${index + 1}`,
        duration: index === 0 ? .8 : Math.max(.1, preview.times[index] - preview.times[index - 1]),
        qpos, pins: [...source.pins], angle_pins: [...(source.angle_pins ?? [])],
      }));
      const target = objects.find(object => object.id === preview.targetId);
      if (!target) throw new Error('파지 대상 용기를 찾지 못했습니다.');
      const groupId = crypto.randomUUID();
      const sourceSignature = labSourceSignature({ sourcePaths: readLabToolPaths(target.id, tools, graspCoefficients) }, target, source,
        labLateralOffsets(project, target.id, tools));
      generated.forEach((frame, index) => { frame.lab_grasp = { group_id: groupId, target_id: target.id,
        tools: [...tools], source_signature: sourceSignature, point_index: index,
        authored_signature: authoredSignature(frame) }; });
      checkpoint();
      const insert = frameIndex + 1;
      const nextProject = { ...project, keyframes: [...project.keyframes.slice(0, insert), ...generated,
        ...project.keyframes.slice(insert)] };
      current.current.project = nextProject; setProject(nextProject); setTransitionSelection([]);
      applyState(solved.first_state);
      setPins([...generated[0].pins]); setAnglePins([...(generated[0].angle_pins ?? [])]);
      setFrameIndex(insert); setPoseDirty(false); setInfo(null); setLabPathPreview(null); invalidate();
      setMessage(`${generated.length}개 파지 키프레임을 선택 프레임 뒤에 생성했습니다.`);
    });
  }
  async function updateLabGraspKeyframes(groupIndex: number) {
    if (!project || !tools || poseDirty) return;
    const range = labGroupRange(project.keyframes, groupIndex);
    if (!range?.valid) { setError('파지법 구간의 순서가 변경되었습니다. 구간을 정리한 뒤 다시 시도해 주세요.'); return; }
    const link = range.frames[0].lab_grasp!;
    const target = objects.find(object => object.id === link.target_id);
    if (!target) { setError('파지법에 연결된 용기를 찾지 못했습니다.'); return; }
    if (tools.some((tool, index) => tool !== link.tools[index])) { setError('연결된 파지법의 장착 툴로 전환한 뒤 업데이트해 주세요.'); return; }
    setPendingLabReplacement(null);
    await run(async () => {
      const source = project.keyframes[range.start - 1];
      const mounted = await api<{ model_id: string; tools: [string, string] | null }>('tools');
      if (mounted.model_id !== modelId || !mounted.tools || mounted.tools.some((tool, index) => tool !== tools[index]))
        throw new Error('서버의 장착 툴이 다릅니다. 메인 편집기를 새로고침해 주세요.');
      const paths = readLabToolPaths(target.id, tools, graspCoefficients);
      const initialSourceSignature = labSourceSignature({ sourcePaths: paths }, target, source,
        labLateralOffsets(project, target.id, tools));
      for (const path of Object.values(paths)) {
        if (!path || path.source || path.projectName.startsWith('데모:')) continue;
        try {
          const savedProject = await api<Project>(`project/${fileApiPath(path.projectName)}`);
          const response = await api<{ scene: CylinderScene & { plane_y: number } }>('cylinder-lab/scene',
            { project: savedProject, target_id: target.id });
          const settled = restingScene(response.scene);
          path.source = { center: [...settled.target.center], radius: settled.target.radius, planeY: response.scene.plane_y };
        } catch { throw new Error(`${path.toolId}의 경로 기준 위치를 찾지 못했습니다. Cylinder Lab에서 경로를 다시 저장해 주세요.`); }
      }
      const anchorPose = await api<PoseState>('pose', { qpos: source.qpos });
      const pelvis = anchorPose.handles.pelvis;
      const next = withLabLateralOffsets(buildLabPathPreview(target, pelvis.position, pelvis.quaternion, paths,
        100 - project.keyframes.length + range.frames.length), labLateralOffsets(project, target.id, tools));
      const activeSides = (['left', 'right'] as HandSide[]).filter(side => !!next.tracks[side]);
      for (const side of activeSides) {
        if (source.pins.includes(`${side}_hand`) || source.angle_pins?.includes(`${side}_hand`))
          throw new Error(`${side === 'left' ? '왼손' : '오른손'} TCP 고정을 해제한 뒤 다시 시도하세요.`);
      }
      const waypoints = next.times.map((_, index) => {
        const targets: Record<string, number[]> = {}, orientations: Record<string, number[]> = {};
        for (const side of activeSides) {
          const pose = toolPoseAtKeyPoint(next.tracks[side]!, index);
          targets[`${side}_hand`] = pose.position;
          orientations[`${side}_hand`] = pose.quaternion_xyzw;
        }
        return { targets, orientations };
      });
      setLabPathPreview(next);
      setMessage(`변경된 파지법 계산 중 · 웨이포인트 ${waypoints.length}개`);
      const solved = await api<{ qpos: number[][]; first_state: PoseState }>('cylinder-lab/solve-path', {
        qpos: source.qpos, tools, pins: source.pins, angle_pins: source.angle_pins ?? [], waypoints,
      });
      if (solved.qpos.length !== next.times.length) throw new Error('계산된 키프레임 개수가 웨이포인트와 다릅니다.');
      if (current.current.project !== project) throw new Error('계산 중 프로젝트가 변경되었습니다. 다시 업데이트해 주세요.');
      const sourceSignature = labSourceSignature({ sourcePaths: readLabToolPaths(target.id, tools, graspCoefficients) }, target, source,
        labLateralOffsets(project, target.id, tools));
      if (sourceSignature !== initialSourceSignature) throw new Error('계산 중 Cylinder Lab 경로가 변경되었습니다. 다시 업데이트해 주세요.');
      const frames: Keyframe[] = solved.qpos.map((qpos, index) => ({
        name: index === 0 ? '그린 점 1 · 파지점' : `그린 점 ${index + 1}`,
        duration: index === 0 ? .8 : Math.max(.1, next.times[index] - next.times[index - 1]),
        qpos, pins: [...source.pins], angle_pins: [...(source.angle_pins ?? [])],
      }));
      frames.forEach((frame, index) => { frame.lab_grasp = { group_id: range.id, target_id: target.id,
        tools: [...tools], source_signature: sourceSignature, point_index: index,
        authored_signature: authoredSignature(frame) }; });
      setPendingLabReplacement({ project, start: range.start, end: range.end, frames,
        firstState: solved.first_state, editedNames: range.edited.map(frame => frame.name), targetId: target.id });
      setMessage(`새 파지법 ${frames.length}개를 계산했습니다. 교체 내용을 확인해 주세요.`);
    });
  }
  function applyLabReplacement() {
    const pending = pendingLabReplacement;
    if (!pending || !project || project !== pending.project) { setPendingLabReplacement(null); setError('프로젝트가 변경되었습니다. 다시 계산해 주세요.'); return; }
    const target = objects.find(object => object.id === pending.targetId);
    const anchor = project.keyframes[pending.start - 1];
    if (!target || !anchor || !tools || labSourceSignature({ sourcePaths: readLabToolPaths(target.id, tools, graspCoefficients) }, target, anchor,
      labLateralOffsets(project, target.id, tools))
      !== pending.frames[0].lab_grasp?.source_signature) {
      setPendingLabReplacement(null); setError('확인 중 Cylinder Lab 경로나 장면이 변경되었습니다. 다시 계산해 주세요.'); return;
    }
    checkpoint();
    const nextProject = { ...project, keyframes: [...project.keyframes.slice(0, pending.start),
      ...pending.frames, ...project.keyframes.slice(pending.end + 1)] };
    current.current.project = nextProject; setProject(nextProject); setFrameIndex(pending.start);
    setTransitionSelection([]); applyState(pending.firstState);
    setPins([...pending.frames[0].pins]); setAnglePins([...(pending.frames[0].angle_pins ?? [])]);
    setPoseDirty(false); setInfo(null); setLabPathPreview(null); setPendingLabReplacement(null); invalidate();
    setMessage(`파지법 구간을 ${pending.frames.length}개 키프레임으로 교체했습니다.`);
  }
  useEffect(() => {
    if (!project || !state || playing || preview?.physics || policyJob || previewStreaming) return;
    const timer = setTimeout(() => {
      try { localStorage.setItem(draftKey.current, JSON.stringify({ ...project, mounted_tools: tools, current_qpos: state.qpos, pins, angle_pins: anglePins, scene_objects: objects, scene_groups: objectGroups })); }
      catch { setMessage('브라우저 자동 저장 공간이 부족합니다. 파일 저장을 사용하세요.'); }
    }, 500);
    return () => clearTimeout(timer);
  }, [project, state, tools, pins, anglePins, objects, objectGroups, playing, preview, policyJob, previewStreaming]);
  useEffect(() => {
    if (!playing || !preview) return;
    // 재생 속도는 화면 재생 시계에만 적용되며 계산된 모션 데이터는 바뀌지 않는다. 도중에 바꾸면 현재 시각에서 다시 기준을 잡는다.
    let anchorNow = performance.now(), anchorTime = preview.time[sample], anchorSpeed = playbackSpeedRef.current;
    let id = 0;
    const tick = () => {
      const now = performance.now();
      if (anchorSpeed !== playbackSpeedRef.current) {
        anchorTime += (now - anchorNow) / 1000 * anchorSpeed; anchorNow = now; anchorSpeed = playbackSpeedRef.current;
      }
      const time = anchorTime + (now - anchorNow) / 1000 * anchorSpeed;
      const end = preview.time.length - 1;
      // Physics replays may end between display frames when a fall is detected.
      let index = 0;
      while (index < end && preview.time[index + 1] <= time) index++;
      applyState(preview.states[index]);
      if (preview.object_states?.[index]) scene.current?.setObjectPoses(preview.object_states[index]);
      setSample(index);
      if (index === end && !previewStreaming) { setPlaying(false); return; }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, preview, previewStreaming]);

  function togglePin(key: string) {
    if (busy || solving || playing) return;
    checkpoint();
    setPins(p => p.includes(key) ? p.filter(k => k !== key) : [...p, key]);
    invalidate(); setPoseDirty(true);
  }
  function toggleAnglePin(key: string) {
    if (busy || solving || playing) return;
    checkpoint();
    setAnglePins(values => values.includes(key) ? values.filter(value => value !== key) : [...values, key]);
    invalidate(); setPoseDirty(true);
  }
  async function loadProject(next: Project, keepScene = false) {
    setLabPathPreview(null);
    // 모션만 새로 가져올 때는 지금 작업 중인 맵(씬 오브젝트·그룹)을 그대로 이어 쓴다.
    if (keepScene) next = { ...next, scene_objects: structuredClone(objects), scene_groups: structuredClone(objectGroups) };
    if (next.mounted_tools && (modelId !== 'g1-tools' || !tools
      || next.mounted_tools.some((tool, index) => tool !== tools[index]))) {
      if (modelId !== 'g1-tools') await api('model', { model_id: 'g1-tools' });
      await api('tools', { left: next.mounted_tools[0], right: next.mounted_tools[1] });
      localStorage.setItem('g1-motion-draft-v1-g1-tools', JSON.stringify(next));
      location.reload();
      return;
    }
    const valid = await api<Project>('validate', { project: next });
    const pose = await api<PoseState>('pose', { qpos: valid.current_qpos ?? valid.keyframes[0].qpos });
    const nextObjects = objectsFromProject(valid);
    const normalized = withLegacyGraspGhosts({ ...valid, scene_objects: nextObjects }, nextObjects);
    setProject(normalized); current.current.project = normalized; setFrameIndex(0); setTransitionSelection([]);
    setObjects(nextObjects); current.current.objects = nextObjects; selectObject(null);
    const nextObjectGroups = normalized.scene_groups ?? [];
    setObjectGroups(nextObjectGroups); current.current.objectGroups = nextObjectGroups; setSelectedObjectGroupId(null);
    setPins(valid.pins ?? valid.keyframes[0].pins);
    setAnglePins(valid.angle_pins ?? valid.keyframes[0].angle_pins ?? []);
    applyState(pose); invalidate(); setPoseDirty(false); setInfo(null);
    history.current = []; future.current = []; setHistoryCount(0); setFutureCount(0);
    changeGraspPickMode(null);
  }
  async function openFile(source: File) {
    const extension = source.name.slice(source.name.lastIndexOf('.')).toLowerCase();
    let importName = source.name;
    if (extension === '.json') {
      const parsed = JSON.parse(await source.text()) as { info?: unknown; data?: unknown; text?: { goal?: unknown } };
      if ((parsed as { format?: unknown }).format === SCENE_FORMAT) { loadSceneFile(parsed as unknown as SceneFile, source.name); return; }
      if (!(Array.isArray(parsed.data) && parsed.info)) {
        await loadProject(parsed as unknown as Project);
        setMessage(`${source.name} 프로젝트를 불러왔습니다.`);
        return;
      }
      // LeRobot 에피소드는 항상 data.json이므로 작업 목표를 프로젝트 이름으로 쓴다.
      const goal = typeof parsed.text?.goal === 'string' ? parsed.text.goal.trim().replace(/[\\/]/g, '_') : '';
      if (goal) importName = `${goal}.json`;
    }
    const response = await fetch(`/api/import-motion?filename=${encodeURIComponent(importName)}&fps=${fps}`, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: source,
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({})) as { detail?: unknown };
      throw new Error(typeof failure.detail === 'string' ? failure.detail : `파일 불러오기 실패 (${response.status})`);
    }
    await loadProject(await response.json() as Project, true);
    setMessage(`${source.name}의 모션 프레임을 ${fps} FPS 키프레임으로 불러왔습니다. 맵(씬 오브젝트)은 그대로 유지했습니다.`);
  }
  // 열려 있는 프로젝트는 그대로 두고, 가져온 LeRobot 모션을 클립으로 키프레임 목록 끝에 덧붙인다.
  async function addClipFiles(sources: File[]) {
    if (!project) return;
    let keyframes = project.keyframes;
    const added: string[] = [];
    for (const source of sources) {
      let importName = source.name;
      if (source.name.toLowerCase().endsWith('.json')) {
        const parsed = JSON.parse(await source.text()) as { info?: unknown; data?: unknown; text?: { goal?: unknown }; format?: unknown };
        if (!(Array.isArray(parsed.data) && parsed.info)) throw new Error(`${source.name}은 LeRobot 에피소드(data.json)가 아닙니다. 프로젝트나 맵 파일은 "열기"로 불러오세요.`);
        const goal = typeof parsed.text?.goal === 'string' ? parsed.text.goal.trim().replace(/[\\/]/g, '_') : '';
        if (goal) importName = `${goal}.json`;
      }
      const response = await fetch(`/api/import-motion?filename=${encodeURIComponent(importName)}&fps=${fps}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: source,
      });
      if (!response.ok) {
        const failure = await response.json().catch(() => ({})) as { detail?: unknown };
        throw new Error(typeof failure.detail === 'string' ? failure.detail : `${source.name} 불러오기 실패 (${response.status})`);
      }
      const imported = await response.json() as Project;
      const clips = imported.keyframes.filter(frame => frame.samples).map(frame => ({ ...frame, name: imported.name || frame.name }));
      if (keyframes.length + clips.length > 100) throw new Error('키프레임은 최대 100개입니다.');
      for (const clip of clips) {
        let name = clip.name, number = 2;
        while (keyframes.some(frame => frame.name === name)) name = `${clip.name} (${number++})`;
        keyframes = [...keyframes, { ...clip, name }];
        added.push(name);
      }
    }
    if (!added.length) return;
    checkpoint();
    const next = { ...project, keyframes };
    current.current.project = next;
    setProject(next); setFrameIndex(keyframes.length - added.length); setTransitionSelection([]); invalidate();
    setMessage(`클립 ${added.length}개를 목록 끝에 추가했습니다: ${added.join(', ')}. 앞 항목에서 새 클립으로는 자동으로 이어지며, 이동 시간은 앞 카드의 "다음 카드까지"에서 바꿀 수 있습니다.`);
  }
  function saveSceneFile() {
    const scene: SceneFile = { format: SCENE_FORMAT, scene_objects: objects.filter(object => !object.ghost), scene_groups: objectGroups };
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(scene, null, 2)], { type: 'application/json' }));
    link.download = `${(project?.display_name || project?.name || 'map').replace(/[\\/]/g, '_')}.scene.json`;
    link.click(); URL.revokeObjectURL(link.href);
    setMessage(`맵 ${scene.scene_objects.length}개 오브젝트를 별도 파일로 저장했습니다. 모션·로봇 정보는 들어 있지 않습니다.`);
  }
  function loadSceneFile(scene: SceneFile, name: string) {
    if (!project || !Array.isArray(scene.scene_objects)) throw new Error('맵 파일 형식이 올바르지 않습니다.');
    const nextObjects = objectsFromProject({ scene_objects: scene.scene_objects });
    const nextGroups = Array.isArray(scene.scene_groups) ? structuredClone(scene.scene_groups) : [];
    checkpoint();
    setObjects(nextObjects); current.current.objects = nextObjects; selectObject(null);
    setObjectGroups(nextGroups); current.current.objectGroups = nextGroups; setSelectedObjectGroupId(null);
    const nextProject = { ...project, scene_objects: nextObjects, scene_groups: nextGroups };
    setProject(nextProject); current.current.project = nextProject; changeGraspPickMode(null); invalidate();
    setMessage(`${name} 맵을 불러왔습니다. 키프레임과 로봇 정보는 그대로입니다.`);
  }
  function editFrame(index: number, patch: Partial<Keyframe>) {
    setProject(p => p ? { ...p, keyframes: p.keyframes.map((f, i) => {
      if (i !== index) return f;
      if ('grasp' in patch) return { ...f, ...patch };
      const grasp = patch.qpos && f.grasp ? { ...f.grasp, closure_qpos: undefined,
        object_signature: undefined, contact_points_world: undefined, hand_twist_deg: undefined } : f.grasp;
      // A manually authored pose must not keep the auto-planner's old TCP target.
      const frame = { ...f };
      if (patch.qpos) delete frame.interaction;
      return { ...frame, ...patch, ...(grasp ? { grasp } : {}) };
    }) } : p);
    invalidate();
  }
  function changeGrasp(grasp?: TwoHandGrasp) {
    checkpoint();
    editFrame(frameIndex, { grasp });
    if (!grasp) changeGraspPickMode(null);
  }
  async function fitGrasp() {
    const frame = project?.keyframes[frameIndex];
    const grasp = frame?.grasp;
    const baseObject = objects.find(value => value.id === grasp?.object_id);
    const object = objectAtGraspPose(frame, objects);
    if (!frame || !grasp || !object) return;
    await run(async () => {
      const result = await api<{ state: PoseState; grasp: TwoHandGrasp; solver: { contact: SolveInfo } }>('grasp-fit', {
        qpos: frame.qpos, pins: frame.pins, angle_pins: frame.angle_pins ?? [], object,
        grasp: { ...grasp, follow_object: grasp.follow_object !== false,
          object_pose: grasp.object_pose ?? { position: [...object.position], quaternion_xyzw: [...object.quaternion_xyzw] } },
      });
      checkpoint(); applyState(result.state); setPins([...frame.pins]); setAnglePins([...(frame.angle_pins ?? [])]); setPoseDirty(false); setInfo(result.solver.contact);
      const activeProject = current.current.project;
      if (activeProject) {
        const nextProject = { ...activeProject, keyframes: activeProject.keyframes.map((item, index) => index === frameIndex
          ? { ...item, qpos: [...result.state.qpos], grasp: result.grasp, interaction: undefined } : item) };
        current.current.project = nextProject; setProject(nextProject);
      }
      invalidate(); changeGraspPickMode(null);
      setMessage(`${baseObject?.name ?? '박스'} 위치를 키프레임 고스트로 저장하고 양손 파지 자세를 맞췄습니다 · 접촉점 오차 ${result.solver.contact.target_error_mm.toFixed(2)} mm · 손목 비틀림 ${result.grasp.hand_twist_deg?.toFixed(0) ?? 0}°`);
    });
  }
  async function generateRamenSequence(objectId: string, settings: RamenSequenceSettings) {
    if (!project || !state) return;
    const object = objects.find(value => value.id === objectId);
    if (!object) return;
    await run(async () => {
      const result = await api<{ keyframes: Keyframe[]; diagnostics: { target_error_mm: number; angle_error_deg: number }[]; object_reference: { center_world: number[]; diameter_m: number } }>('ramen-sequence', {
        qpos: state.qpos, pins, angle_pins: anglePins, object, settings, attention_pose: project.attention_pose,
      });
      const retained = project.keyframes.filter(frame => frame.interaction?.task !== 'ramen_extract');
      if (retained.length + result.keyframes.length > 100) throw new Error('시퀀스를 추가하면 키프레임 최대 100개를 넘습니다.');
      checkpoint();
      const firstIndex = retained.length;
      const keyframes = [...retained, ...result.keyframes];
      const nextProject = { ...project, keyframes };
      current.current.project = nextProject; setProject(nextProject); setTransitionSelection([]);
      const first = await api<PoseState>('pose', { qpos: result.keyframes[0].qpos });
      applyState(first); setPins([...result.keyframes[0].pins]); setAnglePins([...(result.keyframes[0].angle_pins ?? [])]);
      setFrameIndex(firstIndex); setPoseDirty(false); setInfo(null); invalidate();
      const maxPosition = Math.max(...result.diagnostics.map(value => value.target_error_mm));
      const maxAngle = Math.max(...result.diagnostics.map(value => value.angle_error_deg));
      const center = result.object_reference.center_world.map(value => value.toFixed(3)).join(', ');
      setMessage(`${object.name} 중심 (${center}) 기준 ${result.keyframes.length}단계를 생성했습니다 · 지름 ${(result.object_reference.diameter_m * 1000).toFixed(0)} mm · TCP 오차 최대 ${maxPosition.toFixed(1)} mm / ${maxAngle.toFixed(1)}°`);
    });
  }
  function addFrame() {
    if (!project || !state || !project.keyframes[frameIndex]) return;
    if (project.keyframes[frameIndex].samples) {
      let added;
      try { added = insertPoseAfterClip(project.keyframes, frameIndex, feet); }
      catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); return; }
      const pose = added.keyframes[added.index];
      void run(async () => {
        const restored = await api<PoseState>('pose', { qpos: pose.qpos });
        checkpoint();
        setProject({ ...project, keyframes: added.keyframes }); setFrameIndex(added.index); setTransitionSelection([]);
        applyState(restored); setPins([...pose.pins]); setAnglePins([]); setPoseDirty(false); setInfo(null); invalidate();
        setMessage('클립 뒤에 편집용 자세를 추가했습니다. 손이나 관절을 수정한 뒤 "선택 프레임에 반영"을 누르세요.');
      });
      return;
    }
    const duplicated = duplicateKeyframeKeepingTimes(project.keyframes, frameIndex);
    const source = project.keyframes[frameIndex];
    const finish = (restored?: PoseState) => {
      setProject({ ...project, keyframes: duplicated.keyframes }); setFrameIndex(duplicated.index); setTransitionSelection([]);
      if (restored) applyState(restored);
      setPins([...source.pins]); setAnglePins([...(source.angle_pins ?? [])]); setPoseDirty(false); setInfo(null); invalidate();
      setMessage('선택한 키프레임을 바로 다음 순번에 복제했습니다.');
    };
    const displayedPoseDiffers = source.qpos.some((value, index) => Math.abs(value - state.qpos[index]) > 1e-10)
      || source.pins.length !== pins.length || source.pins.some(pin => !pins.includes(pin))
      || (source.angle_pins?.length ?? 0) !== anglePins.length || (source.angle_pins ?? []).some(pin => !anglePins.includes(pin));
    if (poseDirty || displayedPoseDiffers) {
      void run(async () => { const restored = await api<PoseState>('pose', { qpos: source.qpos }); checkpoint(); finish(restored); });
    } else finish();
  }
  async function chooseFrame(index: number) {
    if (!project) return;
    const f = project.keyframes[index];
    await run(async () => { checkpoint(); changeGraspPickMode(null); applyState(await api<PoseState>('pose', { qpos: f.qpos })); setPins(f.pins); setAnglePins(f.angle_pins ?? []); setFrameIndex(index); setPoseDirty(false); setInfo(null); });
  }
  function selectTransitionFrame(index: number) {
    setTransitionSelection(previous => previous.length === 1 && previous[0] !== index
      ? [previous[0], index] : [index]);
    // 재생 막대도 그 카드가 시작하는 시각으로 옮긴다. chooseFrame이 카드의 첫 자세를 불러오므로 둘이 같은 지점이 된다.
    if (preview && !preview.physics) {
      const start = (preview.keyframe_times ?? cardStarts)[index];
      if (start !== undefined) {
        const at = preview.time.findIndex(time => time >= start - 1e-6);
        setPlaying(false);
        setSample(at < 0 ? preview.time.length - 1 : at);
      }
    }
    void chooseFrame(index);
  }
  function createSmoothTransition() {
    if (!project || transitionSelection.length !== 2 || poseDirty) return;
    const [start, end] = [...transitionSelection].sort((a, b) => a - b);
    if (project.keyframes.slice(start, end + 1).some(frame => frame.samples))
      { setError('클립은 자동 중간 자세를 만들 수 없습니다. 먼저 클립을 키프레임으로 변환하세요.'); return; }
    const retimed = retimeSmoothTransition(project.keyframes, start, end);
    if (retimed === project.keyframes) return;
    void run(async () => {
      if (retimed.slice(start, end + 1).some(frame => frame.grasp || frame.interaction))
        throw new Error('파지·도구 상호작용이 있는 구간은 자동 중간 자세를 지원하지 않습니다. 해당 구간은 직접 키프레임을 추가해 주세요.');
      // Sample the existing pin-aware path, then make its midpoint poses editable
      // keyframes. Verify the new path before changing the project.
      // 선택 구간만 따로 계산해 클립이 섞인 프로젝트에서도 샘플 위치가 어긋나지 않게 한다.
      const section = retimed.slice(start, end + 1);
      const source = await api<Preview>('preview', { project: { ...project, keyframes: section }, fps: 30 });
      const keyframes = [...retimed.slice(0, start), ...insertSmoothTransitionFrames(section, 0, section.length - 1, source.states, 30), ...retimed.slice(end + 1)];
      const updated = { ...project, keyframes };
      const result = await api<Preview>('preview', { project: updated, fps: 30 });
      checkpoint(); current.current.project = updated; setProject(updated);
      const mappedFrame = frameIndex <= start ? frameIndex : frameIndex <= end
        ? frameIndex + frameIndex - start : frameIndex + end - start;
      setFrameIndex(mappedFrame); current.current.frameIndex = mappedFrame;
      setTransitionSelection([start, end + end - start]);
      if (physicsEnabled || objectPhysicsEnabled) invalidate();
      else {
        const firstSample = Math.round((result.keyframe_times?.[start] ?? 0) * 30);
        setPreview(result); setSample(Math.min(firstSample, result.states.length - 1));
      }
      setMessage(`중간 자세 ${end - start}개를 키프레임으로 추가했습니다. ${physicsEnabled || objectPhysicsEnabled ? '물리 재생은 시작 프레임을 선택한 뒤 실행하세요.' : '재생을 누르면 선택 구간부터 확인할 수 있습니다.'} 장애물 회피·물리 안정성은 별도 검증이 필요합니다.`);
    });
  }
  function changeHistory(redo: boolean) {
    if (current.current.busy || current.current.playing || inFlight.current || dragActive.current) return;
    setLabPathPreview(null);
    void run(async () => {
      const from = redo ? future.current : history.current;
      const to = redo ? history.current : future.current;
      const value = from[from.length - 1];
      if (!value || !current.current.state) return;
      const currentSnapshot = editorSnapshot();
      if (!currentSnapshot) return;
      const poseChanged = value.qpos.some((position, index) => Math.abs(position - current.current.state!.qpos[index]) > 1e-12);
      const restored = poseChanged ? await api<PoseState>('pose', { qpos: value.qpos }) : current.current.state;
      from.pop();
      to.push(currentSnapshot);
      const restoredObjects = structuredClone(value.objects);
      const restoredGroups = structuredClone(value.objectGroups);
      applyState(restored); current.current.pins = [...value.pins]; setPins([...value.pins]);
      current.current.anglePins = [...value.anglePins]; setAnglePins([...value.anglePins]); setInfo(null);
      current.current.objects = restoredObjects; setObjects(restoredObjects);
      current.current.objectGroups = restoredGroups; setObjectGroups(restoredGroups);
      setProject(projectValue => projectValue ? { ...projectValue, scene_objects: restoredObjects, scene_groups: restoredGroups, keyframes: structuredClone(value.keyframes) } : projectValue);
      setFrameIndex(value.frameIndex); current.current.frameIndex = value.frameIndex; setTransitionSelection([]);
      if (current.current.selectedObjectId) {
        selectObject(restoredObjects.some(object => object.id === current.current.selectedObjectId)
          ? current.current.selectedObjectId : null);
      }
      current.current.poseDirty = value.poseDirty; setPoseDirty(value.poseDirty);
      setHistoryCount(history.current.length); setFutureCount(future.current.length); invalidate();
      setMessage(redo ? '되돌리기를 취소해 마지막 편집을 다시 적용했습니다.' : '마지막 편집을 되돌렸습니다.');
    });
  }
  function numericMove(values: number[]) {
    if (values.some(v => !Number.isFinite(v)) || busy || solving || members.some(k => pins.includes(k))) return;
    begin(); dragActive.current = false; move(selected, values);
  }
  function numericRotate(values: number[]) {
    if (values.some(v => !Number.isFinite(v)) || busy || solving || rotationBlocked) return;
    begin(); dragActive.current = false; rotate(selected, quaternionFromDegrees(values));
  }
  function nudgeRotation(axis: number, amount: number) {
    if (!state || busy || solving || rotationBlocked) return;
    begin(); dragActive.current = false;
    rotate(selected, incrementRotation(state.handles[controlKey(members, selected)].quaternion, axis, amount, space));
  }
  function applyJointDraft() {
    if (!state || !project || busy || solving) return;
    const joints = Object.fromEntries(project.joint_names.map((name, i) => [name, jointDraft[i] * Math.PI / 180] as const).filter(([,value], i) => Math.abs(value-state.qpos[7+i]) > 1e-6));
    if (!Object.keys(joints).length) return;
    begin(); dragActive.current = false;
    pending.current = { kind: 'joint', key: '', target: [], joints }; void drain();
  }
  function changeJoint(index: number, value: number) {
    if (!Number.isFinite(value) || !limits[index]) return;
    setJointDraft(draft => draft.map((v, i) => i === index ? Math.max(limits[i][0], Math.min(limits[i][1], value)) : v));
  }
  const nameOnFocus = useRef('');
  // Renaming keeps the saved project's folder and contents; only its name and folder name change.
  async function renameSavedProject() {
    const value = current.current.project;
    if (!value || value.name === nameOnFocus.current) return;
    nameOnFocus.current = value.name;
    if (!value.project_id) return;
    try {
      const result = await api<{ renamed: boolean; display_name?: string }>('rename-project', { project_id: value.project_id, name: value.name });
      if (!alive.current) return;
      if (result.renamed) {
        setSaved(await api<string[]>('saved'));
        setProject(previous => previous ? { ...previous, display_name: result.display_name } : previous);
        setMessage(`저장된 파일 이름을 '${result.display_name}'(으)로 변경했습니다. 기존 파일은 그대로 유지됩니다.`);
      } else setMessage('이름을 바꿨습니다. 처음 저장할 때 이 이름의 파일이 만들어집니다.');
    } catch (failure) { setError((failure as Error).message); }
  }
  function exportProject(saveAs = false) {
    if (!project || !state) return;
    void run(async () => {
      const result = await api<{ files: string[]; directory: string; project: Project; display_name: string; reused: boolean; warnings?: string[]; metadata: { samples: number; validation: { max_pin_error_mm: number } } }>(saveAs ? 'save-as' : 'save', {
        project: { ...project, mounted_tools: tools, current_qpos: state.qpos, pins, angle_pins: anglePins, scene_objects: objects, scene_groups: objectGroups }, fps, protomotions: exportProto,
      });
      setProject(result.project); current.current.project = result.project;
      setFiles(result.files); setSaved(await api<string[]>('saved'));
      const action = result.reused ? '프로젝트 업데이트' : saveAs ? '새 프로젝트로 저장' : '프로젝트 생성';
      setMessage(`${action} 완료 · ${result.display_name} · ${result.metadata.samples}프레임 · 고정 오차 최대 ${result.metadata.validation.max_pin_error_mm.toFixed(2)} mm`);
      if (result.warnings?.length) setError(result.warnings.join(' '));
    });
  }
  const controllerLabel = objectPhysicsEnabled ? '오브젝트 물리' : policyEnabled ? 'GEAR-SONIC' : '기본 PD';
  async function play(forcePhysics = false, startFrameOverride?: number) {
    if (playing) { setPlaying(false); return; }
    if (preview) { if (sample >= preview.states.length - 1) setSample(0); setPlaying(true); return; }
    await run(async () => {
      if (!physicsEnabled && !objectPhysicsEnabled && !forcePhysics) {
        setPreviewStreaming(true);
        setMessage('키프레임 모션을 구간별로 계산하며 재생합니다…');
        let connectionFailed = false;
        try {
          await new Promise<void>((resolve, reject) => {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const socket = new WebSocket(`${protocol}//${location.host}/api/preview/stream`);
            const request = { socket, cancelled: false, sourceState: state };
            previewStream.current = request;
            let settled = false, started = false;
            const finish = (failure?: Error) => {
              if (settled) return;
              settled = true;
              socket.close();
              if (failure) reject(failure); else resolve();
            };
            socket.onopen = () => socket.send(JSON.stringify({ project, fps: 30 }));
            socket.onmessage = event => {
              const packet = JSON.parse(event.data) as { type: 'batch'; time: number[]; states: PoseState[] }
                | { type: 'done'; max_pin_error_mm: number; keyframe_times?: number[] | null; dynamic_reports?: DynamicReport[]; motion_reports?: MotionReport[]; original_states?: (PoseState | null)[] | null } | { type: 'error'; message: string };
              if (request.cancelled || !alive.current) return;
              if (packet.type === 'batch') {
                if (!started) {
                  started = true;
                  setPreview({ time: packet.time, states: packet.states, max_pin_error_mm: 0 });
                  setSample(0); setPlaying(true); setPoseDirty(false);
                } else setPreview(previous => previous ? { ...previous,
                  time: [...previous.time, ...packet.time], states: [...previous.states, ...packet.states] } : previous);
              } else if (packet.type === 'done') {
                setPreview(previous => previous ? { ...previous, max_pin_error_mm: packet.max_pin_error_mm, keyframe_times: packet.keyframe_times, dynamic_reports: packet.dynamic_reports, motion_reports: packet.motion_reports, original_states: packet.original_states } : previous);
                finish();
              } else finish(new Error(packet.message));
            };
            socket.onerror = () => { connectionFailed = true; finish(new Error('키프레임 스트리밍 연결이 끊겼습니다.')); };
            socket.onclose = () => {
              if (!request.cancelled) connectionFailed = true;
              finish(request.cancelled ? undefined : new Error('키프레임 스트리밍이 중단되었습니다.'));
            };
          });
          if (previewStream.current?.cancelled) return;
          setMessage('원본 모션 재생 · 물리 OFF');
        } catch (failure) {
          setPreview(null); setPlaying(false);
          if (state) applyState(state);
          if (!connectionFailed || previewStream.current?.cancelled) throw failure;
          setMessage('실시간 연결을 사용할 수 없어 키프레임 모션을 계산하는 중…');
          const result = await api<Preview>('preview', { project, fps: 30 });
          setPreview(result); setSample(0); setPlaying(true); setPoseDirty(false);
          setMessage('원본 모션 재생 · 실시간 연결 재시도 필요');
        } finally {
          previewStream.current = null;
          setPreviewStreaming(false);
        }
        return;
      }
      const request: { id?: string; cancelled: boolean; socket?: WebSocket } = { cancelled: false };
      const physicsStartFrame = startFrameOverride ?? frameIndex;
      // The override selects where the simulation begins; restoring Physics OFF must
      // still return to the pose/keyframe that was visible before the run.
      policySource.current = state ? { state, pins: [...pins], anglePins: [...anglePins], frameIndex, dirty: poseDirty } : null;
      policyRequest.current = request;
      try {
        let streamFallback = false;
        const startFrame = project?.keyframes[physicsStartFrame];
        setMessage(`${controllerLabel} · ${physicsStartFrame + 1}번 ${startFrame?.name ?? '키프레임'}부터 물리 계산 준비 중…`);
        let job = await api<PolicyJob>('policy-preview', { project, controller: objectPhysicsEnabled && !forcePhysics ? 'objects' : policyEnabled ? 'gear-sonic' : 'pd', start_frame_index: physicsStartFrame });
        request.id = job.id;
        if (request.cancelled) {
          await api(`policy-preview/${job.id}/cancel`, {});
          return;
        }
        setPolicyJob(job);
        if (job.status === 'running') {
          try {
            job = await new Promise<PolicyJob>((resolve, reject) => {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const socket = new WebSocket(`${protocol}//${location.host}/api/policy-preview/${job.id}/stream`);
            request.socket = socket;
            let settled = false;
            const finish = (result?: PolicyJob, failure?: Error) => {
              if (settled) return;
              settled = true;
              socket.close();
              if (failure) reject(failure);
              else resolve(result!);
            };
            socket.onmessage = event => {
              const packet = JSON.parse(event.data) as PhysicsStreamFrame | ({ type: 'status' } & PolicyJob);
              if (packet.type === 'frame') {
                if (request.cancelled || !alive.current) return;
                applyState(packet.state);
                scene.current?.setObjectPoses(packet.objects);
                setStreamTime(packet.time);
                setMessage(`${controllerLabel} · 물리 실시간 재생 ${packet.time.toFixed(2)}초 · CPU`);
              } else {
                setPolicyJob(packet);
                if (packet.status !== 'running') finish(packet);
              }
            };
            socket.onerror = () => finish(undefined, new Error('물리 스트리밍 연결이 끊겼습니다.'));
            socket.onclose = () => finish(request.cancelled
              ? { id: job.id, status: 'cancelled', progress: 0 } : undefined,
              request.cancelled ? undefined : new Error('물리 스트리밍이 중단되었습니다.'));
            });
          } catch (streamFailure) {
            if (!alive.current || request.cancelled) return;
            streamFallback = true;
            // A development proxy or an older server may reject the WebSocket while
            // the simulation job itself continues. Keep the result reachable.
            request.socket = undefined;
            setStreamTime(null);
            setMessage(`${controllerLabel} · 실시간 연결이 끊겨 계산 결과를 기다리는 중…`);
            while (job.status === 'running') {
              await new Promise(resolve => setTimeout(resolve, 500));
              if (!alive.current || request.cancelled) return;
              try { job = await api<PolicyJob>(`policy-preview/${job.id}`); }
              catch { throw streamFailure; }
              setPolicyJob(job);
              setMessage(`${controllerLabel} · 계산 ${Math.round(job.progress * 100)}% · 실시간 연결 재시도 필요`);
            }
          }
        }
        if (!alive.current || request.cancelled || job.status === 'cancelled') return;
        if (job.status !== 'completed') throw new Error(job.message || '물리 계산에 실패했습니다.');
        const result = await api<Preview>(`policy-preview/${job.id}/result`);
        if (!alive.current || request.cancelled) return;
        setStreamTime(null); setPreview(result); setPoseDirty(false);
        if (job.status === 'completed' && job.progress === 1 && result.states.length && request.socket) {
          setSample(result.states.length - 1);
          applyState(result.states[result.states.length - 1]);
          if (result.object_states?.length) scene.current?.setObjectPoses(result.object_states[result.object_states.length - 1]);
        } else {
          setSample(0); setPlaying(true);
          if (result.object_states?.[0]) scene.current?.setObjectPoses(result.object_states[0]);
        }
        const summary = result.summary!;
        const outcome = summary.reason === 'fallen' ? '넘어짐으로 조기 종료' : summary.reason === 'completed' ? '계산 완료'
          : summary.reason === 'interaction_force_limit' ? '접촉력 상한으로 안전 중단' : '수치 불안정으로 중단';
        const grip = summary.grasp;
        const gripText = grip ? ` · 양손 ${grip.bilateral_contact ? '접촉' : '접촉 실패'} · 힘 L ${grip.left.max_normal_n.toFixed(1)} / R ${grip.right.max_normal_n.toFixed(1)} N${grip.force_limit_exceeded ? ' · 힘 상한 초과' : ''}` : '';
        const interaction = summary.interaction;
        const interactionText = interaction ? ` · TCP ${interaction.max_tcp_error_mm.toFixed(1)} mm / ${interaction.max_orientation_error_deg.toFixed(1)}° · 접촉 L ${interaction.max_contact_force_n.left.toFixed(1)} / R ${interaction.max_contact_force_n.right.toFixed(1)} N${interaction.force_limit_exceeded ? ' · 힘 상한 중단' : ''}` : '';
        setMessage(`${controllerLabel} ${outcome} · ${physicsStartFrame + 1}번 키프레임부터 ${summary.sim_seconds.toFixed(2)}초 · 관절 추종 오차 ${(summary.joint_rmse_rad * 180 / Math.PI).toFixed(1)}°${gripText}${interactionText}${streamFallback ? ' · 실시간 연결 실패로 결과 재생' : ''}`);
      } catch (failure) {
        if (request.id) await api(`policy-preview/${request.id}/cancel`, {}).catch(() => {});
        restorePolicySource();
        throw failure;
      } finally {
        request.socket?.close();
        if (alive.current) setPolicyJob(null);
        policyRequest.current = null;
      }
    });
  }
  async function validateGraspPhysics() {
    const grasp = project?.keyframes[frameIndex]?.grasp;
    if (!grasp?.object_signature || grasp.closure_qpos) return;
    if (!physicsEnabled) { setPhysicsEnabled(true); setObjectPhysicsEnabled(false); setMessage('양손 파지 물리 검증을 준비합니다.'); }
    await play(true);
  }
  async function cancelPolicy() {
    const request = policyRequest.current;
    if (!request?.id) return;
    try {
      request.cancelled = true;
      await api(`policy-preview/${request.id}/cancel`, {});
      request.socket?.close();
      restorePolicySource();
      setMessage('물리 재생을 중단하고 편집 자세로 돌아왔습니다.');
    } catch (failure) { setError((failure as Error).message); }
  }
  function cancelPreviewStream() {
    const request = previewStream.current;
    if (!request) return;
    request.cancelled = true;
    request.socket.close();
    setPreview(null); setSample(0); setPlaying(false);
    if (request.sourceState) applyState(request.sourceState);
    setMessage('키프레임 재생을 중단하고 편집으로 돌아왔습니다.');
  }
  function restorePolicySource() {
    const source = policySource.current;
    if (source) {
      applyState(source.state); setFrameIndex(source.frameIndex); setPins(source.pins);
      setAnglePins(source.anglePins); setPoseDirty(source.dirty);
    }
    policySource.current = null;
    scene.current?.setSceneObjects(current.current.objects);
    setStreamTime(null); setPreview(null); setSample(0); setPlaying(false);
  }
  function changeSimulation(physics: boolean, policy: boolean, objects = false) {
    const source = policySource.current;
    if (preview?.physics && source) {
      applyState(source.state); setFrameIndex(source.frameIndex); setPins(source.pins); setAnglePins(source.anglePins); setPoseDirty(source.dirty);
    }
    policySource.current = null;
    scene.current?.setSceneObjects(current.current.objects);
    invalidate(); setPhysicsEnabled(physics); setObjectPhysicsEnabled(objects); setPolicyEnabled(physics && policy);
    setMessage(objects ? '키프레임 로봇 자세를 유지하고 오브젝트만 중력·접촉으로 움직입니다. 로봇 토크·파지 가능성은 검증하지 않습니다.'
      : !physics ? '원본 모션 재생 · 물리 OFF' : policy
      ? 'MuJoCo + GEAR-SONIC · 계산되는 자세와 충돌을 실시간으로 보여줍니다.'
      : 'MuJoCo + 기본 PD · 계산되는 자세와 충돌을 실시간으로 보여줍니다. 균형 정책이 없어 넘어질 수 있습니다.');
  }
  // 재생 버튼을 누르지 않아도 타임라인 슬라이더가 바로 쓰이도록, 편집이 멈추면 미리보기를 미리 계산해 둔다(재생은 하지 않는다).
  const failedPreview = useRef<unknown>(null);
  useEffect(() => {
    if (!project || !state || preview || playing || previewStreaming || busy || solving || failedPreview.current === project) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      api<Preview>('preview', { project, fps: 30 }).then(result => {
        const now = current.current;
        if (cancelled || now.project !== project || now.playing || now.busy || now.solving) return;
        setPreview(previous => previous ?? result);
        setSample(Math.max(0, result.time.findIndex(time => time >= scrubTime.current - 1e-9)));
      }).catch(() => { failedPreview.current = project; });      // a project that does not compile reports its error when played
    }, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [project, !!state, preview, playing, previewStreaming, busy, solving]); // eslint-disable-line react-hooks/exhaustive-deps -- project edits go through invalidate()
  const disabled = !state || busy || solving || playing || !!preview?.physics;
  const selectionPinned = members.some(k => pins.includes(k));
  const controls = controlSelection(members);
  const mirrorAvailable = canMirrorSelection(controls);
  const mirrorActive = mirror && mirrorAvailable && (transformMode === 'translate' || isJointPairSelection(controls));
  const activeControl = controlKey(members, selected);
  const rotationAllowed = canRotateSelection(controls);
  const selectionAnglePinned = members.some(k => anglePins.includes(k)) || controls.some(k => anglePins.includes(k));
  const rotationBlocked = !rotationAllowed || selectionAnglePinned || controls.some(k => pins.includes(k) && (controls.length > 1 || feet.includes(k)));
  const hinge = controls.length === 1 || isJointPairSelection(controls) ? state?.hinges?.[activeControl] : undefined;
  const ankle = controls.length === 1 && ANKLE_HANDLES.includes(activeControl);
  const selectedJoints = members.filter(key => key.endsWith('_joint'));
  const partJoints = selectedJoints.length > 1 ? selectedJoints : controls.length === 1 ? COMBINED_JOINTS[activeControl] : undefined;
  const hingeIndex = hinge ? project?.joint_names.indexOf(hinge.joint_name) ?? -1 : -1;
  const jointDirty = !!state && jointDraft.some((v, i) => Math.abs(v - state.qpos[7+i]*180/Math.PI) > .001);
  const activeFrame = project?.keyframes[frameIndex];
  const [splitSample, setSplitSample] = useState(1);
  // 기본 분할 위치는 클립의 가운데이고, 슬라이더가 이 클립 안에 있으면 그 위치를 따라간다.
  const splitClipLength = project?.keyframes[frameIndex]?.samples?.length ?? 0;
  useEffect(() => {
    if (!project || splitClipLength < 3) return;
    const scrubbed = preview ? clipSampleAtTime(project.keyframes, frameIndex, preview.time[sample] ?? 0, preview.keyframe_times?.[frameIndex]) : null;
    setSplitSample(scrubbed ?? middleSample(splitClipLength));
  }, [preview, sample, frameIndex, splitClipLength]); // eslint-disable-line react-hooks/exhaustive-deps -- project edits go through invalidate()
  function mergeActiveClip() {
    if (!project) return;
    try {
      const result = mergeMotionClips(project.keyframes, frameIndex);
      checkpoint();
      setProject({ ...project, keyframes: result.keyframes }); setFrameIndex(result.index); setTransitionSelection([]); invalidate();
      setMessage('선택한 클립을 다음 클립과 합쳤습니다. 원본 동작과 총 재생 시간은 그대로입니다.');
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
  }
  const [convertCount, setConvertCount] = useState(2);
  const [convertSmooth, setConvertSmooth] = useState(true);
  function convertActiveClip() {
    if (!project || !activeFrame?.samples) return;
    const frame = activeFrame;
    void run(async () => {
      let source = project.keyframes;
      if (frame.dynamic) {
        // 동적 파지 클립은 물체의 현재 위치로 보정된 자세를 키프레임으로 굳힌다.
        const resolved = await api<{ samples: number[][] }>('dynamic-resolve', { project: sceneProject(project), index: frameIndex });
        const { dynamic: _bound, ...plain } = frame;
        source = project.keyframes.map((item, index) => index === frameIndex ? { ...plain, samples: resolved.samples, qpos: resolved.samples[0] } : item);
      }
      let result;
      try { result = convertClipToPoses(source, frameIndex, convertCount, feet); }
      catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); return; }
      const first = result.keyframes[frameIndex];
      const restored = await api<PoseState>('pose', { qpos: first.qpos });
      checkpoint();
      // 클립의 동작 방식을 새 키프레임이 이어받는다: 델타는 모두, 카테시안(목표)은 마지막만(앞은 델타), 원본 그대로는 없음.
      const converted = result.keyframes.map((item, i) => {
        if (i < frameIndex || i >= frameIndex + convertCount) return item;
        const last = i === frameIndex + convertCount - 1;
        const motion: CardMotion | undefined = !frame.motion || frame.motion.mode === 'recorded' ? undefined
          : frame.motion.mode === 'cartesian' && last ? structuredClone(frame.motion) : { mode: 'delta' };
        return { ...item, ...(convertSmooth ? { generated_transition: true } : {}), ...(motion ? { motion } : {}) };
      });
      setProject({ ...project, keyframes: converted }); setFrameIndex(frameIndex); setTransitionSelection([]);
      applyState(restored); setPins([...first.pins]); setAnglePins([]); setPoseDirty(false); setInfo(null); invalidate();
      setMessage(`클립을 키프레임 ${convertCount}개로 변환했습니다.${frame.dynamic ? ' 물체의 현재 위치에 맞춘 자세로 만들었고 동적 파지 설정은 해제되었습니다.' : ''} 손이나 관절을 수정한 뒤 "선택 프레임에 반영"을 누르세요. 양발은 고정되어 있습니다.`);
    });
  }
  // 서버가 동적 클립을 풀 때 쓰는 프로젝트: 지금 화면의 물체 위치를 담는다.
  function sceneProject(base: Project): Project { return { ...base, scene_objects: objects.filter(object => !object.ghost), scene_groups: objectGroups }; }
  const motionObjects = objects.filter(object => !object.ghost);
  const motionReport = preview?.motion_reports?.find(report => report.index === frameIndex) ?? null;
  const modeLabel = { recorded: '원본 그대로', delta: '관절 델타', cartesian: '카테시안' } as const;
  /** 선택 프레임에 반영: 뒤 카드가 앞 카드 이동 연결로 이어져 있으면 그 카드들도 새 끝점에서 이어지도록 함께 옮긴다. */
  async function applyPoseToFrame() {
    if (!project || !state) return;
    const qpos = [...state.qpos], index = frameIndex;
    const patch = { pins: [...pins], angle_pins: [...anglePins] };
    if (project.keyframes[index + 1]?.motion?.mode !== 'delta') {
      editFrame(index, { qpos, ...patch }); setPoseDirty(false); setMessage('선택한 키프레임에 현재 자세와 고정 조건을 반영했습니다.'); return;
    }
    await run(async () => {
      const result = await api<{ keyframes: Keyframe[]; reports: Record<string, { status: string; max_position_error_mm: number }> }>('carry-edit', { project, index, qpos });
      checkpoint();
      // 자세가 바뀐 카드는 editFrame과 같이 자동 계획의 옛 TCP 목표와 파지 닫힘 정보를 버린다.
      const keyframes = result.keyframes.map((frame, i) => {
        const before = project.keyframes[i];
        if (i !== index && JSON.stringify(frame.qpos) === JSON.stringify(before.qpos) && frame.samples === undefined) return before;
        const { interaction: _planned, ...rest } = frame;
        const grasp = rest.grasp ? { ...rest.grasp, closure_qpos: undefined, object_signature: undefined, contact_points_world: undefined, hand_twist_deg: undefined } : undefined;
        return { ...rest, ...(grasp ? { grasp } : {}), ...(i === index ? patch : {}) };
      });
      setProject(p => p ? { ...p, keyframes } : p); invalidate(); setPoseDirty(false);
      let last = index;
      while (keyframes[last + 1]?.motion?.mode === 'delta') last++;
      const missed = Object.entries(result.reports).filter(([, report]) => report.status !== 'ok' && report.status !== 'approx');
      setMessage(`선택한 키프레임에 반영하고, 이어지는 ${index + 2}~${last + 1}번 카드를 새 끝점에서 이어지도록 옮겼습니다.${missed.length ? ` ${missed.map(([i, report]) => `${+i + 1}번`).join(', ')} 카드는 손 이동을 다 재현하지 못했습니다(오차 ${Math.max(...missed.map(([, report]) => report.max_position_error_mm)).toFixed(0)}mm).` : ''}`);
    });
  }
  function setMotion(next: CardMotion | undefined, message?: string) {
    editFrame(frameIndex, { motion: next });
    if (message) setMessage(message);
  }
  /** 카드가 끝나는 자세에서 두 손(TCP)의 가운데(mm 단위로 보여 줄 m 좌표). */
  async function recordedHandMiddle(index: number) {
    const frame = project?.keyframes[index];
    if (!frame) return null;
    const end = frame.samples ? frame.samples[frame.samples.length - 1] : frame.qpos;
    const result = await api<PoseState>('pose', { qpos: end });
    const left = result.handles.left_hand.position, right = result.handles.right_hand.position;
    return left.map((value, i) => (value + right[i]) / 2);
  }
  async function chooseMotionMode(mode: CardMotion['mode']) {
    if (!activeFrame) return;
    checkpoint();
    if (mode === 'recorded') { setMotion(undefined, '이 카드는 원본 그대로 재생합니다. 앞 카드가 남긴 차이는 이 카드 동안 서서히 없어집니다.'); return; }
    if (mode === 'delta') { setMotion({ mode: 'delta' }, '이 카드는 앞 카드가 끝난 자세에서 이어서 원본의 관절 각도 변화량만 더합니다.'); return; }
    const chosen = motionObjects.find(object => object.id === selectedObjectId) ?? motionObjects[0];
    const common = { mode: 'cartesian' as const, hands: ['left', 'right'] as ('left' | 'right')[], freedom: 'yaw_xyz' as const, waist: 'lock' as const, offset: { position: [0, 0, 0], yaw_deg: 0 } };
    if (chosen) {
      setMotion({ ...common, target: objectTarget(chosen) },
        `"${chosen.name}"을 기준 물체로 지정했습니다. 각 손은 이 물체의 양 끝 중 가까운 쪽에서 놓인 자리를 유지하므로, 다른 물체를 고르거나 물체를 옮기면 손이 그 물체의 양 끝으로 갑니다. 기준은 녹화 때 물체가 있던 자리여야 합니다.`);
      return;
    }
    await run(async () => {
      const point = await recordedHandMiddle(frameIndex);
      if (!point) return;
      setMotion({ ...common, target: { kind: 'point', point } }, '두 손 가운데의 좌표가 목표입니다. X/Y/Z를 바꾸면 이 카드의 끝 지점이 그 좌표로 이동합니다.');
    });
  }
  /** 포즈 카드: 물체 연결 켜기/끄기. 켜면 장면의 물체(선택한 것, 없으면 첫 번째)의 지금 위치가 기준이 된다. */
  async function togglePoseObjectLink(on: boolean) {
    if (!on) { await chooseMotionMode('recorded'); return; }
    if (!motionObjects.length) { setError('장면에 오브젝트가 없습니다. 먼저 물체를 추가하세요.'); return; }
    await chooseMotionMode('cartesian');
  }
  /** 물체 연결 목표: 지금 물체가 기준 위치이자 양끝 파지의 기준 물체(원본 자세를 만든 물체)가 된다. */
  function objectTarget(object: SceneObject): ObjectTarget {
    return { kind: 'object', object_id: object.id, reference_pose: { position: [...object.position], quaternion_xyzw: [...object.quaternion_xyzw] },
      ends: { reference: { shape: object.shape, size: [...object.size], position: [...object.position], quaternion_xyzw: [...object.quaternion_xyzw] } } };
  }
  /** 기준 물체가 없는(양끝 파지가 기본이 되기 전에 저장된) 카드는 기준 위치의 추적 물체를 기준 물체로 삼는다. 서버도 같은 규칙을 쓴다. */
  function targetEnds(target: ObjectTarget): NonNullable<ObjectTarget['ends']> | undefined {
    if (target.ends) return target.ends;
    const object = objects.find(item => item.id === target.object_id);
    return object && { reference: { shape: object.shape, size: [...object.size], ...target.reference_pose } };
  }
  function patchMotion(patch: Partial<CardMotion>) { if (activeFrame?.motion) setMotion({ ...activeFrame.motion, ...patch }); }
  async function chooseTargetKind(kind: 'object' | 'point') {
    const motion = activeFrame?.motion;
    if (!motion || motion.target?.kind === kind) return;
    checkpoint();
    if (kind === 'object') {
      const chosen = motionObjects.find(object => object.id === selectedObjectId) ?? motionObjects[0];
      if (!chosen) { setError('장면에 오브젝트가 없습니다. 먼저 물체를 추가하세요.'); return; }
      patchMotion({ target: objectTarget(chosen) });
    } else await run(async () => { const point = await recordedHandMiddle(frameIndex); if (point) patchMotion({ target: { kind: 'point', point } }); });
  }
  /** 양끝 파지: 켜면 녹화된 손 거리 대신 각 손의 TCP를 물체 끝점에서 안쪽·옆·위 오프셋(처음 0)만큼 떨어진 점에 둔다. */
  function setRowEndTcp(tcp: { inward_m: number; lateral_m: number; up_m: number } | undefined) {
    const target = activeFrame?.motion?.target;
    if (target?.kind !== 'object') return;
    const ends = targetEnds(target);
    if (!ends) { setError('기준으로 삼을 오브젝트가 장면에 없습니다.'); return; }
    checkpoint();
    const { tcp: _old, ...rest } = ends;
    patchMotion({ target: { ...target, ends: tcp ? { ...rest, tcp } : rest } });
  }
  /** 추적할 물체를 바꾼다. 기준 물체는 그대로라서 손이 새 물체의 양 끝으로 간다. */
  function chooseTrackedObject(id: string) {
    const target = activeFrame?.motion?.target;
    const object = objects.find(item => item.id === id);
    if (target?.kind !== 'object' || !object) return;
    const ends = targetEnds(target);
    checkpoint();
    patchMotion({ target: { ...target, object_id: object.id, reference_pose: { position: [...object.position], quaternion_xyzw: [...object.quaternion_xyzw] }, ...(ends ? { ends } : {}) } });
  }
  function recaptureMotionReference() {
    const target = activeFrame?.motion?.target;
    if (target?.kind !== 'object') return;
    const object = objects.find(item => item.id === target.object_id);
    if (!object) { setError('기준으로 삼을 오브젝트가 장면에 없습니다.'); return; }
    checkpoint();
    patchMotion({ target: { ...target, reference_pose: { position: [...object.position], quaternion_xyzw: [...object.quaternion_xyzw] } } });
    setMessage('물체의 지금 위치를 기준으로 다시 잡았습니다. 손이 가는 양 끝은 기준 물체로 정해지므로 바뀌지 않습니다.');
  }
  /** 월드 좌표 이동량(mm)을 "앞 3.5cm · 위 2.0cm"처럼 읽는다. 5mm 미만 성분은 생략한다. */
  function describeShift(mm: number[]) {
    const words: [string, string][] = [['앞', '뒤'], ['왼쪽', '오른쪽'], ['위', '아래']];
    const parts = mm.flatMap((value, axis) => Math.abs(value) < 5 ? [] : [`${words[axis][value > 0 ? 0 : 1]} ${(Math.abs(value) / 10).toFixed(1)}cm`]);
    return parts.length ? parts.join(' · ') : '거의 움직이지 않는 범위';
  }
  /** 한계에 걸린 이동을 어디까지 줄이면 되는지 안내하는 문장(없으면 null). */
  function motionAdvice(report: MotionReport | null) {
    if (!report?.safe_shift_mm || !report.requested_shift_mm) return null;
    const where = report.caused_by !== undefined && report.caused_by !== frameIndex ? `${report.caused_by + 1}번 카드의 목표를 ` : '';
    const blocked = report.blocks?.length ? `${report.blocks.map(item => item + 1).join(', ')}번 카드가 한계에 걸립니다. ` : '';
    return `${blocked}${where}같은 방향으로 ${describeShift(report.safe_shift_mm)}까지만 옮기면 안전합니다 (지금 ${describeShift(report.requested_shift_mm)}).`;
  }
  function motionStatusText(report: MotionReport | null) {
    if (!report) return '미리보기를 만들면 표시됩니다';
    const turn = report.hand_turn_deg !== undefined ? ` · 손 각도 ${report.hand_turn_deg.toFixed(0)}° 틀어짐` : '';
    const error = (report.max_position_error_mm !== undefined ? ` · ${report.max_position_error_mm.toFixed(1)}mm` : '') + turn;
    const perHand = report.hand_errors_mm ? ` · 오차(도달−목표) ${(['left_hand', 'right_hand'] as const).flatMap(hand => { const e = report.hand_errors_mm![hand]; return e ? [`${hand === 'left_hand' ? '좌' : '우'} x${e[0].toFixed(0)} y${e[1].toFixed(0)} z${e[2].toFixed(0)}`] : []; }).join(' / ')}mm` : '';
    const note = report.after_unreachable !== undefined ? ` · ${report.after_unreachable + 1}번 카드가 도달하지 못한 영향` : '';
    switch (report.status) {
      case 'unchanged': return report.mode === 'cartesian' ? '기준 위치 그대로' : `이어서 이동${report.offset_deg ? ` · 관절 차이 ${report.offset_deg.toFixed(0)}°` : ''}`;
      case 'ok': return report.mode === 'cartesian' ? `도달${error}${perHand}` : `이어서 이동${report.basis === 'position' ? ' · 위치 변화량' : report.basis === 'grip' ? ' · 양손 간격 유지' : ''}${error}${note}`;
      case 'approx': return `근사${error}${perHand}`;
      case 'partial': return `한도까지 이동 · 요청 ${(report.shift_mm ?? 0).toFixed(0)}mm 중 ${(report.reached_shift_mm ?? 0).toFixed(0)}mm${error}${perHand}${note}`;
      case 'unreachable': return `도달 불가${error}${perHand}`;
      case 'limit': return `관절 한계에 걸림 · ${(report.clamped_deg ?? 0).toFixed(0)}°`;
      case 'too_far': return `너무 멂 · ${(report.shift_mm ?? 0).toFixed(0)}mm (한도 ${(report.max_shift_mm ?? 0).toFixed(0)}mm)`;
      default: return '물체를 찾을 수 없음';
    }
  }
  /** 고른 두 카드 사이를 지금 재생되는 그대로(동작 방식 포함) 한 클립으로 합친다. */
  /** 이 카드를 바로 다음 키프레임에 합친다: 이 카드의 자세는 사라지고 다음 키프레임이 남으며, 두 구간의 시간은 더해진다. */
  function mergeActiveIntoNextPose() {
    if (!project || !activeFrame) return;
    try {
      const keyframes = mergeIntoNextPose(project.keyframes, frameIndex);
      const next = project.keyframes[frameIndex + 1];
      checkpoint();
      setProject({ ...project, keyframes }); setTransitionSelection([]); invalidate();
      setMessage(`"${activeFrame.name}"을 "${next.name}"에 합쳤습니다. "${next.name}" 키프레임이 남고, 걸리는 시간은 두 구간의 합입니다.`);
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
  }
  async function bakeSelectionToClip(range?: [number, number]) {
    if (!project || poseDirty) return;
    if (!range && transitionSelection.length !== 2) return;
    const [start, end] = range ?? [...transitionSelection].sort((a, b) => a - b);
    await run(async () => {
      const joined = await api<Keyframe>('keyframes-to-clip', { project: sceneProject(project), start, end, fps: 30 });
      // 합친 클립의 끝은 마지막 카드의 끝이므로 그 카드의 동작 방식을 물려받는다(없으면 첫 카드의 것).
      const motion = [project.keyframes[end].motion, project.keyframes[start].motion].find(item => item && item.mode !== 'recorded');
      // 위치 변화량은 포즈 카드 전용이라 클립은 관절 변화량으로 이어받는다.
      const { basis: _poseOnly, ...clipMotion } = motion ?? { mode: 'recorded' as const };
      const clip: Keyframe = motion ? { ...joined, motion: structuredClone(clipMotion) } : joined;
      checkpoint();
      const keyframes = [...project.keyframes.slice(0, start), clip, ...project.keyframes.slice(end + 1)];
      setProject({ ...project, keyframes });
      setTransitionSelection([]); setFrameIndex(start); invalidate();
      setMessage(`${start + 1}~${end + 1}번 카드를 녹화된 그대로 클립 하나로 합쳤습니다. 중간 카드의 동작 방식은 사라지고, 클립은 ${modeLabel[clip.motion?.mode ?? 'recorded']} 방식을 가집니다.`);
    });
  }
  // 마지막으로 본 재생 막대 시각: 편집으로 미리보기가 지워졌다 다시 계산되어도 같은 자리로 돌려놓는다
  const scrubTime = useRef(0);
  useEffect(() => { if (preview) scrubTime.current = preview.time[sample] ?? 0; }, [preview, sample]);
  function scrubbedClipSample() {
    if (!project) return null;
    return clipSampleIndexAtTime(project.keyframes, frameIndex, scrubTime.current, preview?.keyframe_times?.[frameIndex] ?? cardStarts[frameIndex]);
  }
  function splitActiveClip() {
    if (!project) return;
    try {
      const result = splitMotionClip(project.keyframes, frameIndex, splitSample);
      // 작업 중이던 쪽에 포커스를 둔다: 재생 막대가 자른 위치보다 앞에 있었으면 앞 클립, 아니면 뒤 클립(남은 부분)
      const at = scrubbedClipSample();
      const focus = at !== null && at < splitSample ? result.index : result.index + 1;
      checkpoint();
      setProject({ ...project, keyframes: result.keyframes }); setFrameIndex(focus); setTransitionSelection([]); invalidate();
      setMessage(`프레임 ${splitSample}에서 클립을 둘로 분할했습니다. 원본 동작과 총 재생 시간은 그대로입니다. ${focus === result.index ? '앞' : '뒤'} 클립을 선택했습니다.`);
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
  }
  const selectedLabGroup = (() => {
    if (!project || !selectedObjectId) return null;
    const active = labGroupRange(project.keyframes, frameIndex);
    if (active?.frames[0].lab_grasp?.target_id === selectedObjectId) return active;
    for (let index = project.keyframes.length - 1; index > 0; index--) {
      if (project.keyframes[index].lab_grasp?.target_id === selectedObjectId)
        return labGroupRange(project.keyframes, index);
    }
    return null;
  })();
  const selectedLabStatus = (() => {
    void labStorageRevision;
    if (!selectedLabGroup || !project || !tools || !selectedObjectId) return null;
    const link = selectedLabGroup.frames[0].lab_grasp!;
    const target = objects.find(object => object.id === link.target_id);
    const anchor = project.keyframes[selectedLabGroup.start - 1];
    if (!target || !anchor || tools.some((tool, index) => tool !== link.tools[index])) return '툴 또는 용기 확인 필요';
    const paths = readLabToolPaths(target.id, tools, graspCoefficients);
    if (!paths.left && !paths.right) return '저장 경로 없음';
    return labSourceSignature({ sourcePaths: paths }, target, anchor,
      labLateralOffsets(project, target.id, tools)) === link.source_signature ? '최신' : '경로 변경됨';
  })();
  const selectedLegacyLab = project ? legacyLabRange(project.keyframes, frameIndex) : null;
  function linkLegacyLab() {
    if (!project || !tools || !selectedLegacyLab || !selectedObjectId) return;
    const target = objects.find(object => object.id === selectedObjectId);
    const anchor = project.keyframes[selectedLegacyLab.start - 1];
    if (!target || !anchor) return;
    const paths = readLabToolPaths(target.id, tools, graspCoefficients);
    if (!paths.left && !paths.right) { setError('Cylinder Lab에 저장된 경로가 없습니다.'); return; }
    const sourceSignature = labSourceSignature({ sourcePaths: paths }, target, anchor,
      labLateralOffsets(project, target.id, tools));
    const groupId = crypto.randomUUID();
    checkpoint();
    const frames = project.keyframes.map((frame, index) => index < selectedLegacyLab.start || index > selectedLegacyLab.end ? frame : {
      ...frame, lab_grasp: { group_id: groupId, target_id: target.id, tools: [...tools] as [string, string],
        source_signature: sourceSignature, point_index: index - selectedLegacyLab.start,
        authored_signature: authoredSignature(frame) },
    });
    const nextProject = { ...project, keyframes: frames };
    current.current.project = nextProject; setProject(nextProject);
    setMessage(`${selectedLegacyLab.end - selectedLegacyLab.start + 1}개 기존 파지 키프레임을 ${target.name}에 연결했습니다.`);
  }
  const activeIsClip = !!activeFrame?.samples;
  const transitionHasClip = transitionSelection.length === 2 && !!project
    && project.keyframes.slice(Math.min(...transitionSelection), Math.max(...transitionSelection) + 1).some(f => f.samples);
  const clipTotal = project?.keyframes.filter(f => f.samples).length ?? 0;
  const duration = project?.keyframes.reduce((sum, f, i) => sum + (f.samples ? f.duration + (f.transition ?? 0) : i === 0 ? 0 : f.duration), 0) ?? 0;
  const activeClipLength = activeFrame?.samples?.length ?? 0;
  const selectedGroup = allNodes().find(node => node.children && nodeMembers(node).length === members.length && nodeMembers(node).every(key => members.includes(key)));

  const robotBlocked = !checkedFor ? '' : checkedFor.project !== project || checkedFor.objects !== objects || checkedFor.groups !== objectGroups || checkedFor.tools !== tools
    ? '점검한 뒤에 프로젝트·장면 물체·툴이 바뀌었습니다. 지금 장면으로 다시 점검하세요.' : '';
  return <div className={`editor ${robotOpen ? 'robot-open' : ''}`}>
    <header className="topbar">
      <div className="brand"><span className="brand-icon"><Move3d size={23}/></span><div>MOTION<span>CREATOR</span></div><select aria-label="로봇 모델" title="이 서버의 모델을 변경합니다. 현재 모션은 모델별로 자동 보관됩니다." disabled={disabled} value={modelId} onChange={event => void switchModel(event.target.value)}><option value="g1">G1 · 기본 손</option><option value="g1-tools">G1 그리퍼 · 툴 교환</option></select>{modelId === 'g1-tools' && tools && (['left', 'right'] as const).map((side, index) => <select key={side} className="tool-select" aria-label={side === 'left' ? '왼손 툴' : '오른손 툴'} title={`${side === 'left' ? '왼손' : '오른손'}에 끼울 툴을 고릅니다. 현재 모션은 유지되고 서버가 모델을 다시 만듭니다.`} disabled={disabled} value={tools[index]} onChange={event => void switchTool(side, event.target.value)}>{toolOptions.map(option => <option key={option.id} value={option.id}>{side === 'left' ? '왼 ' : '오른 '}{option.label}</option>)}</select>)}</div>
      <div className="project-title"><span className="status-dot"/>{project ? <div className="project-name-editor"><input aria-label="프로젝트 이름" title="이름을 바꾸고 Enter를 누르면 저장된 파일 이름도 바뀝니다(기존 파일 유지). 비우면 키프레임과 생성일자로 자동 이름을 만듭니다." placeholder={automaticProjectName(project)} maxLength={80} value={project.name} onFocus={e => { nameOnFocus.current = project.name; e.currentTarget.select(); }} onBlur={() => void renameSavedProject()} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); else if (e.key === 'Escape') { setProject({ ...project, name: nameOnFocus.current, name_mode: nameOnFocus.current.trim() ? 'manual' : 'auto' }); e.currentTarget.blur(); } }} onChange={e => setProject({ ...project, name: e.target.value, name_mode: e.target.value.trim() ? 'manual' : 'auto' })}/>{!project.name.trim() && <small>AUTO · {automaticProjectName(project)}</small>}</div> : '연결 중'}</div>
      <div className="top-actions">{!robotOpen && <RobotStatusChip onOpen={() => setRobotOpen(true)}/>}<button className="wbc-link" onClick={() => location.assign('/decoupled-wbc')}>Decoupled WBC</button><button className="wbc-link" onClick={() => location.assign('/controller')}>Task Controller</button><button className="wbc-link" onClick={() => location.assign('/cylinder-lab.html')}>Cylinder Lab</button><button className="wbc-link" onClick={() => location.assign('/live')} title="실제 G1: 하체는 Unitree 기본 컨트롤러, 상체는 저장한 모션(rt/arm_sdk)">Live Robot</button><button disabled={disabled} onClick={() => file.current?.click()}><FolderOpen size={16}/> 열기</button><button disabled={disabled || !project} onClick={() => addClipInput.current?.click()} title="지금 작업을 유지한 채 LeRobot data.json을 클립으로 목록 끝에 추가합니다"><Plus size={16}/> 클립 추가</button><div className="save-split" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setSaveMenuOpen(false); }} onKeyDown={event => { if (event.key === 'Escape') setSaveMenuOpen(false); }}><button className="primary save-main" disabled={disabled} onClick={() => exportProject()}><Save size={16}/> 모션 저장</button><button className="primary save-toggle" aria-label="저장 옵션" aria-haspopup="menu" aria-expanded={saveMenuOpen} disabled={disabled} onClick={() => setSaveMenuOpen(open => !open)}><ChevronDown size={14}/></button>{saveMenuOpen && <div className="save-dropdown" role="menu"><button role="menuitem" onClick={() => { setSaveMenuOpen(false); exportProject(true); }}><Save size={15}/><span><b>복사본으로 저장</b><small>새 프로젝트 ID와 폴더 생성</small></span></button><button role="menuitem" onClick={() => { setSaveMenuOpen(false); saveSceneFile(); }}><Boxes size={15}/><span><b>맵만 저장</b><small>씬 오브젝트·그룹만 별도 파일로</small></span></button></div>}</div></div>
      <input ref={addClipInput} type="file" accept=".json,.npz,.csv" multiple hidden onChange={e => { const sources = Array.from(e.target.files ?? []); if (sources.length) void run(() => addClipFiles(sources)); e.target.value = ''; }}/>
      <input ref={file} type="file" accept=".json,.npz,.csv" hidden onChange={e => { const source = e.target.files?.[0]; if (source) void run(() => openFile(source)); e.target.value = ''; }}/>
      <input ref={assetFile} type="file" accept=".blend,.glb" hidden onChange={e => { const source = e.target.files?.[0]; if (source) void run(() => importSceneAsset(source)); e.target.value = ''; }}/>
    </header>
    <aside className="left-panel panel">
      <div className="panel-heading"><span>BODY GROUPS</span><small>29자유도</small></div>
      <p className="hint">그룹 이름: 전체 선택 · 화살표: 펼치기/접기 · 하위 관절: 하나만 선택 · Shift: 추가/해제</p>
      <BodyControls pose={state} selected={members} pins={pins} anglePins={anglePins} expanded={expanded} disabled={disabled} onExpand={toggleExpanded} onSelect={selectBatch} onPin={togglePin} onAnglePin={toggleAnglePin}/>
      <div className="section-divider"/>
      <div className="panel-heading"><span>사용자 지정 프리셋</span><small>{members.length}개 선택</small></div>
      <select className="group-control" aria-label="그룹 프리셋 선택" value={groupId} disabled={disabled} onChange={e => {
        const id = e.target.value; setGroupId(id);
        const group = groups.find(g => g.id === id);
        if (group) { setGroupName(group.name); applySelection(group.members, group.members[0]); }
        else setGroupName('');
      }}><option value="">프리셋 선택</option>{groups.map(g => <option key={g.id} value={g.id}>{g.name} ({g.members.length})</option>)}</select>
      <input className="group-control" aria-label="그룹 프리셋 이름" placeholder="예: 양손, 골반과 무릎" maxLength={60} value={groupName} disabled={disabled} onChange={e => setGroupName(e.target.value)}/>
      <button className="wide" disabled={disabled || !groupName.trim()} onClick={() => void run(async () => {
        const group = await api<GroupPreset>('groups', { name: groupName, members });
        setGroups(await api<GroupPreset[]>('groups')); setGroupId(group.id);
        setMessage('그룹 프리셋을 디스크에 저장했습니다. 다음 실행에도 사용할 수 있습니다.');
      })}><Plus size={14}/>선택 부위를 새 프리셋으로 저장</button>
      <div className="group-actions"><button disabled={disabled || !groupId || !groupName.trim()} onClick={() => void run(async () => {
        await api<GroupPreset>('groups', { id: groupId, name: groupName, members });
        setGroups(await api<GroupPreset[]>('groups')); setMessage('프리셋 이름과 부위 구성을 수정했습니다.');
      })}>이름·구성 수정</button><button disabled={disabled || !groupId} onClick={() => void run(async () => {
        await api<null>(`groups/${encodeURIComponent(groupId)}`, undefined, 'DELETE');
        setGroups(await api<GroupPreset[]>('groups')); setGroupId(''); setGroupName(''); setMessage('그룹 프리셋을 삭제했습니다.');
      })}><Trash2 size={13}/>삭제</button></div>
      <p className="hint">일반 이동은 함께 옮기고, 대칭 이동은 짝을 벌리거나 모읍니다. 고정된 부위가 포함되면 먼저 해제하세요.</p>
      <div className="section-divider"/>
      <div className="panel-heading"><span>IK BEHAVIOR</span></div>
      <div className="segmented"><button disabled={disabled} className={mode === 'elastic' ? 'chosen' : ''} onClick={() => setMode('elastic')}>유연하게 따라오기</button><button disabled={disabled} className={mode === 'free' ? 'chosen' : ''} onClick={() => setMode('free')}>고정 부위만 유지</button></div>
      <label className="range-label">주변 부위 저항 <span>{resistance.toFixed(1)}</span><input aria-label="주변 부위 저항" type="range" min="0" max="5" step="0.1" value={resistance} disabled={disabled || mode === 'free'} onChange={e => setResistance(+e.target.value)}/></label>
      <p className="hint">먼 부위일수록 원래 위치를 더 유지합니다. 손의 위치를 정확히 유지하려면 손을 고정하세요.</p>
      <div className="pin-summary"><LockKeyhole size={14}/><span>노란 자물쇠: 위치 고정<br/>보라 회전 아이콘: 방향·관절각만 고정</span></div>
      <div className="section-divider"/>
      <div className="panel-heading"><span>자세 프리셋</span></div>
      <button className="wide" disabled={disabled} onClick={() => void run(async () => { checkpoint(); const init = await api<{state: PoseState}>('init'); applyState(init.state); setPins(feet); setAnglePins([]); setPoseDirty(true); invalidate(); })}><RotateCcw size={15}/> 기본 서기 자세</button>
      <button className="wide" disabled={disabled} onClick={() => void run(async () => { checkpoint(); changeGraspPickMode(null); const init = await api<{state: PoseState}>('attention'); applyState(init.state); setPins(feet); setAnglePins([]); setPoseDirty(true); setInfo(null); invalidate(); })}><PersonStanding size={15}/> 차렷 자세</button>
      <button className="wide" disabled={disabled} onClick={() => void run(async () => { checkpoint(); changeGraspPickMode(null); const init = await api<{state: PoseState}>('wbc-stand'); applyState(init.state); setPins(feet); setAnglePins([]); setPoseDirty(true); setInfo(null); invalidate(); })}><ArrowDownToLine size={15}/> WBC 서기 자세</button>
      <p className="hint">기본 서기는 로봇 초기 자세입니다. 차렷 자세는 관절을 좌우대칭으로 맞춰 두 팔을 몸 옆에 내린 자세입니다. WBC 서기 자세는 decoupled-WBC가 실제로 서 있는 하체(무릎을 더 굽힌 낮은 자세)이니, 제어기에서 실행할 모션은 이 자세에서 시작해 만드세요.</p>
      <button className="wide" disabled={disabled} onClick={() => void run(async () => {
        const init = await api<{ project: Project }>('init');
        await loadProject(init.project);
        applySelection(['pelvis'], 'pelvis');
        setFiles([]); setSavedChoice('');
        setMessage('새 모션을 만들었습니다. 기본 서기 자세의 키프레임 하나로 시작합니다.');
      })}><Plus size={15}/> 새로운 모션 만들기</button>
      <p className="hint">새 모션은 현재 타임라인을 기본 서기 자세 하나로 초기화합니다. 필요한 작업은 먼저 저장하세요.</p>
    </aside>
    <main className="viewport">
      <div ref={host} className="canvas-host"/>
      <div className="viewport-top"><div className="view-title"><span className="status-dot"/>POSE WORKSPACE<span>m · rad · Z-up</span></div><div className="view-buttons"><button className={showCollisions ? 'chosen' : ''} aria-pressed={showCollisions} title="원본 메시를 숨기고 Physics/WBC에서 사용하는 오브젝트 충돌체만 표시합니다." onClick={() => setShowCollisions(value => !value)}><Boxes size={13}/>콜리전</button>{(['perspective', 'front', 'side'] as const).map((v, i) => <button key={v} onClick={() => scene.current?.setView(v)}>{['자유', '정면', '측면'][i]}</button>)}</div></div>
      {policyJob && <div className="policy-progress">
        <span>{controllerLabel} 실시간 물리 재생 · 계산 <b>{Math.round(policyJob.progress * 100)}%</b></span>
        <progress aria-label="물리 계산 진행률" max={1} value={policyJob.progress}/>
        <button type="button" onClick={() => void cancelPolicy()}>중단하고 편집</button>
      </div>}
      {previewStreaming && <div className="policy-progress">
        <span>키프레임 구간 계산 · 실시간 재생</span>
        <progress aria-label="키프레임 계산 진행률" max={1} value={duration > 0 ? Math.min(1, (preview?.time.at(-1) ?? 0) / duration) : 0}/>
        <button type="button" onClick={cancelPreviewStream}>중단하고 편집</button>
      </div>}
      <div className="viewport-tools"><button title="되돌리기 (Ctrl+Z)" aria-label="되돌리기" aria-keyshortcuts="Control+Z Meta+Z" disabled={disabled || !historyCount} onClick={() => changeHistory(false)}><Undo2 size={17}/></button><button title="되돌리기 취소 (Ctrl+Shift+Z)" aria-label="되돌리기 취소" aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z" disabled={disabled || !futureCount} onClick={() => changeHistory(true)}><Redo2 size={17}/></button><span/><button className={showHandles ? 'chosen' : ''} title="조작 표시 켜기/끄기" onClick={() => setShowHandles(!showHandles)}><MousePointer2 size={17}/></button></div>
      <div className="scene-legend"><span><i className="cyan"/>이동 가능</span><span><i className="amber"/>위치 고정</span><span><i className="violet"/>각도 고정</span><span><i className="mint"/>선택</span></div>
      <div className="viewport-bottom"><span>드래그: 회전 · 휠: 확대 · 우클릭: 이동 · 로봇 W/E · 물체 W/E/R · F: 선택 보기</span><span className="axes"><b>X</b> 전방 <b>Y</b> 왼쪽 <b>Z</b> 위</span></div>
      {busy && !policyJob && !previewStreaming && <div className="busy-overlay"><span className="spinner"/>{message.startsWith('파지법 계산 중') ? message : '모션을 계산하고 있습니다…'}</div>}
    </main>
    <aside className="right-panel panel">
      <div className="panel-heading"><span>TRANSFORM</span><small>{transformMode === 'rotate' && (hinge || ankle) ? '관절축' : space.toUpperCase()}</small></div>
      <div className="segmented transform-modes"><button className={transformMode === 'translate' ? 'chosen' : ''} disabled={disabled} onClick={() => changeTransformMode('translate')} aria-keyshortcuts="W" title="이동 모드 (W)">이동 W</button><button className={transformMode === 'rotate' ? 'chosen' : ''} disabled={disabled || !rotationAllowed} onClick={() => changeTransformMode('rotate')} aria-keyshortcuts="E" title="회전 모드 (E)">회전 E</button></div>
      <div className="segmented"><button className={space === 'world' ? 'chosen' : ''} disabled={disabled || transformMode === 'rotate' && (!!hinge || ankle)} onClick={() => setSpace('world')} title="장면에 고정된 XYZ 축으로 드래그">월드 축</button><button className={space === 'local' ? 'chosen' : ''} disabled={disabled || transformMode === 'rotate' && (!!hinge || ankle)} onClick={() => setSpace('local')} title="선택 부위의 방향을 따라가는 XYZ 축으로 드래그">로컬 축</button></div>
      <div className="segmented"><button className={!mirror ? 'chosen' : ''} disabled={disabled} onClick={() => setMirror(false)}>일반 이동</button><button className={mirror ? 'chosen' : ''} disabled={disabled} aria-pressed={mirror} onClick={() => setMirror(true)}>대칭 이동</button></div>
      {mirror && <p className="hint">{mirrorActive ? transformMode === 'rotate' ? `기준: ${state?.handles[activeControl]?.label ?? activeControl}. 선택한 관절 링의 변화량을 반대쪽 관절축에 좌우 대칭으로 적용합니다.` : `기준: ${state?.handles[activeControl]?.label ?? activeControl}. 이쪽을 바깥/안쪽으로 드래그하면 반대쪽도 대칭 이동합니다. 로봇의 좌우 평면(팔은 몸통, 다리는 골반)을 기준으로 합니다.` : !mirrorAvailable ? '좌우 짝을 모두 선택하면 대칭 이동이 활성화됩니다.' : 'W 이동 또는 대칭 관절 2개를 선택한 E 회전 모드에서 사용할 수 있습니다.'}</p>}
      <h2>{selectedGroup?.label ?? (members.length > 1 ? `${members.length}개 부위` : state?.handles[selected].label ?? '골반')} <span>{selectionPinned ? '위치 고정 포함' : selectionAnglePinned ? '각도 고정 포함' : '이동 가능'}</span></h2>
      <p className="hint">{members.map(k => state?.handles[k].label ?? k).join(' · ')}</p>
      <button className="wide" disabled={!state} onClick={() => scene.current?.focusSelection()}>선택 부위 보기 <kbd>F</kbd></button>
      {partJoints && <div className="part-angle-editor">
        <label className="inspector-label">실제 관절각 <small>°</small></label>
        {partJoints.map(name => {
          const index = project?.joint_names.indexOf(name) ?? -1;
          const joint = state?.hinges[name];
          if (!joint || index < 0) return null;
          return <label className="part-angle-row" key={name}><span title={name}>{state?.handles[name]?.label ?? name}</span>
            <NumericInput aria-label={`${name} 부위 각도`}  step=".1" min={joint.limits[0]*180/Math.PI} max={joint.limits[1]*180/Math.PI} value={Number((jointDraft[index] ?? joint.angle*180/Math.PI).toFixed(2))} disabled={disabled} onChange={e => changeJoint(index, +e.target.value)}/>
            <small>{(joint.limits[0]*180/Math.PI).toFixed(0)} ~ {(joint.limits[1]*180/Math.PI).toFixed(0)}°</small>
          </label>;
        })}
        <button className="wide" disabled={disabled} onClick={() => applyPartAngles(partJoints)}>이 부위 관절각 적용</button>
        <p className="hint">실제 모터별 각도입니다. 입력 후 적용하면 고정 조건을 유지하며 IK를 풉니다.</p>
      </div>}
      {transformMode === 'rotate' && ankle ? <p className="hint">빨간 링은 roll, 초록 링은 pitch입니다. 각 링은 실제 모터 각도를 조정하며 yaw 회전은 제공하지 않습니다. 두 링은 발목 롤 위치에 표시합니다. 발 고정으로 인한 주변 관절 보정은 발생할 수 있습니다.</p> : transformMode === 'rotate' && hinge ? <>
        <label className="inspector-label">관절각 <small>°</small></label>
        <p className="hint">{hinge.joint_name}<br/>범위 {(hinge.limits[0]*180/Math.PI).toFixed(1)}° ~ {(hinge.limits[1]*180/Math.PI).toFixed(1)}°</p>
        <input className="joint-selected-slider" aria-label="선택 관절각 슬라이더" type="range" step=".1" min={hinge.limits[0]*180/Math.PI} max={hinge.limits[1]*180/Math.PI} value={jointDraft[hingeIndex] ?? hinge.angle*180/Math.PI} disabled={disabled} onChange={e => changeJoint(hingeIndex, +e.target.value)}/>
        <NumericInput className="group-control" aria-label="선택 관절각"  step="1" min={hinge.limits[0]*180/Math.PI} max={hinge.limits[1]*180/Math.PI} value={Number((jointDraft[hingeIndex] ?? hinge.angle*180/Math.PI).toFixed(2))} disabled={disabled} onChange={e => changeJoint(hingeIndex, +e.target.value)}/>
        <button className="wide" disabled={disabled} onClick={() => applyHingeAngle((jointDraft[hingeIndex] ?? hinge.angle*180/Math.PI)*Math.PI/180)}>관절각 적용</button>
        <div className="group-actions"><button disabled={disabled} onClick={() => applyHingeAngle(hinge.angle-Math.PI/180)}>−1°</button><button disabled={disabled} onClick={() => applyHingeAngle(hinge.angle+Math.PI/180)}>+1°</button></div>
        <p className="hint">링 하나로 실제 관절축의 각도를 조정합니다. 월드/로컬 선택과 관계없이 관절축을 사용하며, 손·발 등 고정 조건을 함께 유지합니다. 슬라이더·숫자 입력 후에는 관절각 적용을 누르세요.</p>
      </> : transformMode === 'rotate' ? <>
        {(HIP_HANDLES.includes(activeControl) || activeControl === 'waist') && <p className="hint">롤 관절 위치의 세 회전 링으로 {activeControl === 'waist' ? '상체' : '다리'} 방향을 조정합니다. IK가 yaw·pitch·roll을 함께 계산합니다. 아래 방향 값은 월드 Euler 각도이며 위 실제 관절각과 구분됩니다.</p>}
        <label className="inspector-label">기준 부위 방향 · 월드 XYZ <small>°</small></label>
        <div className="xyz">{rotation.map((v, i) => <label key={i}><span className={`axis-${i}`}>{'XYZ'[i]}</span><NumericInput aria-label={`목표 ${'XYZ'[i]} 회전각`}  step="1" value={Number(v.toFixed(2))} disabled={disabled || rotationBlocked} onChange={e => setRotation(t => t.map((n, j) => i === j ? +e.target.value : n))}/></label>)}</div>
        <button className="wide" disabled={disabled || rotationBlocked} onClick={() => numericRotate(rotation)}>회전 적용</button>
        <div className="nudge"><span>1° 회전</span>{['X', 'Y', 'Z'].map((a, i) => <div key={a}><button aria-label={`${a} 마이너스 1도`} disabled={disabled || rotationBlocked} onClick={() => nudgeRotation(i, -1)}>−</button><span>{a}</span><button aria-label={`${a} 플러스 1도`} disabled={disabled || rotationBlocked} onClick={() => nudgeRotation(i, 1)}>+</button></div>)}</div>
        <p className="hint">회전 링을 드래그하세요. 1° 버튼은 선택한 축 좌표계를 사용합니다. 숫자 입력은 월드 XYZ 순서의 Euler 각도입니다. {rotationBlocked ? '회전할 발 또는 그룹에 포함된 부위의 고정을 해제하세요.' : '단일 손·골반은 위치를 고정한 채 회전할 수 있습니다.'}</p>
      </> : <>
        <label className="inspector-label">{mirrorActive ? '대칭 기준 부위 목표 위치' : members.length > 1 ? '그룹 중심 목표 위치' : '목표 위치'} · 월드 XYZ <small>m</small></label>
        <div className="xyz">{target.map((v, i) => <label key={i}><span className={`axis-${i}`}>{'XYZ'[i]}</span><NumericInput aria-label={`목표 ${'XYZ'[i]} 위치`}  step="0.01" value={Number(v.toFixed(4))} disabled={disabled || selectionPinned} onChange={e => setTarget(t => t.map((n, j) => i === j ? +e.target.value : n))}/></label>)}</div>
        {space === 'local' && localFrame && <>
          <label className="inspector-label" title="같은 목표 위치를 선택 부위의 로컬 축(로컬 기즈모와 같은 방향)으로 읽은 값입니다. 원점은 월드 원점이라, 한 칸을 바꾸면 목표가 그 로컬 축 방향으로만 움직입니다.">{mirrorActive ? '대칭 기준 부위 목표 위치' : members.length > 1 ? '그룹 중심 목표 위치' : '목표 위치'} · 로컬 XYZ <small>m</small></label>
          <div className="xyz">{toLocalAxes(target, localFrame).map((v, i) => <label key={i}><span className={`axis-${i}`}>{'XYZ'[i]}</span><NumericInput aria-label={`목표 로컬 ${'XYZ'[i]} 위치`} step="0.01" value={Number(v.toFixed(4))} disabled={disabled || selectionPinned} onChange={e => setTarget(t => fromLocalAxes(toLocalAxes(t, localFrame).map((n, j) => i === j ? +e.target.value : n), localFrame))}/></label>)}</div>
        </>}
        <button className="wide" disabled={disabled || selectionPinned} onClick={() => numericMove(target)}>목표 위치 적용</button>
        <div className="nudge"><span>1cm · {space === 'local' && localFrame ? '로컬' : '월드'}</span>{['X', 'Y', 'Z'].map((a, i) => <div key={a}><button aria-label={`${a} 마이너스 1cm`} disabled={disabled || selectionPinned} onClick={() => numericMove(nudgedPosition(i, -.01))}>−</button><span>{a}</span><button aria-label={`${a} 플러스 1cm`} disabled={disabled || selectionPinned} onClick={() => numericMove(nudgedPosition(i, .01))}>+</button></div>)}</div>
        <p className="hint">고정된 부위를 이동하려면 자물쇠를 해제하세요. {isJointHandle(selected) && 'W는 관절 중심의 위치를 IK로 이동합니다. 해당 관절 자체의 회전각을 바꾸려면 E를 누르세요.'}</p>
      </>}
      <div className={`solver-card ${info && !info.converged ? 'warn' : ''}`}><div>{solving ? <span className="spinner"/> : info && !info.converged ? <AlertCircle size={15}/> : <Check size={15}/>} {solving ? 'IK 계산 중' : info ? info.converged ? '목표 도달' : '목표에 도달하지 못함' : '편집 준비 완료'}</div><dl><dt>{members.length > 1 ? '최대 목표 오차' : '목표 오차'}</dt><dd>{info ? info.target_error_mm.toFixed(2) : '—'} mm</dd><dt>고정 오차</dt><dd>{info ? info.pin_error_mm.toFixed(3) : '—'} mm</dd><dt>회전·각도 오차</dt><dd>{info?.angle_error_deg?.toFixed(3) ?? '—'}°</dd></dl></div>
      <div className="section-divider"/>
      <SceneObjectControls objects={objects} selectedId={selectedObjectId} mode={objectTransformMode} disabled={disabled} preventOverlap={preventObjectOverlap} surfaceSnap={objectSurfaceSnap} groundLock={objectGroundLock}
        graspPreview={graspPreview} graspCoefficients={graspCoefficients}
        canApplyGraspPreview={!!activeFrame && !activeIsClip} onApplyGraspPreview={applyCandidateGraspToFrame}
        onGraspCoefficientsChange={(side, field, value) => setGraspCoefficients(current => ({ ...current, [side]: { ...current[side], [field]: value } }))}
        onResetGraspCoefficients={() => setGraspCoefficients(DEFAULT_RAMEN_GRASP_COEFFICIENTS)}
        onToggleGraspPreview={id => setGraspPreviewId(current => current === id ? null : id)}
        groups={objectGroups} selectedGroupId={selectedObjectGroupId} onSelect={id => { setSelectedObjectGroupId(null); selectObject(id); }} onAdd={addObject} onRemove={removeObject} onChange={changeObject} onModeChange={changeObjectMode} onPlacementChange={changeObjectPlacement}
        onImport={() => assetFile.current?.click()} onParentChange={changeObjectParent} onCreateGroup={createObjectGroup} onSelectGroup={id => { setSelectedObjectGroupId(id); if (id) selectObject(null); }} onChangeGroup={changeObjectGroup} onRemoveGroup={removeObjectGroup}/>
      {modelId === 'g1-tools' && selectedObjectId && (() => {
        const object = objects.find(value => value.id === selectedObjectId);
        if (!object || !(isRamenBundle(object) || object.shape === 'cylinder' || object.collision_shape === 'cylinder')) return null;
        const path = labPathPreview?.targetId === object.id ? labPathPreview : null;
        const savedPaths = path?.sourcePaths ?? (tools ? readLabToolPaths(object.id, tools, graspCoefficients) : {});
        const lateralOffsets = tools ? labLateralOffsets(project, object.id, tools) : {};
        const dimensions = object.collision_size ?? object.size;
        const axial = dimensions.indexOf(Math.max(...dimensions));
        const radius = Math.max(...dimensions.filter((_, index) => index !== axial)) / 2;
        const mismatched = !!path && Object.values(path.sourcePaths).some(value => value?.source && Math.abs(value.source.radius - radius) > .005);
        return <div className="lab-grasp-apply">
          <div className="panel-heading"><span>CYLINDER LAB 파지법</span><small>{object.name}</small></div>
          {selectedLabGroup && <div className="lab-group-status">
            <strong>파지법 구간 · {selectedLabGroup.frames.length}개 키프레임</strong>
            <span>{selectedLabGroup.valid ? selectedLabStatus : '구간 순서 변경됨'} · 직접 수정 {selectedLabGroup.edited.length}개</span>
            <button className="wide" disabled={disabled || poseDirty || !selectedLabGroup.valid || selectedLabStatus === '툴 또는 용기 확인 필요' || selectedLabStatus === '저장 경로 없음'}
              onClick={() => void updateLabGraspKeyframes(selectedLabGroup.start)}>Cylinder Lab 경로로 업데이트</button>
            <p className="hint">계산 후 교체 내용을 확인합니다. 이 구간 밖의 키프레임은 유지됩니다.</p>
          </div>}
          {!selectedLabGroup && selectedLegacyLab && <div className="lab-group-status">
            <strong>기존 파지법 키프레임 {selectedLegacyLab.end - selectedLegacyLab.start + 1}개</strong>
            <p className="hint">현재 키프레임들을 선택한 용기에 연결하면 다음부터 경로 변경을 확인하고 일괄 업데이트할 수 있습니다.</p>
            <button className="wide" disabled={disabled} onClick={linkLegacyLab}>기존 구간 연결</button>
          </div>}
          {pendingLabReplacement?.targetId === object.id && <div className="lab-group-review">
            <strong>교체 전 확인 · {pendingLabReplacement.end - pendingLabReplacement.start + 1}개 → {pendingLabReplacement.frames.length}개</strong>
            {pendingLabReplacement.editedNames.length > 0 && <p className="hint lab-grasp-warning">직접 수정한 {pendingLabReplacement.editedNames.length}개도 교체됩니다: {pendingLabReplacement.editedNames.join(', ')}</p>}
            <div className="lab-grasp-actions"><button className="primary" disabled={disabled || project !== pendingLabReplacement.project} onClick={applyLabReplacement}>이 구간 교체</button>
              <button onClick={() => { setPendingLabReplacement(null); setLabPathPreview(null); }}>취소</button></div>
          </div>}
          <button className="wide" disabled={disabled || !activeFrame || activeIsClip} onClick={() => void prepareLabGrasp()}>파지법 적용</button>
          {(['left', 'right'] as HandSide[]).some(side => !!savedPaths[side]) && <div className="lab-lateral-adjustment">
            <strong>메인 편집기 lateral 보정</strong>
            <p className="hint">Cylinder Lab의 기본 파지법을 기준으로 손별 경로 전체를 Y축으로 평행이동합니다. XZ 경로와 툴 자세는 유지됩니다.</p>
            {(['left', 'right'] as HandSide[]).map(side => {
              const base = savedPaths[side]?.coefficients.lateral;
              if (base === undefined) return null;
              const adjusted = Number((base + (lateralOffsets[side] ?? 0)).toFixed(6));
              return <div className="lab-lateral-row" key={side}>
                <label htmlFor={`lab-lateral-${side}`}><span className={side}>{side === 'left' ? '왼손' : '오른손'}</span> lateral</label>
                <NumericInput id={`lab-lateral-${side}`} aria-label={`${side === 'left' ? '왼손' : '오른손'} 파지법 lateral`}
                  value={adjusted} min={0} step=".005" disabled={disabled} onChange={event => changeLabLateral(object.id, side, base, +event.target.value)}/>
                <span>m</span><button disabled={disabled || !lateralOffsets[side]} onClick={() => changeLabLateral(object.id, side, base, base)}>기본값</button>
              </div>;
            })}
            <p className="hint">보정 후 파지법 적용으로 3D 경로를 확인하세요. 이미 만든 키프레임은 위의 업데이트 버튼으로 다시 계산합니다.</p>
          </div>}
          {path && <>
            <p className="hint">장면의 초록 경로는 왼손, 주황 경로는 오른손입니다. 큰 점과 자세 축이 파지점입니다.</p>
            {(['left', 'right'] as HandSide[]).map(side => {
              const track = path.tracks[side];
              const toolId = tools?.[side === 'left' ? 0 : 1] ?? '';
              return <div className={`lab-grasp-side ${side}`} key={side} title={path.sourcePaths[side]?.projectName ?? ''}><strong>{side === 'left' ? '왼손' : '오른손'} · {toolOptions.find(option => option.id === toolId)?.label ?? toolId}</strong>
                <span>{track ? `TCP 샘플 ${track.points.length}개 · 그린 점 ${track.drawnCount}개` : '저장 경로 없음 · 현재 자세에 가깝게 유지'}</span></div>;
            })}
            {mismatched && <p className="hint lab-grasp-warning">Cylinder Lab에서 조절한 용기 지름이 메인 장면과 다릅니다. 메인 장면의 지름을 맞춘 뒤 다시 미리보세요.</p>}
            <p className="hint">양손의 그린 점을 순서대로 묶어 선택 키프레임 다음에 {path.times.length}개 생성 · 구간별 보간 시간을 키프레임 이동시간으로 적용 · 최대 100개</p>
            <p className="hint">파지법 IK는 화면의 IK Behavior 설정과 별개로 계산합니다. 손 목표를 맞추는 동안 현재 골반 위치와 고정된 부위를 유지합니다.</p>
            <p className="hint">키프레임 생성은 TCP 도달 여부를 확인합니다. 툴과 용기의 물리 충돌은 생성 후 Physics 재생으로 확인하세요.</p>
            <div className="lab-grasp-actions"><button className="primary" disabled={disabled || poseDirty || mismatched || !activeFrame || activeIsClip} onClick={() => void generateLabGraspKeyframes()}>키프레임 생성</button>
              <button disabled={disabled} onClick={() => setLabPathPreview(null)}>미리보기 닫기</button></div>
          </>}
        </div>;
      })()}
      <div className="section-divider"/>
      {state?.handles.left_hand?.label.includes('TCP')
        ? <><p className="hint">G1 그리퍼 작업점: 왼손 주걱 · 오른손 받침. 타임라인에서 자세를 하나씩 추가하고 양쪽 TCP를 조정한 뒤 선택 프레임에 반영하세요. Physics에서는 두 도구의 실제 충돌 형상·질량·관성이 적용됩니다.</p>
          {SHOW_AUTOMATIC_RAMEN_SEQUENCE && <RamenSequenceControls objects={objects} selectedObjectId={selectedObjectId} keyframes={project?.keyframes ?? []} disabled={disabled || activeIsClip}
            onTargetChange={id => { setSelectedObjectGroupId(null); selectObject(id); }} onGenerate={(id, settings) => void generateRamenSequence(id, settings)}
            onSelectFrame={index => void chooseFrame(index)} onRun={index => { setPhysicsEnabled(true); void play(true, index); }}/>}</>
        : <GraspControls objects={objects} selectedObjectId={selectedObjectId} grasp={activeFrame?.grasp} disabled={disabled || activeIsClip} pickMode={graspPickMode} onChange={changeGrasp} onPickMode={graspPickMode => void changeGraspPickMode(graspPickMode)} onFit={() => void fitGrasp()} onEditGhost={editGraspGhost} onValidate={() => void validateGraspPhysics()}/>}
      <div className="section-divider"/>
      <div className="panel-heading"><span>최근 저장한 프로젝트</span><small>서버 저장</small></div>
      <select aria-label="최근 저장한 프로젝트" value={savedChoice} onChange={e => setSavedChoice(e.target.value)}><option value="">최근 프로젝트 선택</option>{saved.map(name => <option key={name} value={name}>{name}</option>)}</select>
      <button className="wide" disabled={disabled || !savedChoice} onClick={() => void run(async () => loadProject(await api<Project>(`project/${fileApiPath(savedChoice)}`)))}>최근 프로젝트 열기</button>
      <p className="hint"><code>motions/</code>에 저장한 편집 프로젝트를 빠르게 다시 엽니다. 상단 열기는 컴퓨터의 임의 파일을 선택합니다.</p>
      <label className="hint"><input type="checkbox" checked={exportProto} disabled={disabled} onChange={e => setExportProto(e.target.checked)}/> 저장 시 ProtoMotions .motion / .pt 추가</label>
      <p className="hint">유지 자세는 같은 키프레임을 복제해 시간을 지정하세요.</p>
      {files.map(name => <a className="download" key={name} href={`/api/files/${fileApiPath(name)}`} download><Download size={13}/>{name}</a>)}
      <details className="joint-editor"><summary>29개 관절각 정밀 조정 {jointDirty ? '· 변경 대기' : ''}</summary>
        <p className="hint">슬라이더나 숫자를 수정한 뒤 적용하세요. 발 등 고정 조건을 유지하며 IK를 풉니다. 달성하지 못한 각도는 오차로 표시합니다.</p>
        <button className="wide primary" disabled={disabled || !jointDirty} onClick={applyJointDraft}>변경 각도 적용</button>
        <button className="wide" disabled={disabled || !jointDirty} onClick={() => setJointDraft(state!.qpos.slice(7).map(v => v*180/Math.PI))}>입력 취소</button>
        <div>{project?.joint_names.map((name, i) => <div className="joint-angle-row" key={name}><label htmlFor={`joint-${i}`}>{name.replace('_joint', '')}</label><div><input aria-label={`${name} 슬라이더`} type="range" min={limits[i]?.[0] ?? -180} max={limits[i]?.[1] ?? 180} step=".1" value={jointDraft[i] ?? 0} disabled={disabled} onChange={e => changeJoint(i, +e.target.value)}/><NumericInput id={`joint-${i}`} aria-label={`${name} 각도`}  step=".1" min={limits[i]?.[0]} max={limits[i]?.[1]} value={Number((jointDraft[i] ?? 0).toFixed(2))} disabled={disabled} onChange={e => changeJoint(i, +e.target.value)}/><span>°</span></div></div>)}</div>
      </details>
    </aside>
    <section className="timeline">
      <div className="timeline-header"><div className="timeline-title">KEYFRAMES <span>{clipTotal ? `${clipTotal} clip · ${(project?.keyframes.length ?? 0) - clipTotal} poses` : `${project?.keyframes.length ?? 0} poses`} · {duration.toFixed(1)}s</span></div>{stage && project?.keyframes[stage.index] && <span className="stage-now" role="status"><b>{String(stage.index + 1).padStart(2, '0')} / {String(project.keyframes.length).padStart(2, '0')}</b> {project.keyframes[stage.index].name}<small>{project.keyframes[stage.index].samples ? '클립' : '포즈'} · {playing ? '재생 중' : '일시정지'} · 전체 {Math.round(stage.overall * 100)}%</small><i><u style={{ width: `${stage.overall * 100}%` }}/></i></span>}<div className="timeline-actions"><button className={`replay-open ${robotOpen ? 'chosen' : ''}`} disabled={!project} aria-pressed={robotOpen} onClick={() => setRobotOpen(open => !open)} title="오른쪽에 로봇 패널을 열고 닫습니다: Unitree 기본 정책으로 실제 로봇(또는 가상 로봇)에 지금 장면 그대로 재생합니다">로봇 재생</button>{poseDirty && <span className="dirty-tag">편집 자세 · 반영 필요</span>}<small className="transition-hint">{transitionSelection.length === 2 ? `${Math.min(...transitionSelection) + 1}–${Math.max(...transitionSelection) + 1}번 선택` : transitionSelection.length === 1 ? '두 번째 프레임 선택' : '프레임 2개 선택'}</small><button disabled={disabled || transitionSelection.length !== 2 || poseDirty || transitionHasClip} onClick={createSmoothTransition} title="두 키프레임 사이에 편집 가능한 중간 자세를 생성합니다. 파지·도구 상호작용이 포함된 구간과 장애물 회피는 아직 지원하지 않습니다.">중간 키프레임 생성</button><button disabled={disabled || transitionSelection.length !== 2 || poseDirty} onClick={() => void bakeSelectionToClip()} title="고른 두 카드와 그 사이 카드를 녹화된 그대로 클립 하나로 합칩니다. 합친 클립은 마지막 카드의 동작 방식을 가집니다.">선택 구간을 클립으로</button><button disabled={disabled || !activeFrame || activeIsClip} onClick={() => void applyPoseToFrame()}>선택 프레임에 반영</button><button disabled={disabled} className="primary" onClick={addFrame} title={activeIsClip ? "선택한 클립 바로 뒤에, 클립의 마지막 자세에서 시작하는 편집용 자세를 추가합니다." : "선택한 키프레임을 복제해 바로 다음에 추가합니다."}><Plus size={14}/>자세 추가</button></div></div>
      <div className="keyframe-toolbar">{activeFrame ? <>
        <div className="tool-group"><span className={`tool-chip ${activeIsClip ? 'clip' : ''}`}>{frameIndex + 1}번 · {activeIsClip ? `클립 ${activeClipLength}프레임` : '포즈'}</span>
          <input aria-label="키프레임 이름" className="tool-name" value={activeFrame.name} disabled={disabled} onChange={e => editFrame(frameIndex, { name: e.target.value })}/></div>
        {activeIsClip && <div className="tool-group"><label title="클립 안의 원본 동작을 재생하는 시간입니다.">재생 시간 <NumericInput aria-label="클립 재생 시간" min={1/120} max={600} step=".01" disabled={disabled} value={activeFrame.duration} onChange={e => editFrame(frameIndex, { duration: Math.max(1/120, Math.min(600, +e.target.value || 1/120)) })}/>s</label></div>}
        <div className="tool-group"><label title="이 카드의 자세에서 다음 카드의 자세까지 이동하는 시간입니다. 다음이 클립이면 비워 두어 자동 계산할 수 있습니다.">다음 카드까지 {(() => {
          const out = outgoingSeconds(project!.keyframes, frameIndex);
          if (out.kind === 'none') return <input className="tool-readonly" aria-label="다음 카드까지 시간" disabled value="—" readOnly title="마지막 카드는 다음 카드가 없습니다."/>;
          if (out.kind === 'joined') return <input className="tool-readonly" aria-label="다음 카드까지 시간" disabled value="이어짐" readOnly title="다음 클립이 이 클립의 끝에서 바로 이어지므로 이동 시간이 필요 없습니다."/>;
          if (out.kind === 'clip') return <input type="number" aria-label="다음 카드까지 시간" min=".1" max="60" step=".1" placeholder="자동" disabled={disabled} value={out.seconds ?? ''}
            onChange={e => editFrame(frameIndex + 1, { transition: e.target.value === '' ? undefined : Math.max(.1, Math.min(60, +e.target.value || .1)) })}/>;
          return <NumericInput aria-label="다음 카드까지 시간" min={.1} max={60} step=".1" disabled={disabled} value={out.seconds} onChange={e => editFrame(frameIndex + 1, { duration: Math.max(.1, Math.min(60, +e.target.value || .1)) })}/>;
        })()}{outgoingSeconds(project!.keyframes, frameIndex).kind === 'pose' || outgoingSeconds(project!.keyframes, frameIndex).kind === 'clip' ? 's' : ''}</label></div>
        <div className="tool-group">
          <button title="이전으로 이동" disabled={disabled || frameIndex === 0} onClick={() => { setProject({ ...project!, keyframes: moveKeyframe(project!.keyframes, frameIndex, -1) }); setFrameIndex(frameIndex-1); setTransitionSelection([]); invalidate(); }}><ChevronLeft size={14}/></button><button title="다음으로 이동" disabled={disabled || frameIndex === project!.keyframes.length-1} onClick={() => { setProject({ ...project!, keyframes: moveKeyframe(project!.keyframes, frameIndex, 1) }); setFrameIndex(frameIndex+1); setTransitionSelection([]); invalidate(); }}><ChevronRight size={14}/></button><button title="키프레임 삭제" disabled={disabled || project!.keyframes.length < 2} onClick={() => { setProject({ ...project!, keyframes: removeKeyframe(project!.keyframes, frameIndex) }); setFrameIndex(Math.max(0, frameIndex-1)); setTransitionSelection([]); invalidate(); }}><Trash2 size={14}/></button></div>
        {activeIsClip ? <>
          <div className="tool-group"><span className="tool-label">분할</span>
            <NumericInput aria-label="클립 분할 프레임" min={1} max={Math.max(1, activeClipLength - 2)} step="1" disabled={disabled || activeClipLength < 3} value={splitSample} onChange={e => setSplitSample(Math.round(+e.target.value) || 1)}/>
            <span className="tool-unit">/ {Math.max(0, activeClipLength - 2)}</span>
            <button disabled={disabled || activeClipLength < 3} onClick={splitActiveClip} title="선택한 클립을 이 프레임에서 둘로 나눕니다. 기본값은 클립의 가운데이며, 슬라이더를 이 클립 안에 두거나 숫자를 바꿔 원하는 위치에서 자를 수 있습니다."><Scissors size={14}/> 분할</button>
            <button disabled={disabled || frameIndex >= project!.keyframes.length - 1 || !project!.keyframes[frameIndex + 1]?.samples} onClick={mergeActiveClip} title="선택한 클립을 바로 다음 클립과 하나로 합칩니다.">다음과 합치기</button></div>
          <div className="tool-group"><span className="tool-label">변환</span>
            <NumericInput aria-label="변환할 키프레임 수" min={2} max={Math.max(2, activeClipLength)} step="1" disabled={disabled} value={convertCount} onChange={e => setConvertCount(Math.max(2, Math.round(+e.target.value) || 2))}/>
            <span className="tool-unit">개</span>
            <label className="dynamic-hand" title="켜면 변환한 키프레임을 부드러운 연결로 만듭니다. 키프레임마다 멈추지 않고 속도가 이어져 원래 클립처럼 자연스럽게 움직입니다."><input type="checkbox" disabled={disabled} checked={convertSmooth} onChange={e => setConvertSmooth(e.target.checked)}/>부드러운 연결</label>
            <button className="primary" disabled={disabled || activeClipLength < 2} onClick={convertActiveClip} title="이 클립을 편집 가능한 포즈 키프레임으로 바꿉니다(시작·끝 포함 균등 간격, 양발 고정). 앞뒤 클립과는 자동으로 이어집니다.">키프레임으로 변환</button>
            <button disabled={disabled || poseDirty || !project!.keyframes[frameIndex + 1] || !!project!.keyframes[frameIndex + 1]?.samples} onClick={mergeActiveIntoNextPose}
              title="이 클립을 없애고 바로 다음 키프레임 하나로 합칩니다. 남는 것은 다음 키프레임이고, 걸리는 시간은 클립 재생 시간을 포함한 합입니다. 클립의 중간 동작은 사라집니다.">다음 키프레임에 합치기</button></div>
        </> : <div className="tool-group"><span className="tool-label">연결</span>
          <label className="dynamic-hand" title="켜면 이 카드를 지날 때 멈추지 않고 앞뒤 카드와 속도가 이어집니다(부드러운 연결). 끄면 이 카드에서 잠깐 멈춥니다. 첫 카드와 마지막 카드에서는 효과가 없습니다."><input type="checkbox" disabled={disabled} checked={!!activeFrame.generated_transition} onChange={e => { checkpoint(); editFrame(frameIndex, { generated_transition: e.target.checked ? true : undefined }); invalidate(); }}/>부드러운 연결</label>
        </div>}
        {!activeIsClip && <div className="tool-group"><span className="tool-label">이동</span>
          <select aria-label="이동 방식" disabled={disabled || frameIndex === 0} value={activeFrame.path?.kind ?? 'joint'}
            title="앞 카드에서 이 카드로 오는 동안 손이 지나는 길입니다. 관절(MoveJ): 관절각을 섞음 · 직선(MoveL): 두 손(TCP)이 직선으로 · 원호(MoveC): 두 손이 지정한 중간점을 지나는 호로. 시작과 끝 자세는 그대로입니다."
            onChange={e => {
              checkpoint();
              const kind = e.target.value as 'joint' | 'linear' | 'circular';
              editFrame(frameIndex, { path: kind === 'joint' ? undefined : kind === 'circular' ? { kind, via_offset: activeFrame.path?.via_offset ?? [0, 0, .05] } : { kind } });
              invalidate();
            }}>
            <option value="joint">관절 (MoveJ)</option><option value="linear">직선 (MoveL)</option><option value="circular">원호 (MoveC)</option></select>
          {activeFrame.path?.kind === 'circular' && <span className="dynamic-offset" title="호의 휨: 시작과 끝의 한가운데에서 이만큼(월드 좌표, mm) 옮긴 점을 두 손이 지납니다.">
            {(['X', 'Y', 'Z'] as const).map((axis, i) => <label key={axis}>{`+${axis}`}
              <NumericInput aria-label={`호의 휨 ${axis} (mm)`} step="5" disabled={disabled} value={Math.round((activeFrame.path!.via_offset?.[i] ?? 0) * 1000)}
                onChange={e => { const via = [...(activeFrame.path!.via_offset ?? [0, 0, 0])]; via[i] = (+e.target.value || 0) / 1000; editFrame(frameIndex, { path: { kind: 'circular', via_offset: via } }); invalidate(); }}/></label>)}
          </span>}
        </div>}
        {!activeIsClip && <div className="tool-group"><span className="tool-label">변환</span>
          <NumericInput aria-label="클립으로 합칠 카드 수" min={2} max={Math.max(2, project!.keyframes.length - frameIndex)} step="1" disabled={disabled} value={Math.min(bakeCount, Math.max(2, project!.keyframes.length - frameIndex))}
            onChange={e => setBakeCount(Math.max(2, Math.round(+e.target.value) || 2))}/>
          <span className="tool-unit">장</span>
          <button className="primary" disabled={disabled || poseDirty || frameIndex >= project!.keyframes.length - 1}
            onClick={() => void bakeSelectionToClip([frameIndex, Math.min(project!.keyframes.length - 1, frameIndex + Math.max(2, bakeCount) - 1)])}
            title="이 카드부터 지정한 장수만큼(이 카드 포함)을 녹화된 그대로 클립 하나로 합칩니다. 합친 클립은 마지막 카드의 동작 방식을 가집니다. 구간이 멀면 헤더의 '선택 구간을 클립으로'로 두 카드를 직접 고를 수도 있습니다.">클립으로 변환</button>
          <button disabled={disabled || poseDirty || !project!.keyframes[frameIndex + 1] || !!project!.keyframes[frameIndex + 1]?.samples} onClick={mergeActiveIntoNextPose}
            title="이 카드의 자세를 없애고 바로 다음 키프레임 하나로 합칩니다. 남는 것은 다음 키프레임(이름·동작 방식 포함)이고, 걸리는 시간은 두 구간의 합입니다.">다음 키프레임에 합치기</button>
        </div>}
        <div className="tool-group dynamic-group"><span className="tool-label">{activeIsClip ? '동작 방식' : '물체'}</span>
          {!activeIsClip && <label className="dynamic-hand" title="켜면 이 카드의 끝 자세가 선택한 물체를 따라갑니다. 물체를 옮기면 손이 그 위치로 갑니다."><input type="checkbox" disabled={disabled} checked={activeFrame.motion?.mode === 'cartesian'} onChange={e => void togglePoseObjectLink(e.target.checked)}/>물체 연결</label>}
          {!activeIsClip && frameIndex > 0 && activeFrame.motion?.mode !== 'cartesian' && <label className="dynamic-hand" title="켜면 이 카드가 바로 앞 카드가 끝난 자세에서 이어서 움직입니다(앞에서 물체 연결로 옮겨진 만큼 유지). 끄면 이 카드 동안 원래 녹화 자세로 서서히 돌아옵니다. 중간 카드 하나를 끄면 그 뒤 카드는 켜 두어도 이어받을 차이가 없습니다."><input type="checkbox" disabled={disabled} checked={activeFrame.motion?.mode === 'delta'} onChange={e => { checkpoint(); setMotion(e.target.checked ? { mode: 'delta', ...(activeFrame.motion?.basis ? { basis: activeFrame.motion.basis } : {}) } : undefined, e.target.checked ? '이 카드는 앞 카드가 끝난 자세에서 이어서 움직입니다.' : '이 카드는 앞 카드의 이동 차이를 서서히 없애며 원래 자세로 돌아옵니다.'); }}/>앞 카드 이동 연결</label>}
          {!activeIsClip && frameIndex > 0 && activeFrame.motion?.mode === 'delta' && <select aria-label="이어받는 변화량" disabled={disabled} value={activeFrame.motion.basis ?? 'joint'}
            onChange={e => { checkpoint(); const basis = e.target.value as 'joint' | 'position' | 'grip'; setMotion(basis === 'joint' ? { mode: 'delta' } : { mode: 'delta', basis }, basis === 'joint' ? '녹화의 관절 각도 변화량을 그대로 더합니다. 앞 카드 자세가 바뀌면 손이 움직이는 방향과 거리는 달라질 수 있습니다.' : basis === 'position' ? '녹화에서 손(TCP)이 움직이고 돈 양을 앞 카드가 실제로 끝난 손 위치에서 그대로 재현합니다. 관절은 IK로 다시 풉니다.' : '두 손이 녹화된 양손 가운데점의 이동량만큼 함께 움직여, 앞 카드가 끝난 양손 간격을 그대로 유지합니다(물체를 든 채 옮길 때). 직선 이동과 함께 쓰면 두 손이 평행한 직선으로 움직입니다.'); }}
            title="관절 각도 변화량: 녹화된 관절 회전을 그대로 더합니다(손목 회전처럼 관절 동작이 중요한 카드). 위치 변화량: 녹화에서 손(TCP)이 움직이고 돈 양을 그대로 재현합니다(양손을 모으는 것처럼 손 이동이 중요한 카드). 양손 간격 유지: 두 손이 함께 같은 만큼 움직여 간격이 그대로입니다(물체를 든 채 옮기는 카드).">
            <option value="joint">관절 각도 변화량</option><option value="position">위치 변화량</option><option value="grip">양손 간격 유지</option></select>}
          {activeIsClip && <select aria-label="카드 동작 방식" disabled={disabled} value={activeFrame.motion?.mode ?? 'recorded'} onChange={e => void chooseMotionMode(e.target.value as CardMotion['mode'])}
            title="원본 그대로: 녹화된 동작 · 관절 델타: 앞 카드가 끝난 자세에서 이어서 원본의 관절 변화만 더함(동작이 중요한 카드) · 카테시안: 끝 지점을 목표 위치로 옮김(도착 위치가 중요한 카드)">
            {(['recorded', 'delta', 'cartesian'] as const).map(mode => <option key={mode} value={mode}>{modeLabel[mode]}</option>)}</select>}
          {activeFrame.motion?.mode === 'cartesian' && activeFrame.motion.target && <>
            {(activeIsClip || activeFrame.motion.target.kind === 'point') && <select aria-label="목표 종류" disabled={disabled} value={activeFrame.motion.target.kind} onChange={e => void chooseTargetKind(e.target.value as 'object' | 'point')}>
              <option value="object">물체 추적</option><option value="point">좌표</option></select>}
            {activeFrame.motion.target.kind === 'object' ? <>
              <select aria-label="추적할 물체" disabled={disabled} value={activeFrame.motion.target.object_id} onChange={e => chooseTrackedObject(e.target.value)}>
                {activeFrame.motion.target.object_id && !motionObjects.some(object => object.id === (activeFrame.motion!.target as { object_id: string }).object_id) ? <option value={activeFrame.motion.target.object_id}>(없는 물체)</option> : null}
                {motionObjects.map(object => <option key={object.id} value={object.id}>{object.name}</option>)}</select>
              <button disabled={disabled} onClick={recaptureMotionReference} title="물체의 지금 위치를 녹화 때의 위치로 다시 저장합니다. 이 위치에서는 카드가 원본 그대로입니다.">기준 다시 잡기</button>
              <label className="dynamic-hand" title="물체 연결은 항상 양 끝을 기준으로 손을 놓습니다. 끄면 각 손이 녹화 때 기준 물체의 끝에서 놓인 거리를 새 물체의 같은 쪽 끝에 유지합니다. 켜면 그 대신 각 손의 TCP를 물체 끝점(중심축 끝)에서 아래 안쪽·옆·위 값만큼 떨어진 점에 둡니다(양손 같은 값).">
                <input type="checkbox" disabled={disabled} checked={!!activeFrame.motion.target.ends?.tcp} onChange={e => setRowEndTcp(e.target.checked ? { inward_m: 0, lateral_m: 0, up_m: 0 } : undefined)}/>양끝 파지</label>
              <label className="dynamic-hand" title="켜면 각 손이 두 어깨 사이에서 자기 파지점을 바라보는 방향을 가리키도록 손 방향을 돌립니다(가장 작은 회전, 그 축을 도는 롤은 녹화 그대로). 팔이 그 방향을 맞추지 못하면 각도 허용만큼 틀어질 수 있습니다.">
                <input type="checkbox" disabled={disabled} checked={!!activeFrame.motion.camera_aim} onChange={e => { const { camera_aim: _old, ...rest } = activeFrame.motion!; setMotion(e.target.checked ? { ...rest, camera_aim: true } : rest); }}/>시선 방향</label>
              {activeFrame.motion.target.ends?.tcp && <span className="dynamic-offset" title="각 손의 TCP가 물체 끝점에서 얼마나 떨어질지(mm, 양손 같은 값). 안쪽은 물체 가운데 방향, 옆은 +가 로봇 쪽, 위는 +z입니다.">
                {([['inward_m', '안쪽'], ['lateral_m', '옆'], ['up_m', '위']] as const).map(([key, label]) => <label key={key}>{label}
                  <NumericInput aria-label={`TCP ${label} (mm)`} step="1" disabled={disabled} value={Math.round(((activeFrame.motion!.target as ObjectTarget).ends!.tcp![key]) * 1000)}
                    onChange={e => setRowEndTcp({ ...(activeFrame.motion!.target as ObjectTarget).ends!.tcp!, [key]: (+e.target.value || 0) / 1000 })}/></label>)}
              </span>}
            </> : <span className="dynamic-offset" title="두 손(TCP) 가운데가 도착할 월드 좌표(mm)입니다. 손 간격과 방향은 녹화 그대로입니다.">
              {(['X', 'Y', 'Z'] as const).map((axis, i) => <label key={axis}>{axis}
                <NumericInput aria-label={`목표 ${axis} (mm)`} step="1" disabled={disabled} value={Math.round(((activeFrame.motion!.target as { point: number[] }).point[i]) * 1000)}
                  onChange={e => { const point = [...(activeFrame.motion!.target as { point: number[] }).point]; point[i] = (+e.target.value || 0) / 1000; patchMotion({ target: { kind: 'point', point } }); }}/></label>)}
            </span>}
            <span className="dynamic-offset" title="목표에 더하는 TCP 오프셋(mm)과 수직축 회전(도)입니다.">
              {(['X', 'Y', 'Z'] as const).map((axis, i) => <label key={axis}>{`+${axis}`}
                <NumericInput aria-label={`오프셋 ${axis} (mm)`} step="1" disabled={disabled} value={Math.round((activeFrame.motion!.offset?.position[i] ?? 0) * 1000)}
                  onChange={e => { const position = [...(activeFrame.motion!.offset?.position ?? [0, 0, 0])]; position[i] = (+e.target.value || 0) / 1000; patchMotion({ offset: { position, yaw_deg: activeFrame.motion!.offset?.yaw_deg ?? 0 } }); }}/></label>)}
              <label>회전°<NumericInput aria-label="오프셋 회전 (도)" step="1" disabled={disabled} value={activeFrame.motion.offset?.yaw_deg ?? 0}
                onChange={e => patchMotion({ offset: { position: activeFrame.motion!.offset?.position ?? [0, 0, 0], yaw_deg: +e.target.value || 0 } })}/></label>
              <label title="팔이 손 방향을 정확히 맞추지 못할 때 손이 목표 방향에서 이만큼(도)까지 틀어져도 됩니다. 허리를 숙이기 전에 먼저 써 봅니다. 0이면 녹화 방향 그대로입니다.">각도 허용°
                <NumericInput aria-label="손 각도 허용 (도)" step="1" min="0" max="45" disabled={disabled} value={activeFrame.motion.angle_tolerance_deg ?? 0}
                  onChange={e => { const value = Math.min(45, Math.max(0, +e.target.value || 0)); const { angle_tolerance_deg: _old, ...rest } = activeFrame.motion!; setMotion(value ? { ...rest, angle_tolerance_deg: value } : rest); }}/></label>
            </span>
            {!activeIsClip && <button disabled={disabled} aria-pressed={showMotionAdvanced} onClick={() => setShowMotionAdvanced(value => !value)} title="손 선택·허리 고정·기울기·회전 오프셋">{showMotionAdvanced ? '세부 닫기' : '세부'}</button>}
            {(activeIsClip || showMotionAdvanced) && <>
            <span className="dynamic-hands">{(['left', 'right'] as const).map(side => <label key={side} className="dynamic-hand" title={`${side === 'left' ? '왼손' : '오른손'}을 목표에 맞춥니다`}>
              <input type="checkbox" disabled={disabled} checked={(activeFrame.motion!.hands ?? ['left', 'right']).includes(side)} onChange={e => {
                const hands = (['left', 'right'] as const).filter(item => item === side ? e.target.checked : (activeFrame.motion!.hands ?? ['left', 'right']).includes(item));
                if (hands.length) patchMotion({ hands });
              }}/>{side === 'left' ? '왼손' : '오른손'}</label>)}</span>
            <label className="dynamic-hand" title="켜면 허리를 고정하고 팔로만 목표에 닿습니다"><input type="checkbox" disabled={disabled} checked={activeFrame.motion.waist !== 'free'} onChange={e => patchMotion({ waist: e.target.checked ? 'lock' : 'free' })}/>허리 고정</label>
            <label className="dynamic-hand" title="켜면 물체가 기울어진 만큼 손도 기울입니다(기본은 이동과 수직축 회전만)"><input type="checkbox" disabled={disabled} checked={activeFrame.motion.freedom === 'full6d'} onChange={e => patchMotion({ freedom: e.target.checked ? 'full6d' : 'yaw_xyz' })}/>기울기</label>
            </>}
          </>}
          {activeFrame.motion && activeFrame.motion.mode !== 'recorded' && <span className={`dynamic-chip ${motionReport?.status ?? 'pending'}`} title="이 카드의 끝 지점 판정">{motionStatusText(motionReport)}</span>}
          {motionAdvice(motionReport) && <span className="motion-advice" role="status">{motionAdvice(motionReport)}</span>}
          {activeFrame.dynamic && <button disabled={disabled} onClick={() => { checkpoint(); editFrame(frameIndex, { dynamic: undefined }); setMessage('이전 방식(창 번호)의 동적 파지를 해제했습니다.'); }} title="이 클립에는 이전 방식의 동적 파지가 남아 있습니다. 해제하면 동작 방식으로 대체할 수 있습니다.">이전 동적 파지 해제</button>}
        </div>
      </> : <span className="tool-hint">키프레임을 선택하세요.</span>}</div>
      <div className="timeline-body"><div className="transport"><button className="play" title={playing ? '일시정지' : physicsEnabled || objectPhysicsEnabled ? `${controllerLabel} 재생` : '모션 재생'} disabled={!state || busy || solving} onClick={() => void play()}>{playing ? <Pause size={21}/> : <Play size={21}/>}</button><span>{(streamTime ?? preview?.time[sample] ?? 0).toFixed(2)}<small> / {(preview?.summary?.reference_seconds ?? duration).toFixed(2)}s</small></span><div className="transport-selects"><label><select aria-label="재생 속도" title="재생 속도 (화면 재생에만 적용)" value={playbackSpeed} onChange={e => setPlaybackSpeed(+e.target.value)}>{[0.25, 0.5, 1, 1.5, 2, 4].map(v => <option key={v} value={v}>{v}×</option>)}</select></label><label><select aria-label="출력 FPS" value={fps} disabled={disabled} onChange={e => setFps(+e.target.value)}>{[15, 30, 50, 60, 100, 120].map(f => <option key={f} value={f}>{f} fps</option>)}</select></label><label className="path-toggle" title="미리보기 전체 동안 두 손(TCP)이 지나는 경로를 선으로 보여 줍니다(분홍 왼손, 파랑 오른손). 물체 연결 카드가 따라가는 물체는 파란빛으로 강조합니다."><input type="checkbox" checked={showPaths} onChange={e => setShowPaths(e.target.checked)}/>경로</label></div>
        <div className="simulation-toggles">
        <button className="policy-toggle" type="button" aria-pressed={physicsEnabled} disabled={!state || busy || solving || playing || !physicsAvailable} onClick={() => changeSimulation(!physicsEnabled, false)} title={!physicsAvailable ? '물리 재생을 사용하려면 서버를 업데이트하고 재시작하세요.' : 'ON: 중력·접촉·관절 토크 계산 · OFF: 정책도 끄고 원본 편집으로 돌아가기'}>
          <span className="policy-indicator"/>Physics <b>{physicsEnabled ? 'ON' : 'OFF'}</b>
        </button>
        <button className="policy-toggle" type="button" aria-pressed={objectPhysicsEnabled} disabled={!state || busy || solving || playing || !physicsAvailable} onClick={() => changeSimulation(false, false, !objectPhysicsEnabled)} title="로봇은 키프레임 궤적을 정확히 따르고 오브젝트에만 중력·접촉·마찰을 적용합니다. 로봇 토크나 파지 안정성은 검증하지 않습니다.">
          <span className="policy-indicator"/>Object Physics <b>{objectPhysicsEnabled ? 'ON' : 'OFF'}</b>
        </button>
        <button className="policy-toggle" type="button" aria-pressed={policyEnabled} disabled={!state || busy || solving || playing || !physicsAvailable || !policyAvailable} onClick={() => changeSimulation(true, !policyEnabled)} title={!physicsAvailable || !policyAvailable ? 'SONIC CPU 환경 또는 서버 업데이트를 확인하세요.' : 'ON: Physics도 함께 켜고 GEAR-SONIC 정책 추종 · OFF: 물리는 유지하고 기본 PD 관절 제어 사용'}>
          <span className="policy-indicator"/>GEAR-SONIC <b>{policyEnabled ? 'ON' : 'OFF'}</b>
        </button>
        </div>
      </div>
        <div className="frame-track">{project?.keyframes.map((f, i) => {
          const group = f.lab_grasp ? labGroupRange(project.keyframes, i) : null;
          const collapsed = !!group && collapsedLabGroups.includes(group.id);
          return <Fragment key={i}>
            {group?.start === i && <button className="lab-frame-header" disabled={disabled} onClick={() => setCollapsedLabGroups(values => values.includes(group.id) ? values.filter(id => id !== group.id) : [...values, group.id])} title="파지법 구간 접기/펼치기">파지법 {group.frames.length}개 {collapsed ? '▸' : '▾'}</button>}
            {(!collapsed || group?.start === i) && <button disabled={!state || busy || solving || !!preview?.physics} data-frame-index={i} className={`frame-card ${frameIndex === i ? 'active' : ''} ${playing && playbackFrame === i ? 'playing' : ''} ${!playing && playbackFrame === i ? 'playhead' : ''} ${playing && stage && i < stage.index ? 'stage-done' : ''} ${transitionSelection.includes(i) ? 'transition-selected' : ''} ${f.generated_transition ? 'generated-transition' : ''} ${f.lab_grasp ? 'lab-linked-frame' : ''}`} aria-pressed={transitionSelection.includes(i)} onClick={() => selectTransitionFrame(i)} title={f.lab_grasp ? `Cylinder Lab 파지법 · 그린 점 ${f.lab_grasp.point_index + 1}` : '두 키프레임을 순서대로 클릭해 연결 구간을 선택합니다'}>{playing && stage?.index === i && <i className="stage-fill" style={{ width: `${stage.progress * 100}%` }}/>}<span className="frame-number">{String(i+1).padStart(2, '0')}</span><div><strong>{f.name}{f.path && f.path.kind !== 'joint' && <b className="mode-tag path-tag" title={f.path.kind === 'linear' ? '직선 이동(MoveL): 앞 카드에서 이 카드까지 두 손이 직선으로' : '원호 이동(MoveC): 앞 카드에서 이 카드까지 두 손이 호를 그리며'}>{f.path.kind === 'linear' ? 'L' : 'C'}</b>}{f.generated_transition && <b className="mode-tag smooth-tag" title="부드러운 연결: 이 카드를 지날 때 멈추지 않고 앞뒤 카드와 속도가 이어집니다">∿</b>}</strong><small><span>{f.lab_grasp ? '파지법 · ' : ''}{f.samples ? `${f.dynamic ? '동적 ' : ''}클립 ${f.samples.length}프레임 · ${f.duration.toFixed(1)}s` : '포즈'} · <b className="card-start">{formatClock((preview?.keyframe_times ?? cardStarts)[i] ?? 0)}</b></span><span>{cardOutgoingText(project.keyframes, i)} · 위치 {f.pins.length} · 각도 {f.angle_pins?.length ?? 0}</span></small></div><div className="mini-pose"><i style={{ height: `${22 + (f.qpos[2] - .5) * 40}px` }}/></div></button>}
          </Fragment>;
        })}</div>
      </div>
      <input className="scrubber" aria-label="모션 시간 탐색" type="range" min="0" max={Math.max(1, (preview?.states.length ?? 1)-1)} value={sample} disabled={!preview || busy} onChange={e => { setPlaying(false); const i = +e.target.value; setSample(i); if (preview) { applyState(preview.states[i]); if (preview.object_states?.[i]) scene.current?.setObjectPoses(preview.object_states[i]); } }}/>
    </section>
    {robotOpen && project && <aside className="robot-dock">
      <ReplayPanel inline fps={fps} frameNames={project.keyframes.map(frame => frame.name)} onClose={() => setRobotOpen(false)}
        getProject={() => sceneProject(current.current.project ?? project)}
        onPrepared={() => setCheckedFor({ project, objects, groups: objectGroups, tools })}
        startBlocked={robotBlocked}/>
    </aside>}
    <footer className={`statusbar ${error ? 'error' : ''}`}><span>{error ? <AlertCircle size={13}/> : <span className="status-dot"/>}{error || message}</span><span>{physicsEnabled || objectPhysicsEnabled ? `MuJoCo + ${controllerLabel} · ${preview?.physics ? '시뮬레이션 결과' : 'CPU 재생 준비'}` : '원본 모션 · 물리 OFF'}</span></footer>
  </div>;
}
