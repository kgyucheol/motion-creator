'use client';
import { useEffect, useRef, useState } from 'react';
import TaskWorkbench from './task-workbench';
import SceneObjectControls from './scene-object-controls';
import BodyControls from './body-controls';
import { allNodes, nodeMembers, selectMembers, controlKey, controlSelection, groupForControl, visibleTreeHandles, expandVirtualControls } from '../lib/body-groups';
import { Play, Pause, Plus, Save, FolderOpen, RotateCcw, Undo2, Redo2, LockKeyhole, MousePointer2, Move3d, Box, ChevronLeft, ChevronRight, Trash2, Download, Check, AlertCircle } from 'lucide-react';
import { RobotScene, canRotateSelection, isJointHandle, HIP_HANDLES, ANKLE_HANDLES, COMBINED_JOINTS, type PoseState, type TransformMode } from '../lib/robot-scene';
import { eulerDegrees, quaternionFromDegrees, rotatedGroupTargets, incrementRotation, canMirrorSelection, translatedTargets } from '../lib/pose-transforms';
import { createSceneObject, normalizedObjectSize, objectsFromProject, type ObjectTransformMode, type SceneObject, type SceneObjectPose, type SceneObjectShape } from '../lib/scene-objects';

type Keyframe = { name: string; duration: number; qpos: number[]; pins: string[]; samples?: number[][] };
type Project = { format: string; name: string; model_sha256: string; joint_names: string[]; coordinate_system: string; units: Record<string, string>; keyframes: Keyframe[]; current_qpos?: number[]; pins?: string[]; scene_objects?: SceneObject[]; box?: { position: number[]; size: number[]; visible: boolean } };
type Preview = { time: number[]; states: PoseState[]; object_states?: Record<string, SceneObjectPose>[]; max_pin_error_mm: number; physics?: boolean; summary?: { reason: string; joint_rmse_rad: number; sim_seconds: number; reference_seconds: number } };
type PolicyJob = { id: string; status: 'running' | 'completed' | 'cancelled' | 'failed'; progress: number; message?: string };
type SolveInfo = { target_error_mm: number; pin_error_mm: number; rejected: boolean; converged: boolean; target_errors_mm?: Record<string, number>; angle_error_deg?: number };
type GroupPreset = { id: string; name: string; members: string[] };
const feet = ['left_foot', 'right_foot'];
const copy = <T,>(value: T): T => structuredClone(value);
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
  const [taskOpen, setTaskOpen] = useState(() => typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('task'));
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<RobotScene | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<PoseState | null>(null);
  const [project, setProject] = useState<Project | null>(null);
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
  const [mode, setMode] = useState('elastic');
  const [transformMode, setTransformMode] = useState<TransformMode>('translate');
  const [mirror, setMirror] = useState(false);
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
  const [poseDirty, setPoseDirty] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [sample, setSample] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [physicsEnabled, setPhysicsEnabled] = useState(false);
  const [physicsAvailable, setPhysicsAvailable] = useState(false);
  const [policyEnabled, setPolicyEnabled] = useState(false);
  const [policyAvailable, setPolicyAvailable] = useState(false);
  const [policyJob, setPolicyJob] = useState<PolicyJob | null>(null);
  const policyRequest = useRef<{ id?: string; cancelled: boolean } | null>(null);
  const policySource = useRef<{ state: PoseState; pins: string[]; frameIndex: number; dirty: boolean } | null>(null);
  const [fps, setFps] = useState(30);
  const [exportProto, setExportProto] = useState(false);
  const [showHandles, setShowHandles] = useState(true);
  const [objects, setObjects] = useState<SceneObject[]>(() => objectsFromProject({}));
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [objectTransformMode, setObjectTransformMode] = useState<ObjectTransformMode>('translate');
  const [files, setFiles] = useState<string[]>([]);
  const [saved, setSaved] = useState<string[]>([]);
  const [savedChoice, setSavedChoice] = useState('');
  const [historyCount, setHistoryCount] = useState(0);
  const [futureCount, setFutureCount] = useState(0);
  const current = useRef({ state, project, pins, mode, resistance, selected, members, transformMode, mirror, busy, playing, solving, objects, selectedObjectId, objectTransformMode });
  current.current = { state, project, pins, mode, resistance, selected, members, transformMode, mirror, busy, playing, solving, objects, selectedObjectId, objectTransformMode };
  const history = useRef<{ qpos: number[]; pins: string[] }[]>([]);
  const future = useRef<{ qpos: number[]; pins: string[] }[]>([]);
  const anchor = useRef<number[]>([]);
  const anchorPositions = useRef<Record<string, number[]>>({});
  const anchorCenter = useRef<number[]>([0, 0, 0]);
  const anchorOrientations = useRef<Record<string, number[]>>({});
  const anchorQuaternion = useRef([0, 0, 0, 1]);
  const anchorMirror = useRef<{ active: string; rootQuaternion: number[] } | undefined>(undefined);
  const pending = useRef<{ kind: TransformMode | 'joint'; key: string; target: number[]; joints?: Record<string, number> } | null>(null);
  const inFlight = useRef(false);
  const dragActive = useRef(false);
  const alive = useRef(true);
  const actions = useRef({ select: (_key: string, _additive: boolean, _hover: boolean) => {}, begin: () => {}, move: (_key: string, _target: number[]) => {}, rotate: (_key: string, _quaternion: number[]) => {}, jointAngle: (_key: string, _angle: number) => {}, transformMode: (_mode: TransformMode) => {}, history: (_redo: boolean) => {}, end: () => {} });

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
  function checkpoint() {
    if (!current.current.state) return;
    history.current.push({ qpos: [...current.current.state.qpos], pins: [...current.current.pins] });
    if (history.current.length > 60) history.current.shift();
    future.current = [];
    setHistoryCount(history.current.length); setFutureCount(0);
  }
  function invalidate() { setPreview(null); setSample(0); setPlaying(false); }
  function commitObjects(next: SceneObject[]) {
    current.current.objects = next;
    setObjects(next);
    setProject(value => value ? { ...value, scene_objects: next } : value);
    invalidate();
  }
  function changeObject(id: string, patch: Partial<SceneObject>) {
    commitObjects(current.current.objects.map(object => {
      if (object.id !== id) return object;
      const next = { ...object, ...patch };
      return { ...next, size: normalizedObjectSize(next.shape, next.size) };
    }));
  }
  function selectObject(id: string | null) {
    setSelectedObjectId(id); current.current.selectedObjectId = id;
    scene.current?.selectSceneObject(id, current.current.objectTransformMode);
  }
  function changeObjectMode(mode: ObjectTransformMode) {
    setObjectTransformMode(mode); current.current.objectTransformMode = mode;
    scene.current?.selectSceneObject(current.current.selectedObjectId, mode);
  }
  function addObject(shape: SceneObjectShape) {
    const object = createSceneObject(current.current.objects.length + 1, shape);
    commitObjects([...current.current.objects, object]);
    selectObject(object.id);
  }
  function removeObject(id: string) {
    commitObjects(current.current.objects.filter(object => object.id !== id));
    selectObject(null);
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
    if (scene.current && !hover) scene.current.selectionLocked = true;
    const previousMode = current.current.transformMode;
    const controls = controlSelection(next);
    const nextMode = controls.length === 1 && isJointHandle(controls[0]) && !hover ? 'rotate' : previousMode === 'rotate' && !canRotateSelection(controls) ? 'translate' : previousMode;
    current.current.transformMode = nextMode; setTransformMode(nextMode);
    scene.current?.select(active, current.current.pins, next);
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
      ? { active: controlKey(current.current.members, current.current.selected), rootQuaternion: [...current.current.state.handles.pelvis.quaternion] } : undefined;
    anchorOrientations.current = Object.fromEntries(controls.map(k => [k, [...current.current.state!.handles[k].quaternion]]));
    anchorQuaternion.current = [...current.current.state.handles[controlKey(current.current.members, current.current.selected)].quaternion];
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
          pins: c.pins, mode: c.mode, resistance: c.resistance,
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
    pending.current = { kind: 'joint', key, target: [], joints: { [hinge.joint_name]: value } };
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
  }, begin, move, rotate, jointAngle, transformMode: changeTransformMode, history: changeHistory, end: () => {
    dragActive.current = false;
    if (current.current.state) applyState(current.current.state);
  } };

  useEffect(() => {
    alive.current = true;
    let viewer: RobotScene | null = null;
    try {
      viewer = new RobotScene(host.current!, {
        select: (key, additive, hover) => actions.current.select(key, additive, hover), begin: () => actions.current.begin(),
        move: (key, value) => actions.current.move(key, value), rotate: (key, value) => actions.current.rotate(key, value), end: () => actions.current.end(), error: setError,
        transformMode: mode => actions.current.transformMode(mode),
        jointAngle: (key, angle) => actions.current.jointAngle(key, angle),
        history: redo => actions.current.history(redo),
        selectObject,
        transformObject: changeObject,
        objectTransformMode: changeObjectMode,
      });
      scene.current = viewer;
    } catch { setError('WebGL을 시작하지 못했습니다. 브라우저의 하드웨어 가속 설정을 확인하세요.'); }
    api<{ state: PoseState; project: Project; limits: number[][] }>('init').then(async init => {
      if (!alive.current) return;
      setLimits(init.limits.map(range => range.map(v => v * 180 / Math.PI)));
      let nextProject = init.project;
      let initialState = init.state;
      let initialPins = feet;
      try {
        const draft = localStorage.getItem('g1-motion-draft-v1');
        if (draft) {
          nextProject = await api<Project>('validate', { project: JSON.parse(draft) });
          initialState = await api<PoseState>('pose', { qpos: nextProject.current_qpos ?? nextProject.keyframes[0].qpos });
          initialPins = nextProject.pins ?? feet;
        }
      } catch { setMessage('이전 자동 저장을 복원하지 못해 기본 자세로 시작했습니다.'); nextProject = init.project; }
      if (!alive.current) return;
      const nextObjects = objectsFromProject(nextProject);
      nextProject = { ...nextProject, scene_objects: nextObjects };
      setObjects(nextObjects); current.current.objects = nextObjects;
      setProject(nextProject); current.current.project = nextProject;
      setPins(initialPins); current.current.pins = initialPins;
      applyState(initialState); setTarget(initialState.handles.pelvis.position);
      setMessage('부위에 마우스를 올리고 축을 드래그하세요. 양발은 고정되어 있습니다.');
    }).catch(e => setError(`계산 서버에 연결할 수 없습니다: ${e.message}`));
    api<string[]>('saved').then(setSaved).catch(() => {});
    api<{ available: boolean; physics_available?: boolean }>('policy-preview/runtime').then(result => {
      if (alive.current) { setPolicyAvailable(result.available); setPhysicsAvailable(!!result.physics_available); }
    }).catch(() => {});
    api<GroupPreset[]>('groups').then(setGroups).catch(e => setError(`그룹 프리셋 불러오기 실패: ${e.message}`));
    return () => {
      alive.current = false; viewer?.dispose(); scene.current = null;
      const request = policyRequest.current;
      if (request) {
        request.cancelled = true;
        if (request.id) void api(`policy-preview/${request.id}/cancel`, {}).catch(() => {});
      }
    };
  }, []);
  useEffect(() => { scene.current?.select(selected, pins, members); }, [selected, pins, members]);
  useEffect(() => {
    scene.current?.setTransformMode(transformMode, space);
    const pose = current.current.state;
    if (pose) setTarget(selectionPosition(pose, members, selected));
  }, [transformMode, space, mirror]);
  useEffect(() => { if (scene.current) { scene.current.setEditable(!busy && !playing && !taskOpen && !preview?.physics); scene.current.keyboardEnabled = !taskOpen && !preview?.physics; } }, [busy, playing, taskOpen, preview]);
  useEffect(() => { scene.current?.showHandles(showHandles); }, [showHandles]);
  useEffect(() => { scene.current?.setVisibleHandles(visibleTreeHandles(expanded)); }, [expanded]);
  useEffect(() => { scene.current?.setMirrorTranslation(mirror); }, [mirror]);
  useEffect(() => { scene.current?.setSceneObjects(objects); }, [objects]);
  useEffect(() => {
    if (!project || !state || playing || preview?.physics) return;
    const timer = setTimeout(() => {
      try { localStorage.setItem('g1-motion-draft-v1', JSON.stringify({ ...project, current_qpos: state.qpos, pins, scene_objects: objects })); }
      catch { setMessage('브라우저 자동 저장 공간이 부족합니다. 파일 저장을 사용하세요.'); }
    }, 500);
    return () => clearTimeout(timer);
  }, [project, state, pins, objects, playing, preview]);
  useEffect(() => {
    if (!playing || !preview) return;
    const start = performance.now() - preview.time[sample] * 1000;
    let id = 0;
    const tick = () => {
      const time = (performance.now() - start) / 1000;
      const end = preview.time.length - 1;
      // Physics replays may end between display frames when a fall is detected.
      let index = 0;
      while (index < end && preview.time[index + 1] <= time) index++;
      applyState(preview.states[index]);
      if (preview.object_states?.[index]) scene.current?.setObjectPoses(preview.object_states[index]);
      setSample(index);
      if (index === end) { setPlaying(false); return; }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, preview]);

  function togglePin(key: string) {
    if (busy || solving || playing) return;
    checkpoint();
    setPins(p => p.includes(key) ? p.filter(k => k !== key) : [...p, key]);
    invalidate(); setPoseDirty(true);
  }
  async function loadProject(next: Project) {
    const valid = await api<Project>('validate', { project: next });
    const pose = await api<PoseState>('pose', { qpos: valid.current_qpos ?? valid.keyframes[0].qpos });
    const nextObjects = objectsFromProject(valid);
    const normalized = { ...valid, scene_objects: nextObjects };
    setProject(normalized); current.current.project = normalized; setFrameIndex(0);
    setObjects(nextObjects); current.current.objects = nextObjects; selectObject(null);
    setPins(valid.pins ?? valid.keyframes[0].pins);
    applyState(pose); invalidate(); setPoseDirty(false); setInfo(null);
    history.current = []; future.current = []; setHistoryCount(0); setFutureCount(0);
  }
  async function openFile(source: File) {
    const extension = source.name.slice(source.name.lastIndexOf('.')).toLowerCase();
    if (extension === '.json') {
      await loadProject(JSON.parse(await source.text()));
      setMessage(`${source.name} 프로젝트를 불러왔습니다.`);
      return;
    }
    const response = await fetch(`/api/import-motion?filename=${encodeURIComponent(source.name)}&fps=${fps}`, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: source,
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({})) as { detail?: unknown };
      throw new Error(typeof failure.detail === 'string' ? failure.detail : `파일 불러오기 실패 (${response.status})`);
    }
    await loadProject(await response.json() as Project);
    setMessage(`${source.name}의 모션 프레임을 ${fps} FPS 키프레임으로 불러왔습니다.`);
  }
  function editFrame(index: number, patch: Partial<Keyframe>) {
    setProject(p => p ? { ...p, keyframes: p.keyframes.map((f, i) => i === index ? { ...f, ...patch } : f) } : p);
    invalidate();
  }
  function addFrame() {
    if (!project || !state) return;
    const next = { name: `Pose ${project.keyframes.length + 1}`, duration: 2, qpos: [...state.qpos], pins: [...pins] };
    setProject({ ...project, keyframes: [...project.keyframes, next] }); setFrameIndex(project.keyframes.length);
    setPoseDirty(false); invalidate(); setMessage('현재 자세를 새 키프레임에 추가했습니다.');
  }
  async function chooseFrame(index: number) {
    if (!project) return;
    const f = project.keyframes[index];
    await run(async () => { checkpoint(); applyState(await api<PoseState>('pose', { qpos: f.qpos })); setPins(f.pins); setFrameIndex(index); setPoseDirty(false); setInfo(null); });
  }
  function changeHistory(redo: boolean) {
    if (current.current.busy || current.current.playing || inFlight.current || dragActive.current) return;
    void run(async () => {
      const from = redo ? future.current : history.current;
      const to = redo ? history.current : future.current;
      const value = from[from.length - 1];
      if (!value || !current.current.state) return;
      const restored = await api<PoseState>('pose', { qpos: value.qpos });
      from.pop();
      to.push({ qpos: [...current.current.state.qpos], pins: [...current.current.pins] });
      applyState(restored); setPins(value.pins); setInfo(null);
      setHistoryCount(history.current.length); setFutureCount(future.current.length); setPoseDirty(true); invalidate();
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
  function exportProject() {
    if (!project || !state) return;
    void run(async () => {
      const result = await api<{ files: string[]; directory: string; warnings?: string[]; metadata: { samples: number; validation: { max_pin_error_mm: number } } }>('save', {
        project: { ...project, current_qpos: state.qpos, pins, scene_objects: objects }, fps, protomotions: exportProto,
      });
      setFiles(result.files); setSaved(await api<string[]>('saved'));
      setMessage(`${result.metadata.samples}프레임 저장 완료 · ${result.directory} · 고정 오차 최대 ${result.metadata.validation.max_pin_error_mm.toFixed(2)} mm`);
      if (result.warnings?.length) setError(result.warnings.join(' '));
    });
  }
  const controllerLabel = policyEnabled ? 'GEAR-SONIC' : '기본 PD';
  async function play() {
    if (playing) { setPlaying(false); return; }
    if (preview) { if (sample >= preview.states.length - 1) setSample(0); setPlaying(true); return; }
    await run(async () => {
      if (!physicsEnabled) {
        const result = await api<Preview>('preview', { project, fps: 30 });
        setPreview(result); setSample(0); setPlaying(true); setPoseDirty(false);
        if (result.object_states?.[0]) scene.current?.setObjectPoses(result.object_states[0]);
        setMessage('원본 모션 재생 · 물리 OFF');
        return;
      }
      const request: { id?: string; cancelled: boolean } = { cancelled: false };
      policySource.current = state ? { state, pins: [...pins], frameIndex, dirty: poseDirty } : null;
      policyRequest.current = request;
      try {
        setMessage(`${controllerLabel} 물리 계산 준비 중…`);
        let job = await api<PolicyJob>('policy-preview', { project, controller: policyEnabled ? 'gear-sonic' : 'pd' });
        request.id = job.id;
        if (request.cancelled) {
          await api(`policy-preview/${job.id}/cancel`, {});
          return;
        }
        setPolicyJob(job);
        while (job.status === 'running') {
          await new Promise(resolve => setTimeout(resolve, 500));
          if (!alive.current || request.cancelled) return;
          job = await api<PolicyJob>(`policy-preview/${job.id}`);
          setPolicyJob(job);
          setMessage(`${controllerLabel} 물리 계산 ${Math.round(job.progress * 100)}% · CPU`);
        }
        if (job.status === 'cancelled') { setMessage('물리 계산을 취소했습니다.'); return; }
        if (job.status !== 'completed') throw new Error(job.message || '물리 계산에 실패했습니다.');
        const result = await api<Preview>(`policy-preview/${job.id}/result`);
        if (!alive.current || request.cancelled) return;
        setPreview(result); setSample(0); setPlaying(true); setPoseDirty(false);
        if (result.object_states?.[0]) scene.current?.setObjectPoses(result.object_states[0]);
        const summary = result.summary!;
        const outcome = summary.reason === 'fallen' ? '넘어짐으로 조기 종료' : summary.reason === 'completed' ? '계산 완료' : '수치 불안정으로 중단';
        setMessage(`${controllerLabel} ${outcome} · ${summary.sim_seconds.toFixed(2)}초 · 관절 추종 오차 ${(summary.joint_rmse_rad * 180 / Math.PI).toFixed(1)}°`);
      } catch (failure) {
        if (request.id) await api(`policy-preview/${request.id}/cancel`, {}).catch(() => {});
        throw failure;
      } finally {
        if (alive.current) setPolicyJob(null);
        policyRequest.current = null;
      }
    });
  }
  async function cancelPolicy() {
    const request = policyRequest.current;
    if (!request?.id) return;
    try {
      await api(`policy-preview/${request.id}/cancel`, {});
      request.cancelled = true;
      setMessage('물리 계산을 취소했습니다.');
    } catch (failure) { setError((failure as Error).message); }
  }
  function changeSimulation(physics: boolean, policy: boolean) {
    const source = policySource.current;
    if (preview?.physics && source) {
      applyState(source.state); setFrameIndex(source.frameIndex); setPins(source.pins); setPoseDirty(source.dirty);
    }
    policySource.current = null;
    scene.current?.setSceneObjects(current.current.objects);
    invalidate(); setPhysicsEnabled(physics); setPolicyEnabled(physics && policy);
    setMessage(!physics ? '원본 모션 재생 · 물리 OFF' : policy
      ? 'MuJoCo + GEAR-SONIC · 재생하면 CPU 계산 후 결과를 보여줍니다.'
      : 'MuJoCo + 기본 PD · 균형 정책 없이 관절을 제어합니다. 불안정한 모션은 넘어질 수 있습니다.');
  }
  const disabled = !state || busy || solving || playing || !!preview?.physics;
  const selectionPinned = members.some(k => pins.includes(k));
  const controls = controlSelection(members);
  const mirrorAvailable = canMirrorSelection(controls);
  const mirrorActive = mirror && mirrorAvailable && transformMode === 'translate';
  const activeControl = controlKey(members, selected);
  const rotationAllowed = canRotateSelection(controls);
  const rotationBlocked = !rotationAllowed || controls.some(k => pins.includes(k) && (controls.length > 1 || feet.includes(k)));
  const hinge = controls.length === 1 ? state?.hinges?.[activeControl] : undefined;
  const ankle = controls.length === 1 && ANKLE_HANDLES.includes(activeControl);
  const selectedJoints = members.filter(key => key.endsWith('_joint'));
  const partJoints = selectedJoints.length > 1 ? selectedJoints : controls.length === 1 ? COMBINED_JOINTS[activeControl] : undefined;
  const hingeIndex = hinge ? project?.joint_names.indexOf(hinge.joint_name) ?? -1 : -1;
  const jointDirty = !!state && jointDraft.some((v, i) => Math.abs(v - state.qpos[7+i]*180/Math.PI) > .001);
  const activeFrame = project?.keyframes[frameIndex];
  const motionClip = project?.keyframes.length === 1 ? project.keyframes[0].samples : undefined;
  const duration = motionClip ? project!.keyframes[0].duration : project?.keyframes.slice(1).reduce((sum, f) => sum + f.duration, 0) ?? 0;
  const selectedGroup = allNodes().find(node => node.children && nodeMembers(node).length === members.length && nodeMembers(node).every(key => members.includes(key)));

  return <div className="editor">
    {taskOpen && state && <TaskWorkbench initialQ={state.qpos} onClose={() => { setTaskOpen(false); scene.current?.setEditable(true); }}/>}
    <header className="topbar">
      <div className="brand"><span className="brand-icon"><Move3d size={23}/></span><div>MOTION<span>CREATOR</span></div><b>G1 / 29 DOF</b></div>
      <div className="project-title"><span className="status-dot"/>{project ? <input aria-label="프로젝트 이름" value={project.name} onChange={e => setProject({ ...project, name: e.target.value })}/> : '연결 중'}</div>
      <div className="top-actions"><button disabled={disabled} onClick={() => { setPlaying(false); scene.current?.setEditable(false); setTaskOpen(true); }}><Box size={16}/>상자 태스크</button><button disabled={disabled} onClick={() => file.current?.click()}><FolderOpen size={16}/> 열기</button><button className="primary" disabled={disabled} onClick={exportProject}><Save size={16}/> 모션 저장</button></div>
      <input ref={file} type="file" accept=".json,.npz,.csv" hidden onChange={e => { const source = e.target.files?.[0]; if (source) void run(() => openFile(source)); e.target.value = ''; }}/>
    </header>
    <aside className="left-panel panel">
      <div className="panel-heading"><span>BODY GROUPS</span><small>29자유도</small></div>
      <p className="hint">그룹 이름: 전체 선택 · 화살표: 펼치기/접기 · 하위 관절: 하나만 선택 · Shift: 추가/해제</p>
      <BodyControls pose={state} selected={members} pins={pins} expanded={expanded} disabled={disabled} onExpand={toggleExpanded} onSelect={selectBatch} onPin={togglePin}/>
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
      <p className="hint">일반 이동은 함께 옮기고, 좌우 미러 이동은 짝을 벌리거나 모읍니다. 고정된 부위가 포함되면 먼저 해제하세요.</p>
      <div className="section-divider"/>
      <div className="panel-heading"><span>IK BEHAVIOR</span></div>
      <div className="segmented"><button disabled={disabled} className={mode === 'elastic' ? 'chosen' : ''} onClick={() => setMode('elastic')}>유연하게 따라오기</button><button disabled={disabled} className={mode === 'free' ? 'chosen' : ''} onClick={() => setMode('free')}>고정 부위만 유지</button></div>
      <label className="range-label">주변 부위 저항 <span>{resistance.toFixed(1)}</span><input aria-label="주변 부위 저항" type="range" min="0" max="5" step="0.1" value={resistance} disabled={disabled || mode === 'free'} onChange={e => setResistance(+e.target.value)}/></label>
      <p className="hint">먼 부위일수록 원래 위치를 더 유지합니다. 손의 위치를 정확히 유지하려면 손을 고정하세요.</p>
      <div className="pin-summary"><LockKeyhole size={14}/><span>발: 위치 + 방향<br/>그 외: 위치 고정</span></div>
      <div className="section-divider"/>
      <button className="wide" disabled={disabled} onClick={() => void run(async () => { checkpoint(); const init = await api<{state: PoseState}>('init'); applyState(init.state); setPins(feet); setPoseDirty(true); invalidate(); })}><RotateCcw size={15}/> 기본 서기 자세</button>
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
      <div className="viewport-top"><div className="view-title"><span className="status-dot"/>POSE WORKSPACE<span>m · rad · Z-up</span></div><div className="view-buttons">{(['perspective', 'front', 'side'] as const).map((v, i) => <button key={v} onClick={() => scene.current?.setView(v)}>{['자유', '정면', '측면'][i]}</button>)}</div></div>
      <div className={`simulation-status ${physicsEnabled ? 'enabled' : ''}`}>
        <span>Physics <b data-active={physicsEnabled}>{physicsEnabled ? 'ON' : 'OFF'}</b></span>
        <span>GEAR-SONIC <b data-active={policyEnabled}>{policyEnabled ? 'ON' : 'OFF'}</b></span>
        {physicsEnabled && <small>{controllerLabel} · {preview?.physics ? '계산된 결과' : policyJob ? '계산 중' : '재생 대기'}</small>}
        {preview?.summary && <small>{preview.summary.reason === 'completed' ? '추종 결과' : preview.summary.reason === 'fallen' ? '넘어짐 감지' : '계산 중단'} · 오차 {(preview.summary.joint_rmse_rad * 180 / Math.PI).toFixed(1)}°</small>}
      </div>
      {policyJob && <div className="policy-progress">
        <span>{controllerLabel} 물리 계산 <b>{Math.round(policyJob.progress * 100)}%</b></span>
        <progress aria-label="물리 계산 진행률" max={1} value={policyJob.progress}/>
        <button type="button" onClick={() => void cancelPolicy()}>계산 취소</button>
      </div>}
      <div className="viewport-tools"><button title="자세 실행 취소 (Ctrl+Z)" aria-keyshortcuts="Control+Z Meta+Z" disabled={disabled || !historyCount} onClick={() => changeHistory(false)}><Undo2 size={17}/></button><button title="자세 다시 실행 (Ctrl+Shift+Z)" aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z" disabled={disabled || !futureCount} onClick={() => changeHistory(true)}><Redo2 size={17}/></button><span/><button className={showHandles ? 'chosen' : ''} title="조작 표시 켜기/끄기" onClick={() => setShowHandles(!showHandles)}><MousePointer2 size={17}/></button></div>
      <div className="scene-legend"><span><i className="cyan"/>이동 가능</span><span><i className="amber"/>고정</span><span><i className="mint"/>선택</span></div>
      <div className="viewport-bottom"><span>드래그: 회전 · 휠: 확대 · 우클릭: 이동 · 로봇 W/E · 물체 W/E/R · F: 선택 보기</span><span className="axes"><b>X</b> 전방 <b>Y</b> 왼쪽 <b>Z</b> 위</span></div>
      {busy && <div className="busy-overlay"><span className="spinner"/>모션을 계산하고 있습니다…</div>}
    </main>
    <aside className="right-panel panel">
      <div className="panel-heading"><span>TRANSFORM</span><small>{transformMode === 'rotate' && (hinge || ankle) ? '관절축' : space.toUpperCase()}</small></div>
      <div className="segmented transform-modes"><button className={transformMode === 'translate' ? 'chosen' : ''} disabled={disabled} onClick={() => changeTransformMode('translate')} aria-keyshortcuts="W" title="이동 모드 (W)">이동 W</button><button className={transformMode === 'rotate' ? 'chosen' : ''} disabled={disabled || !rotationAllowed} onClick={() => changeTransformMode('rotate')} aria-keyshortcuts="E" title="회전 모드 (E)">회전 E</button></div>
      <div className="segmented"><button className={space === 'world' ? 'chosen' : ''} disabled={disabled || transformMode === 'rotate' && (!!hinge || ankle)} onClick={() => setSpace('world')} title="장면에 고정된 XYZ 축으로 드래그">월드 축</button><button className={space === 'local' ? 'chosen' : ''} disabled={disabled || transformMode === 'rotate' && (!!hinge || ankle)} onClick={() => setSpace('local')} title="선택 부위의 방향을 따라가는 XYZ 축으로 드래그">로컬 축</button></div>
      <div className="segmented"><button className={!mirror ? 'chosen' : ''} disabled={disabled} onClick={() => setMirror(false)}>일반 이동</button><button className={mirror ? 'chosen' : ''} disabled={disabled} aria-pressed={mirror} onClick={() => setMirror(true)}>좌우 미러 이동</button></div>
      {mirror && <p className="hint">{mirrorActive ? `기준: ${state?.handles[activeControl]?.label ?? activeControl}. 이쪽을 바깥/안쪽으로 드래그하면 반대쪽도 대칭 이동합니다. 골반의 좌우 평면을 기준으로 합니다.` : !mirrorAvailable ? '좌우 짝을 모두 선택하면 미러 이동이 활성화됩니다.' : 'W 이동 모드에서 미러 이동을 사용할 수 있습니다.'}</p>}
      <h2>{selectedGroup?.label ?? (members.length > 1 ? `${members.length}개 부위` : state?.handles[selected].label ?? '골반')} <span>{selectionPinned ? '고정 포함' : '이동 가능'}</span></h2>
      <p className="hint">{members.map(k => state?.handles[k].label ?? k).join(' · ')}</p>
      <button className="wide" disabled={!state} onClick={() => scene.current?.focusSelection()}>선택 부위 보기 <kbd>F</kbd></button>
      {partJoints && <div className="part-angle-editor">
        <label className="inspector-label">실제 관절각 <small>°</small></label>
        {partJoints.map(name => {
          const index = project?.joint_names.indexOf(name) ?? -1;
          const joint = state?.hinges[name];
          if (!joint || index < 0) return null;
          return <label className="part-angle-row" key={name}><span title={name}>{state?.handles[name]?.label ?? name}</span>
            <input aria-label={`${name} 부위 각도`} type="number" step=".1" min={joint.limits[0]*180/Math.PI} max={joint.limits[1]*180/Math.PI} value={Number((jointDraft[index] ?? joint.angle*180/Math.PI).toFixed(2))} disabled={disabled} onChange={e => changeJoint(index, +e.target.value)}/>
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
        <input className="group-control" aria-label="선택 관절각" type="number" step="1" min={hinge.limits[0]*180/Math.PI} max={hinge.limits[1]*180/Math.PI} value={Number((jointDraft[hingeIndex] ?? hinge.angle*180/Math.PI).toFixed(2))} disabled={disabled} onChange={e => changeJoint(hingeIndex, +e.target.value)}/>
        <button className="wide" disabled={disabled} onClick={() => applyHingeAngle((jointDraft[hingeIndex] ?? hinge.angle*180/Math.PI)*Math.PI/180)}>관절각 적용</button>
        <div className="group-actions"><button disabled={disabled} onClick={() => applyHingeAngle(hinge.angle-Math.PI/180)}>−1°</button><button disabled={disabled} onClick={() => applyHingeAngle(hinge.angle+Math.PI/180)}>+1°</button></div>
        <p className="hint">링 하나로 실제 관절축의 각도를 조정합니다. 월드/로컬 선택과 관계없이 관절축을 사용하며, 손·발 등 고정 조건을 함께 유지합니다. 슬라이더·숫자 입력 후에는 관절각 적용을 누르세요.</p>
      </> : transformMode === 'rotate' ? <>
        {(HIP_HANDLES.includes(activeControl) || activeControl === 'waist') && <p className="hint">롤 관절 위치의 세 회전 링으로 {activeControl === 'waist' ? '상체' : '다리'} 방향을 조정합니다. IK가 yaw·pitch·roll을 함께 계산합니다. 아래 방향 값은 월드 Euler 각도이며 위 실제 관절각과 구분됩니다.</p>}
        <label className="inspector-label">기준 부위 방향 · 월드 XYZ <small>°</small></label>
        <div className="xyz">{rotation.map((v, i) => <label key={i}><span className={`axis-${i}`}>{'XYZ'[i]}</span><input aria-label={`목표 ${'XYZ'[i]} 회전각`} type="number" step="1" value={Number(v.toFixed(2))} disabled={disabled || rotationBlocked} onChange={e => setRotation(t => t.map((n, j) => i === j ? +e.target.value : n))}/></label>)}</div>
        <button className="wide" disabled={disabled || rotationBlocked} onClick={() => numericRotate(rotation)}>회전 적용</button>
        <div className="nudge"><span>1° 회전</span>{['X', 'Y', 'Z'].map((a, i) => <div key={a}><button aria-label={`${a} 마이너스 1도`} disabled={disabled || rotationBlocked} onClick={() => nudgeRotation(i, -1)}>−</button><span>{a}</span><button aria-label={`${a} 플러스 1도`} disabled={disabled || rotationBlocked} onClick={() => nudgeRotation(i, 1)}>+</button></div>)}</div>
        <p className="hint">회전 링을 드래그하세요. 1° 버튼은 선택한 축 좌표계를 사용합니다. 숫자 입력은 월드 XYZ 순서의 Euler 각도입니다. {rotationBlocked ? '회전할 발 또는 그룹에 포함된 부위의 고정을 해제하세요.' : '단일 손·골반은 위치를 고정한 채 회전할 수 있습니다.'}</p>
      </> : <>
        <label className="inspector-label">{mirrorActive ? '미러 기준 부위 목표 위치' : members.length > 1 ? '그룹 중심 목표 위치' : '목표 위치'} · 월드 XYZ <small>m</small></label>
        <div className="xyz">{target.map((v, i) => <label key={i}><span className={`axis-${i}`}>{'XYZ'[i]}</span><input aria-label={`목표 ${'XYZ'[i]} 위치`} type="number" step="0.01" value={Number(v.toFixed(4))} disabled={disabled || selectionPinned} onChange={e => setTarget(t => t.map((n, j) => i === j ? +e.target.value : n))}/></label>)}</div>
        <button className="wide" disabled={disabled || selectionPinned} onClick={() => numericMove(target)}>목표 위치 적용</button>
        <div className="nudge"><span>1cm · 월드</span>{['X', 'Y', 'Z'].map((a, i) => <div key={a}><button aria-label={`${a} 마이너스 1cm`} disabled={disabled || selectionPinned} onClick={() => numericMove(selectionPosition(state!, members, selected).map((n, j) => i === j ? n - .01 : n))}>−</button><span>{a}</span><button aria-label={`${a} 플러스 1cm`} disabled={disabled || selectionPinned} onClick={() => numericMove(selectionPosition(state!, members, selected).map((n, j) => i === j ? n + .01 : n))}>+</button></div>)}</div>
        <p className="hint">고정된 부위를 이동하려면 자물쇠를 해제하세요. {isJointHandle(selected) && 'W는 관절 중심의 위치를 IK로 이동합니다. 해당 관절 자체의 회전각을 바꾸려면 E를 누르세요.'}</p>
      </>}
      <div className={`solver-card ${info && !info.converged ? 'warn' : ''}`}><div>{solving ? <span className="spinner"/> : info && !info.converged ? <AlertCircle size={15}/> : <Check size={15}/>} {solving ? 'IK 계산 중' : info ? info.converged ? '목표 도달' : '목표에 도달하지 못함' : '편집 준비 완료'}</div><dl><dt>{members.length > 1 ? '최대 목표 오차' : '목표 오차'}</dt><dd>{info ? info.target_error_mm.toFixed(2) : '—'} mm</dd><dt>고정 오차</dt><dd>{info ? info.pin_error_mm.toFixed(3) : '—'} mm</dd><dt>회전·각도 오차</dt><dd>{info?.angle_error_deg?.toFixed(3) ?? '—'}°</dd></dl></div>
      <div className="section-divider"/>
      <SceneObjectControls objects={objects} selectedId={selectedObjectId} mode={objectTransformMode} disabled={disabled} onSelect={selectObject} onAdd={addObject} onRemove={removeObject} onChange={changeObject} onModeChange={changeObjectMode}/>
      <div className="section-divider"/>
      <div className="panel-heading"><span>최근 저장한 프로젝트</span><small>서버 저장</small></div>
      <select aria-label="최근 저장한 프로젝트" value={savedChoice} onChange={e => setSavedChoice(e.target.value)}><option value="">최근 프로젝트 선택</option>{saved.map(name => <option key={name} value={name}>{name}</option>)}</select>
      <button className="wide" disabled={disabled || !savedChoice} onClick={() => void run(async () => loadProject(await api<Project>(`files/${fileApiPath(savedChoice)}`)))}>최근 프로젝트 열기</button>
      <p className="hint"><code>motions/</code>에 저장한 편집 프로젝트를 빠르게 다시 엽니다. 상단 열기는 컴퓨터의 임의 파일을 선택합니다.</p>
      <label className="hint"><input type="checkbox" checked={exportProto} disabled={disabled} onChange={e => setExportProto(e.target.checked)}/> 저장 시 ProtoMotions .motion / .pt 추가</label>
      <p className="hint">유지 자세는 같은 키프레임을 복제해 시간을 지정하세요.</p>
      {files.map(name => <a className="download" key={name} href={`/api/files/${fileApiPath(name)}`} download><Download size={13}/>{name}</a>)}
      <details className="joint-editor"><summary>29개 관절각 정밀 조정 {jointDirty ? '· 변경 대기' : ''}</summary>
        <p className="hint">슬라이더나 숫자를 수정한 뒤 적용하세요. 발 등 고정 조건을 유지하며 IK를 풉니다. 달성하지 못한 각도는 오차로 표시합니다.</p>
        <button className="wide primary" disabled={disabled || !jointDirty} onClick={applyJointDraft}>변경 각도 적용</button>
        <button className="wide" disabled={disabled || !jointDirty} onClick={() => setJointDraft(state!.qpos.slice(7).map(v => v*180/Math.PI))}>입력 취소</button>
        <div>{project?.joint_names.map((name, i) => <div className="joint-angle-row" key={name}><label htmlFor={`joint-${i}`}>{name.replace('_joint', '')}</label><div><input aria-label={`${name} 슬라이더`} type="range" min={limits[i]?.[0] ?? -180} max={limits[i]?.[1] ?? 180} step=".1" value={jointDraft[i] ?? 0} disabled={disabled} onChange={e => changeJoint(i, +e.target.value)}/><input id={`joint-${i}`} aria-label={`${name} 각도`} type="number" step=".1" min={limits[i]?.[0]} max={limits[i]?.[1]} value={Number((jointDraft[i] ?? 0).toFixed(2))} disabled={disabled} onChange={e => changeJoint(i, +e.target.value)}/><span>°</span></div></div>)}</div>
      </details>
    </aside>
    <section className="timeline">
      <div className="timeline-header"><div className="timeline-title">KEYFRAMES <span>{motionClip ? `1 clip · ${motionClip.length} frames` : `${project?.keyframes.length ?? 0} poses`} · {duration.toFixed(1)}s</span></div><div className="timeline-actions">{poseDirty && <span className="dirty-tag">편집 자세 · 반영 필요</span>}<button disabled={disabled || !activeFrame || !!motionClip} onClick={() => { editFrame(frameIndex, { qpos: [...state!.qpos], pins: [...pins] }); setPoseDirty(false); setMessage('선택한 키프레임에 현재 자세를 반영했습니다.'); }}>선택 프레임에 반영</button><button disabled={disabled || !!motionClip} className="primary" onClick={addFrame}><Plus size={14}/>자세 추가</button></div></div>
      <div className="timeline-body"><div className="transport"><button className="play" title={playing ? '일시정지' : physicsEnabled ? `${controllerLabel} 물리 재생` : '모션 재생'} disabled={!state || busy || solving} onClick={() => void play()}>{playing ? <Pause size={21}/> : <Play size={21}/>}</button><span>{(preview?.time[sample] ?? 0).toFixed(2)}<small> / {(preview?.summary?.reference_seconds ?? duration).toFixed(2)}s</small></span><label><select aria-label="출력 FPS" value={fps} disabled={disabled} onChange={e => setFps(+e.target.value)}>{[15, 30, 50, 60, 100, 120].map(f => <option key={f} value={f}>{f} fps</option>)}</select></label>
        <div className="simulation-toggles">
        <button className="policy-toggle" type="button" aria-pressed={physicsEnabled} disabled={!state || busy || solving || playing || !physicsAvailable} onClick={() => changeSimulation(!physicsEnabled, false)} title={!physicsAvailable ? '물리 재생을 사용하려면 서버를 업데이트하고 재시작하세요.' : 'ON: 중력·접촉·관절 토크 계산 · OFF: 정책도 끄고 원본 편집으로 돌아가기'}>
          <span className="policy-indicator"/>Physics <b>{physicsEnabled ? 'ON' : 'OFF'}</b>
        </button>
        <button className="policy-toggle" type="button" aria-pressed={policyEnabled} disabled={!state || busy || solving || playing || !physicsEnabled || !policyAvailable} onClick={() => changeSimulation(true, !policyEnabled)} title={!physicsEnabled ? 'Physics를 먼저 켜세요.' : !policyAvailable ? 'SONIC CPU 환경 또는 서버 업데이트를 확인하세요.' : 'ON: GEAR-SONIC 정책 추종 · OFF: 물리는 유지하고 기본 PD 관절 제어 사용'}>
          <span className="policy-indicator"/>GEAR-SONIC <b>{policyEnabled ? 'ON' : 'OFF'}</b>
        </button>
        </div>
      </div>
        <div className="frame-track">{project?.keyframes.map((f, i) => <button disabled={disabled} key={i} className={`frame-card ${frameIndex === i ? 'active' : ''}`} onClick={() => void chooseFrame(i)}><span className="frame-number">{String(i+1).padStart(2, '0')}</span><div><strong>{f.name}</strong><small>{f.samples ? `${f.samples.length} 프레임 클립 · ${f.duration.toFixed(2)}s` : i === 0 ? '시작 자세' : `${f.duration.toFixed(1)}s 이동`} · {f.pins.length} 고정</small></div><div className="mini-pose"><i style={{ height: `${22 + (f.qpos[2] - .5) * 40}px` }}/></div></button>)}</div>
        <div className="frame-edit">{activeFrame && <><input aria-label="키프레임 이름" value={activeFrame.name} disabled={disabled} onChange={e => editFrame(frameIndex, { name: e.target.value })}/><div><label>{motionClip ? '클립 재생 시간' : '이동 시간'} <input aria-label="키프레임 이동 시간" type="number" min={motionClip ? 1/120 : .1} max={motionClip ? 600 : 60} step=".01" disabled={disabled || (!motionClip && frameIndex === 0)} value={activeFrame.duration} onChange={e => { const minimum = motionClip ? 1/120 : .1; const maximum = motionClip ? 600 : 60; editFrame(frameIndex, { duration: Math.max(minimum, Math.min(maximum, +e.target.value || minimum)) }); }}/>s</label><button title="이전으로 이동" disabled={disabled || frameIndex === 0} onClick={() => { const frames = [...project!.keyframes]; [frames[frameIndex-1], frames[frameIndex]] = [frames[frameIndex], frames[frameIndex-1]]; setProject({ ...project!, keyframes: frames }); setFrameIndex(frameIndex-1); invalidate(); }}><ChevronLeft size={14}/></button><button title="다음으로 이동" disabled={disabled || frameIndex === project!.keyframes.length-1} onClick={() => { const frames = [...project!.keyframes]; [frames[frameIndex+1], frames[frameIndex]] = [frames[frameIndex], frames[frameIndex+1]]; setProject({ ...project!, keyframes: frames }); setFrameIndex(frameIndex+1); invalidate(); }}><ChevronRight size={14}/></button><button title="키프레임 삭제" disabled={disabled || project!.keyframes.length < 2} onClick={() => { setProject({ ...project!, keyframes: project!.keyframes.filter((_, i) => i !== frameIndex) }); setFrameIndex(Math.max(0, frameIndex-1)); invalidate(); }}><Trash2 size={14}/></button></div></>}</div>
      </div>
      <input className="scrubber" aria-label="모션 시간 탐색" type="range" min="0" max={Math.max(1, (preview?.states.length ?? 1)-1)} value={sample} disabled={!preview || busy} onChange={e => { setPlaying(false); const i = +e.target.value; setSample(i); if (preview) { applyState(preview.states[i]); if (preview.object_states?.[i]) scene.current?.setObjectPoses(preview.object_states[i]); } }}/>
    </section>
    <footer className={`statusbar ${error ? 'error' : ''}`}><span>{error ? <AlertCircle size={13}/> : <span className="status-dot"/>}{error || message}</span><span>{physicsEnabled ? `MuJoCo + ${controllerLabel} · ${preview?.physics ? '시뮬레이션 결과' : 'CPU 재생 준비'}` : '원본 모션 · 물리 OFF'}</span></footer>
  </div>;
}
