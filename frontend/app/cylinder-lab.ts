// 2-D lab page: draw (or auto-plan) a TCP path in the cylinder cross-section and check it with Box2D.
import { restingScene, simulateScoop, NSE_SCOOP_PROFILE, DEFAULT_SOFTNESS, type BladePath, type BladeProfile, type Scene, type SimResult, type SoftShape, type Waypoint } from '../lib/cylinder-sim';
import { autoSteps, drawnWaypoints, movePoint, pointAngles, snapToCircle, waypointsToPoints, type DrawnPoint, type DrawnStepping } from '../lib/drawn-path';
import DEMOS from '../lib/cylinder-demos.json';
import { termFor } from '../lib/lab-glossary';
import { EDITOR_GRASP_KEY, editorGraspCoefficients, sectionGrasps, type SectionGrasp } from '../lib/lab-grasp';
import { DEFAULT_RAMEN_GRASP_COEFFICIENTS, type GraspPointCoefficients, type RamenGraspCoefficients } from '../lib/scene-objects';
import { pileAfter, retarget, type Attempt } from '../lib/lab-batch';
import { boxPitchDeg, tiltScene } from '../lib/box-tilt';

type Demo = { id: string; name: string; description: string; source: string; params: Record<string, number>; base_scene: LabScene;
  blades: BladePath[]; stage_steps: Record<string, number>;
  options: { soft?: { mmPer10N: number; scope: 'neighbors' | 'all'; shape: SoftShape }; friction?: Record<string, number> } };

type LabScene = Scene & { toward_robot?: number[]; plane_y?: number; robot_position?: number[]; robot_quaternion_xyzw?: number[] };
type AutoPath = { waypoints: Waypoint[]; stage_steps: Record<string, number>; blades: BladePath[] };
type ToolState = { model_id: string; tools: [string, string] | null; options: { id: string; label: string }[] };
type ToolSection = { side: 'left' | 'right'; tool_id: string; section_y_m: number; outlines: number[][][]; fixtures: number[][][]; area_m2: number };
type Stored = { project?: string; mode?: string; params?: Record<string, number>; stepping?: DrawnStepping;
  tools?: [string, string];
  selectedTool?: string;
  legacyDrawingTool?: string;
  drawings?: Record<string, { points: DrawnPoint[]; flip: boolean; source?: { center: number[]; radius: number; planeY: number;
    far?: number[]; up?: number[]; boxAligned?: boolean } }>;
  pathTemplates?: Record<string, string>;
  pathOverrides?: Record<string, boolean>;
  toolPaths?: Record<string, { points: DrawnPoint[]; flip: boolean; source: { center: number[]; radius: number;
    planeY: number; far: number[]; up: number[]; boxAligned: boolean }; origin?: { project: string; target: string } }>;
  physics?: { material?: string; softMm?: number; scope?: string; shape?: string; force?: number; tracking?: number;
    muCylinder?: number; muBlade?: number; muWall?: number; rolling?: number; mass?: string } };
const $ = <T = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
const input = (id: string) => $<HTMLInputElement>(id);
const num = (id: string) => Number(input(id).value);
const STORAGE_KEY = 'motioncreator-cylinder-lab-v2';
const PARAM_LABELS: Record<string, [string, string?]> = {
  mode: ['방식', '0 = 둘레 감기, 1 = 지렛대, 2 = 양쪽 집기(양손)'], side: ['진입 쪽', '-1 = 로봇 쪽, +1 = 먼 쪽'],
  clearance_m: ['파지점 거리 (m)'], height_m: ['파지점 높이 (m)'], pitch_deg: ['파지점 피치 (°)'],
  approach_height_m: ['틈 위 대기 높이 (m)'], wedge_m: ['쐐기 깊이 (m)', '날 윗면이 원통 표면 안쪽으로'],
  insert_depth: ['삽입 깊이 (×r)', '원통 중심 아래로'], pivot_up: ['지렛대 받침점 (×r)', '원통 중심 위로'], pry_deg: ['젖히는 각도 (°)'],
  wrap_start_deg: ['감기 시작 (°)'], wrap_end_deg: ['감기 끝 (°)'], slide_extent: ['밀어 넣기 (×r)'],
  tilt_deg: ['날 끝 들기 (°)'], tilt_rise_m: ['들기 상승 (m)'], lift_margin_m: ['인양 여유 (m)'], carry_m: ['몸쪽 이동 (m)'],
  step_m: ['한 걸음 거리 (m)'], step_deg: ['한 걸음 각도 (°)'], speed_m_s: ['속도 (m/s)'], turn_deg_s: ['회전 속도 (°/s)'], min_step_s: ['단계 최소 시간 (s)'],
};

let project: Record<string, unknown> | null = null;
let projectName = '';
let baseScene: LabScene | null = null;
let originalPitch: number | null = null; // the box pitch as loaded, for "원래 기울기로"
let scene: LabScene | null = null;
let loadedTargetDiameter: number | null = null;
let defaults: Record<string, number> = {};
let limits: Record<string, [number, number]> = {};
let toolModelId = 'g1';
let mountedTools: [string, string] | null = null;
let toolOptions: ToolState['options'] = [];
let activeToolId = 'scoop';
const toolProfiles: Partial<Record<'left' | 'right', BladeProfile>> = {};
let mode: 'draw' | 'auto' = 'draw';
let points: DrawnPoint[] = [];
let selected = -1;
let usingSharedPath = false;
let autoPath: AutoPath | null = null;
let simBlades: BladePath[] = [];
let simProfiles: BladeProfile[] = [];
let result: SimResult | null = null;
let frame = 0, playing = false, lastTick = 0;
// Path generation by picking cylinders: the reference path, the pile left in the box, the attempts so far,
// the one being replayed and the worker that simulates them.
let pickMode = false;
// Drawn paths are carried over (with the lift stretched to each cylinder's depth); auto plans are re-planned.
let template: { kind: 'drawn'; paths: BladePath[]; referenceCenter: number[]; referenceRimUp: number }
  | { kind: 'auto'; params: Record<string, number> } | null = null;
let pile: LabScene | null = null; // null: the full box
let pileNeedsSettle = false;
let attempts: Attempt[] = [];
let viewing: Attempt | null = null;
let worker: Worker | null = null;
let simulating = false;
let hoverId: string | null = null;
let targetPickMode = false;
const view = { cx: 0, cz: 0, size: .9 };

// ---------------------------------------------------------------- storage

function stored(): Stored { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { return {}; } }
function remember(values: Stored) { try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...stored(), ...values })); } catch { /* optional */ } }
const legacyDrawKey = () => `${projectName}|${scene?.target_id ?? ''}`;
const drawKey = () => `${activeToolId}|${legacyDrawKey()}`;
const templateKey = () => `${activeToolId}|${projectName}`;
const sharedTargetId = () => stored().pathTemplates?.[templateKey()];

// ---------------------------------------------------------------- saved form sections ("저장" next to a heading)

const SAVED_KEY = 'motioncreator-cylinder-lab-saved-v1';
const TOOL_SAVED_KEY = 'motioncreator-cylinder-lab-tool-settings-v1';
const TOOL_GRASP_KEY = 'motioncreator-cylinder-lab-tool-grasps-v1';
const SAVE_GROUPS: Record<string, string[]> = {
  grasp: ['show-grasp', 'snap', 'snap-wedge', 'flip'],
  stepping: ['step-m', 'step-deg', 'speed', 'turn', 'min-step'],
  physics: ['material', 'soft-shape', 'soft-mm', 'soft-scope', 'mass', 'force', 'tracking'],
  friction: ['mu-cylinder', 'mu-blade', 'mu-wall', 'rolling'],
};
type SavedSections = Partial<Record<string, Record<string, string | boolean | number>>>;
function storedMap<T>(key: string): Record<string, T> {
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}
function legacyToolId() {
  if (stored().legacyDrawingTool) return stored().legacyDrawingTool!;
  const old = storedMap<Record<string, string>>(SAVED_KEY).grasp;
  const side = old?.['grasp-side'] === 'right' ? 'right' : 'left';
  const pair = stored().tools ?? mountedTools ?? ['scoop', 'end_support'];
  return pair[side === 'left' ? 0 : 1];
}
function savedSections(): SavedSections {
  return storedMap<SavedSections>(TOOL_SAVED_KEY)[activeToolId] ?? {};
}
function migrateLegacySections() {
  try {
    const toolId = legacyToolId();
    if (!stored().legacyDrawingTool) remember({ legacyDrawingTool: toolId });
    if (localStorage.getItem(TOOL_SAVED_KEY) !== null) return;
    const old = storedMap<Record<string, string | boolean | number>>(SAVED_KEY);
    localStorage.setItem(TOOL_SAVED_KEY, JSON.stringify(Object.keys(old).length ? { [toolId]: old } : {}));
  } catch { /* Browser storage is optional. */ }
}
const DEFAULT_GROUPS: SavedSections = {};
function readSection(group: string): Record<string, string | boolean | number> {
  return group === 'auto' ? readParams()
    : Object.fromEntries(SAVE_GROUPS[group].map(id => { const el = input(id); return [id, el.type === 'checkbox' ? el.checked : el.value]; }));
}
function persistSection(group: string) {
  const map = storedMap<SavedSections>(TOOL_SAVED_KEY);
  map[activeToolId] = { ...savedSections(), [group]: readSection(group) };
  try { localStorage.setItem(TOOL_SAVED_KEY, JSON.stringify(map)); }
  catch { status('브라우저 저장소를 쓸 수 없어 설정을 저장하지 못했습니다.'); }
}

/** Store the section's current inputs; the page starts from them next time it opens. */
function saveSection(group: string, button: HTMLElement) {
  persistSection(group);
  button.textContent = '저장됨'; button.classList.add('saved');
  setTimeout(() => { button.textContent = '저장'; button.classList.remove('saved'); }, 1500);
  status(`${toolLabel(activeToolId)} 설정을 저장했습니다.`);
}

/** Put saved section values back into the form (the auto parameters are applied in renderParams). */
function restoreSections() {
  const saved = savedSections();
  for (const [group, ids] of Object.entries(SAVE_GROUPS)) {
    const values = { ...DEFAULT_GROUPS[group], ...saved[group] };
    for (const id of ids) {
      if (!(id in values)) continue;
      const el = input(id);
      if (el.type === 'checkbox') el.checked = Boolean(values[id]); else el.value = String(values[id]);
    }
  }
  for (const id of ['material', 'soft-scope']) input(id).dispatchEvent(new Event('change'));
  if (Object.keys(defaults).length) renderParams({ ...defaults, ...saved.auto } as Record<string, number>);
}
function saveDrawing() {
  if (!scene) return;
  const saved = stored();
  const drawings = saved.drawings ?? {};
  drawings[drawKey()] = { points, flip: input('flip').value === '1', source: {
    center: [...scene.target.center], radius: scene.target.radius, planeY: scene.plane_y ?? 0,
    far: [...scene.frame.far], up: [...scene.frame.up], boxAligned: !!scene.box,
  } };
  if (usingSharedPath) {
    remember({ drawings, pathOverrides: { ...saved.pathOverrides, [drawKey()]: true } });
    usingSharedPath = false;
  } else remember({ drawings });
  renderPathTemplateInfo();
}
function loadDrawing() {
  const settings = stored(), drawings = settings.drawings ?? {};
  const global = settings.toolPaths?.[activeToolId];
  const useGlobal = !!scene && !!global && !settings.pathOverrides?.[drawKey()];
  const sourceId = sharedTargetId();
  const useShared = !useGlobal && !!scene && !!sourceId && sourceId !== scene.target_id && !settings.pathOverrides?.[drawKey()];
  const source = useShared ? drawings[`${activeToolId}|${projectName}|${sourceId}`] : undefined;
  const center = source?.source?.center ?? scene?.circles.find(circle => circle.id === sourceId)?.center;
  const saved = useGlobal ? global : source && center && scene ? source : drawings[drawKey()]
    ?? (activeToolId === legacyToolId() ? drawings[legacyDrawKey()] : undefined);
  usingSharedPath = useGlobal || !!(source && center && scene);
  const offset = source && center && scene ? [scene.target.center[0] - center[0], scene.target.center[1] - center[1]] : [0, 0];
  points = Array.isArray(saved?.points) ? saved.points.map(point => {
    if (useGlobal && global && scene) {
      const dx = point.x - global.source.center[0], dz = point.z - global.source.center[1];
      const far = dx * global.source.far[0] + dz * global.source.far[1];
      const up = dx * global.source.up[0] + dz * global.source.up[1];
      const directionFar = point.angle === null ? 0 : Math.cos(point.angle) * global.source.far[0] + Math.sin(point.angle) * global.source.far[1];
      const directionUp = point.angle === null ? 0 : Math.cos(point.angle) * global.source.up[0] + Math.sin(point.angle) * global.source.up[1];
      return { ...point, x: scene.target.center[0] + far * scene.frame.far[0] + up * scene.frame.up[0],
        z: scene.target.center[1] + far * scene.frame.far[1] + up * scene.frame.up[1],
        angle: point.angle === null ? null : Math.atan2(directionFar * scene.frame.far[1] + directionUp * scene.frame.up[1],
          directionFar * scene.frame.far[0] + directionUp * scene.frame.up[0]) };
    }
    return { ...point, x: point.x + offset[0], z: point.z + offset[1] };
  }) : [];
  if (saved) input('flip').value = saved.flip ? '1' : '0';
  selected = points.length - 1;
  if (saved && !saved.source && points.length && !usingSharedPath) saveDrawing();
  renderPathTemplateInfo();
}
function renderPathTemplateInfo() {
  const info = $('path-template-info');
  if (!info) return;
  const global = stored().toolPaths?.[activeToolId];
  const sourceId = sharedTargetId();
  const sourceName = global?.origin?.target ?? scene?.circles.find(circle => circle.id === sourceId)?.name ?? sourceId;
  const hasTemplate = !!global || !!sourceId && !!scene && sourceId !== scene.target_id;
  $('detach-path').hidden = !hasTemplate || !usingSharedPath;
  $('restore-individual').hidden = !usingSharedPath || !stored().drawings?.[drawKey()];
  $('reset-to-shared').hidden = !hasTemplate || usingSharedPath;
  $<HTMLButtonElement>('share-path').disabled = !scene || points.length < 2;
  info.textContent = usingSharedPath ? `${sourceName ?? '기준 용기'}의 ${global ? '툴 공통 파지법' : '장면 공통 경로'}을 이 용기 위치에 맞춰 표시 중입니다. 점을 고치면 이 장면의 용기만 개별 경로가 됩니다.`
    : hasTemplate ? `${sourceName}의 공통 경로와 별도로 저장된 이 용기의 개별 경로입니다.`
      : sourceId === scene?.target_id ? '현재 용기가 공통 경로 기준입니다. 이 경로를 고치면 연결된 용기에도 반영됩니다.'
        : '현재 용기의 경로를 다른 용기에 공통으로 적용할 수 있습니다. 바닥·벽 근처는 별도 검증하세요.';
}
function sharePath() {
  if (!scene || points.length < 2) return;
  const saved = stored();
  const overrides = { ...saved.pathOverrides };
  delete overrides[drawKey()];
  remember({ toolPaths: { ...saved.toolPaths, [activeToolId]: { points: structuredClone(points),
    flip: input('flip').value === '1', source: { center: [...scene.target.center], radius: scene.target.radius,
      planeY: scene.plane_y ?? 0, far: [...scene.frame.far], up: [...scene.frame.up], boxAligned: !!scene.box },
    origin: { project: projectName, target: scene.circles.find(circle => circle.id === scene!.target_id)?.name ?? scene.target_id } } },
    pathOverrides: overrides });
  usingSharedPath = true;
  renderPathTemplateInfo();
  status(`${toolLabel(activeToolId)} 공통 파지법을 저장했습니다. 장면을 바꿔도 사용할 수 있습니다. 용기별 충돌을 확인하세요.`);
}
function detachPath() {
  if (!usingSharedPath) return;
  saveDrawing();
  status('현재 용기의 경로를 개별 저장했습니다. 이후 공통 경로를 바꿔도 유지됩니다.');
}
function restoreIndividualPath() {
  if (!scene || !stored().drawings?.[drawKey()]) return;
  const saved = stored();
  remember({ pathOverrides: { ...saved.pathOverrides, [drawKey()]: true } });
  loadDrawing(); resetHistory(); clearResult(); renderPoints(); draw();
  status('이 장면에 저장돼 있던 개별 경로를 다시 불러왔습니다.');
}
function resetToSharedPath() {
  if (!scene || !stored().toolPaths?.[activeToolId] && !sharedTargetId()) return;
  const saved = stored(), overrides = { ...saved.pathOverrides };
  delete overrides[drawKey()];
  remember({ pathOverrides: overrides });
  loadDrawing(); resetHistory(); clearResult(); renderPoints(); draw();
  status('현재 용기를 공통 경로에 다시 연결했습니다.');
}

// ---------------------------------------------------------------- API and scene

async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${url}`, body === undefined ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({})) as { detail?: unknown };
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `요청 실패 (${response.status})`);
  return data as T;
}
const status = (text: string) => { $('status').textContent = text; };

const settled = (source: LabScene): LabScene => restingScene(source);

async function loadProject(name: string) {
  projectName = name;
  project = await api(`project/${name.split('/').map(encodeURIComponent).join('/')}`);
  remember({ project: name });
  await loadScene(undefined);
}
async function loadScene(targetId: string | undefined, preservePitch = false) {
  setTargetPickMode(false);
  const currentPitch = preservePitch && baseScene ? boxPitchDeg(baseScene) : null;
  const response = await api<{ scene: LabScene; defaults: Record<string, number>; limits: Record<string, [number, number]>; targets: { id: string; name: string }[] }>(
    'cylinder-lab/scene', { project, target_id: targetId ?? null });
  const loadedPitch = boxPitchDeg(response.scene);
  baseScene = currentPitch !== null && loadedPitch !== null ? tiltScene(response.scene, currentPitch).scene : response.scene;
  defaults = response.defaults; limits = response.limits;
  syncCylinderDiameter();
  $<HTMLSelectElement>('target').innerHTML = response.targets.map(t => `<option value="${t.id}">${t.name}</option>`).join('');
  $<HTMLSelectElement>('target').value = baseScene.target_id;
  renderParams({ ...defaults, ...Object.fromEntries(Object.entries(savedSections().auto ?? {}).filter(([key]) => key in defaults)) as Record<string, number> });
  restoreSections();
  syncPitch(loadedPitch);
  status('정착 중…');
  await new Promise(resolve => setTimeout(resolve, 10));
  scene = settled(baseScene);
  view.cx = scene.frame.origin[0]; view.cz = scene.frame.origin[1] + .05; view.size = .9;
  loadDrawing(); resetHistory();
  autoPath = null; clearResult(); if (pickMode) setPickMode(false); pile = null; attempts = []; renderAttempts();
  renderGraspInfo(); renderPoints(); draw(); status('');
}

/** Reset the Lab-only diameter control when a project, target, or demo is loaded. */
function syncCylinderDiameter() {
  loadedTargetDiameter = baseScene?.target.radius ? baseScene.target.radius * 2 : null;
  input('cylinder-diameter').value = loadedTargetDiameter === null ? '' : loadedTargetDiameter.toFixed(3);
  input('cylinder-diameter').disabled = loadedTargetDiameter === null;
  $<HTMLButtonElement>('cylinder-diameter-reset').disabled = loadedTargetDiameter === null;
}

/** Resize only the selected target in this in-memory Lab scene, then settle its new geometry. */
function applyCylinderDiameter(diameter: number) {
  if (!baseScene || !scene || !Number.isFinite(diameter)) return;
  const nextDiameter = Math.max(.01, Math.min(.5, diameter));
  const radius = nextDiameter / 2;
  const target = { ...baseScene.target, radius };
  baseScene = { ...baseScene, target,
    circles: baseScene.circles.map(circle => circle.id === baseScene!.target_id ? { ...circle, radius } : circle) };
  input('cylinder-diameter').value = nextDiameter.toFixed(3);
  if (pickMode) setPickMode(false);
  pile = null; attempts = []; renderAttempts(); autoPath = null;
  clearResult();
  status('변경한 지름으로 원통을 다시 정착시키는 중…');
  scene = settled(baseScene);
  view.cx = scene.frame.origin[0]; view.cz = scene.frame.origin[1] + .05;
  if (points.length) saveDrawing();
  renderGraspInfo(); renderPoints(); draw();
  status(`대상 원통 지름 ${(nextDiameter * 1000).toFixed(0)} mm · 질량 유지 · Lab 장면에 적용했습니다.`);
}

function setTargetPickMode(on: boolean) {
  targetPickMode = on;
  const button = $<HTMLButtonElement>('select-sim-target');
  button.setAttribute('aria-pressed', String(on));
  button.textContent = on ? '용기를 클릭하세요…' : '캔버스에서 용기 선택';
  if (on) status('시뮬레이션할 원통을 캔버스에서 클릭하세요.');
  draw();
}

// ---------------------------------------------------------------- frame helpers

function frameAxes() {
  const s = scene!;
  return { c: s.frame.origin, far: s.frame.far, up: s.frame.up, farAngle: Math.atan2(s.frame.far[1], s.frame.far[0]) };
}
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
function toLocal(x: number, z: number) {
  const { c, far, up } = frameAxes();
  return [(x - c[0]) * far[0] + (z - c[1]) * far[1], (x - c[0]) * up[0] + (z - c[1]) * up[1]];
}
function stepping(): DrawnStepping {
  return { stepM: Math.max(.002, num('step-m')), stepDeg: Math.max(.5, num('step-deg')), speedMS: Math.max(.005, num('speed')),
    turnDegS: Math.max(1, num('turn')), minStepS: Math.max(.02, num('min-step')) };
}
const snapRadius = () => scene!.target.radius - num('snap-wedge');

// ---------------------------------------------------------------- panels

function renderParams(values: Record<string, number>) {
  $('params').innerHTML = Object.keys(defaults).map(key => {
    const [label, note] = PARAM_LABELS[key] ?? [key];
    const [low, high] = limits[key] ?? [-1e9, 1e9];
    const fromGrasp = GRASP_PARAMS.includes(key);
    return `<label class="row">${label}${fromGrasp ? '<small>파지점 칸의 계수를 씀</small>' : note ? `<small>${note}</small>` : ''}<input data-key="${key}" type="number" step="any" min="${low}" max="${high}" value="${values[key]}" ${fromGrasp ? 'disabled' : ''}/></label>`;
  }).join('');
  decorateTerms($('params'));
  if (scene) renderGraspInfo();
}
/** Planner parameters with the grasp point taken from the chosen hand's editor coefficients. */
function plannerParams() {
  const c = toolCoefficients(activeToolId);
  return { ...readParams(), clearance_m: c.clearance, height_m: c.height, pitch_deg: c.pitch };
}
const GRASP_PARAMS = ['clearance_m', 'height_m', 'pitch_deg'];
const readParams = () => Object.fromEntries([...document.querySelectorAll<HTMLInputElement>('#params input')].map(i => [i.dataset.key!, Number(i.value)]));

function renderPoints() {
  $('point-count').textContent = points.length ? `(${points.length})` : '';
  if (!scene) return;
  const angles = pointAngles(points);
  const { farAngle } = frameAxes();
  $('points').innerHTML = points.map((p, i) => {
    const [a, b] = toLocal(p.x, p.z);
    const local = Math.round(wrap(angles[i] - farAngle) * 180 / Math.PI);
    const auto = i > 0 ? autoSteps(points, i, stepping()) : '';
    return `<div class="point ${i === selected ? 'selected' : ''}" data-i="${i}"><span class="drag-handle" draggable="true" title="끌어서 순서 바꾸기" aria-label="${i + 1}번 점 순서 바꾸기">⋮⋮</span><b>${i + 1}</b><span class="xy">${(a * 1000).toFixed(0)} / ${(b * 1000).toFixed(0)}</span>`
      + `<input data-field="angle" type="number" step="1" placeholder="자동 ${local}" value="${p.angle === null ? '' : local}"/>`
      + `<input data-field="steps" type="number" min="1" step="1" ${i === 0 ? 'disabled' : ''} placeholder="${auto}" value="${p.steps ?? ''}"/>`
      + `<button data-field="delete" title="삭제">✕</button></div>`;
  }).join('');
  const waypoints = mode === 'draw' ? drawnWaypoints(points, stepping()) : autoPath?.waypoints ?? [];
  $('steps').innerHTML = waypoints.map((w, i) => `<div data-i="${i}"><span>${i}. ${w.label}</span><span>${w.duration.toFixed(2)}s</span></div>`).join('');
}

function editPoint(index: number, field: string, value: string) {
  const point = points[index];
  if (!point) return;
  checkpoint();
  if (field === 'delete') { points.splice(index, 1); selected = Math.min(selected, points.length - 1); }
  else if (field === 'angle') point.angle = value === '' ? null : wrap(Number(value) * Math.PI / 180 + frameAxes().farAngle);
  else if (field === 'steps') point.steps = value === '' ? null : Math.max(1, Math.round(Number(value)));
  changed();
}
function changed() { clearResult(); saveDrawing(); persistSection('grasp'); renderPoints(); draw(); }

// ---------------------------------------------------------------- undo / redo of drawing edits

type Snapshot = { points: DrawnPoint[]; flip: string; selected: number };
const undoStack: Snapshot[] = [], redoStack: Snapshot[] = [];
const snapshot = (): Snapshot => ({ points: structuredClone(points), flip: input('flip').value, selected });
/** Remember the drawing before an edit so Ctrl+Z can bring it back. */
function checkpoint() {
  undoStack.push(snapshot());
  if (undoStack.length > 200) undoStack.shift();
  redoStack.length = 0;
}
function restore(state: Snapshot) {
  points = structuredClone(state.points); input('flip').value = state.flip; selected = Math.min(state.selected, points.length - 1);
  changed();
}
function undo() {
  const previous = undoStack.pop();
  if (!previous) { status('더 되돌릴 편집이 없습니다.'); return; }
  redoStack.push(snapshot()); restore(previous);
  status(`되돌렸습니다 (남은 되돌리기 ${undoStack.length}).`);
}
function redo() {
  const next = redoStack.pop();
  if (!next) { status('다시 실행할 편집이 없습니다.'); return; }
  undoStack.push(snapshot()); restore(next);
  status(`다시 실행했습니다 (남은 다시 실행 ${redoStack.length}).`);
}
function resetHistory() { undoStack.length = 0; redoStack.length = 0; }
function clearResult() {
  result = null; frame = 0; playing = false; viewing = null;
  document.querySelectorAll('.batch-row.viewing').forEach(row => row.classList.remove('viewing'));
  $('result').className = 'result'; $('result').textContent = '아직 실행하지 않았습니다.';
  input('scrub').max = '0';
}

// ---------------------------------------------------------------- running

async function run() {
  if (!scene || !baseScene) return;
  $<HTMLButtonElement>('run').disabled = true;
  try {
    if (mode === 'draw') {
      if (points.length < 2) throw new Error('점을 두 개 이상 찍으세요.');
      simBlades = [{ waypoints: drawnWaypoints(points, stepping()), flip: input('flip').value === '1', side: 'drawn' }];
    } else {
      const params = plannerParams();
      status('궤적 계획 중…');
      const planned = await api<{ scene: LabScene; path: AutoPath }>('cylinder-lab/plan', {
        scene: baseScene, params, positions: Object.fromEntries(scene.circles.map(c => [c.id, c.center])) });
      autoPath = planned.path; simBlades = planned.path.blades;
      renderPoints();
    }
    await simulate(simBlades);
  } catch (error) {
    status(error instanceof Error ? error.message : String(error));
  } finally { $<HTMLButtonElement>('run').disabled = false; }
}

/** The page's physics settings as simulator options. */
function physicsOptions() {
  const soft = $<HTMLSelectElement>('material').value === 'soft'
    ? { ...DEFAULT_SOFTNESS, mmPer10N: Math.max(.5, num('soft-mm')), scope: $<HTMLSelectElement>('soft-scope').value as 'neighbors' | 'all',
      shape: $<HTMLSelectElement>('soft-shape').value as SoftShape } : null;
  const friction = { cylinder: Math.max(0, num('mu-cylinder')), blade: Math.max(0, num('mu-blade')), wall: Math.max(0, num('mu-wall')),
    rolling: Math.max(0, num('rolling')) };
  const mass = input('mass').value.trim() === '' ? null : Math.max(.01, num('mass'));
  return { soft, friction, cylinderMassKg: mass, forceLimitN: num('force'), trackingLimitM: num('tracking') / 1000, frameEvery: 4 };
}

/** Run the physics for `blades` with the page's physics settings and start the replay. */
async function simulate(blades: BladePath[]) {
  if (!scene) return;
  if (!profilesReadyForBlades(blades)) {
    status(blades.length > 1 ? '장착된 두 툴의 실제 단면을 모두 불러온 뒤 시뮬레이션하세요.'
      : '작업 툴을 선택하고 실제 단면을 불러온 뒤 시뮬레이션하세요.');
    return;
  }
  simBlades = blades;
  simProfiles = profilesForBlades(blades);
  $<HTMLButtonElement>('run').disabled = true;
  try {
    const { soft, friction, cylinderMassKg } = physicsOptions();
    status(soft ? '말랑한 원통으로 물리 계산 중… (몇 초 걸립니다)' : '물리 계산 중…');
    await new Promise(resolve => setTimeout(resolve, 20));
    result = simulateScoop(scene, simBlades, simProfiles,
      { forceLimitN: num('force'), trackingLimitM: num('tracking') / 1000, frameEvery: 4, soft, friction, cylinderMassKg });
    frame = 0; input('scrub').max = String(result.frames.length - 1);
    showResult(); status(''); playing = true;
  } catch (error) {
    status(error instanceof Error ? error.message : String(error));
  } finally { $<HTMLButtonElement>('run').disabled = false; }
}

function showResult() {
  if (!result) return;
  const box = $('result');
  box.className = `result ${result.success ? 'ok' : 'fail'}`;
  const row = (label: string, value: string) => `<div class="metric"><span>${label}</span><b>${value}</b></div>`;
  box.innerHTML = `<b>${result.success ? '성공' : '실패'}</b><p class="hint">${result.reason}</p>`
    + row('단계 수', `${result.steps}`) + row('시간', `${result.seconds.toFixed(1)} s`)
    + row('들린 높이', `${(result.liftM * 1000).toFixed(0)} / ${(result.requiredLiftM * 1000).toFixed(0)} mm`)
    + row('최대 힘', `${result.peakForceN.toFixed(0)} N`)
    + row('접촉 중 덜 간 거리', `${(result.stalledM * 1000).toFixed(0)} mm`) + row('끝 자세 오차', `${(result.finalErrorM * 1000).toFixed(0)} mm`)
    + row('이웃 최대 이동', `${(result.neighborMaxM * 1000).toFixed(0)} mm`)
    + (result.maxSquashMm > 0 ? row('최대 찌그러짐', `${result.maxSquashMm.toFixed(1)} mm`) : '');
  decorateTerms(box);
}

// ---------------------------------------------------------------- drawing

function metrics() {
  const canvas = $<HTMLCanvasElement>('view');
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width, rect.height) / view.size;
  return { canvas, rect, scale,
    X: (x: number) => rect.width / 2 + (x - view.cx) * scale, Z: (z: number) => rect.height / 2 - (z - view.cz) * scale,
    world: (px: number, py: number) => ({ x: view.cx + (px - rect.width / 2) / scale, z: view.cz - (py - rect.height / 2) / scale }) };
}
const HANDLE_PX = 38;

function draw() {
  const { canvas, rect, scale, X, Z } = metrics();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0e141b'; ctx.fillRect(0, 0, rect.width, rect.height);
  if (!scene) return;
  const current = result?.frames[frame];
  const shown: LabScene = viewing?.scene ?? pile ?? scene; // an attempt shows its own pile
  // grid every 5 cm in the target frame
  const { c, far, up } = frameAxes();
  ctx.strokeStyle = '#18222c'; ctx.lineWidth = 1;
  for (let g = -1; g <= 1.0001; g += .05) {
    for (const [a, b] of [[far, up], [up, far]]) {
      ctx.beginPath();
      ctx.moveTo(X(c[0] + a[0] * g - b[0]), Z(c[1] + a[1] * g - b[1])); ctx.lineTo(X(c[0] + a[0] * g + b[0]), Z(c[1] + a[1] * g + b[1]));
      ctx.stroke();
    }
  }
  const quad = (center: number[], half: number[], angle: number, fill: string) => {
    ctx.save(); ctx.translate(X(center[0]), Z(center[1])); ctx.rotate(-angle);
    ctx.fillStyle = fill; ctx.fillRect(-half[0] * scale, -half[1] * scale, half[0] * 2 * scale, half[1] * 2 * scale); ctx.restore();
  };
  for (const wall of shown.walls) quad(wall.center, wall.half, wall.angle, '#8a6b4a');
  for (const circle of shown.circles) {
    const fill = circle.id === shown.target_id ? '#e8a33d' : '#56708a';
    const shape = current?.shapes[circle.id];
    if (shape) { // soft: the outline traced over the ring of discs
      ctx.fillStyle = fill;
      ctx.beginPath(); shape.forEach(([x, z], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Z(z))); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = '#0e141b66'; ctx.stroke();
      continue;
    }
    const pose = current?.bodies[circle.id];
    const [x, z, a] = pose ? [pose.x, pose.z, pose.angle] : [circle.center[0], circle.center[1], 0];
    ctx.beginPath(); ctx.arc(X(x), Z(z), circle.radius * scale, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = '#0e141b'; ctx.beginPath(); ctx.moveTo(X(x), Z(z)); ctx.lineTo(X(x + Math.cos(a) * circle.radius), Z(z + Math.sin(a) * circle.radius)); ctx.stroke();
  }
  if ((pickMode || targetPickMode) && hoverId && !current) { // the cylinder under the mouse in a selection mode
    const circle = (pile ?? scene).circles.find(c => c.id === hoverId);
    if (circle) {
      ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 3; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.arc(X(circle.center[0]), Z(circle.center[1]), circle.radius * scale + 4, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  for (const r of shown.rectangles) {
    const pose = current?.bodies[r.id];
    quad(pose ? [pose.x, pose.z] : r.center, r.half, pose ? pose.angle : r.angle, '#56708a');
  }
  const bladeShape = (pose: { x: number; z: number; angle: number }, fill: string | null, stroke: string, dashed: boolean,
    flipped: boolean, profile = activeToolProfile()) => {
    const p = profile, flip = flipped ? -1 : 1;
    ctx.save(); ctx.translate(X(pose.x), Z(pose.z)); ctx.rotate(-pose.angle); ctx.scale(scale, -scale * flip);
    ctx.fillStyle = fill ?? 'transparent';
    ctx.lineWidth = 1.5 / scale; ctx.setLineDash(dashed ? [4 / scale, 3 / scale] : []); ctx.strokeStyle = stroke;
    if (p.outlines?.length) {
      for (const outline of p.outlines) {
        if (outline.length < 3) continue;
        ctx.beginPath(); ctx.moveTo(outline[0][0], outline[0][1]);
        for (const point of outline.slice(1)) ctx.lineTo(point[0], point[1]);
        ctx.closePath();
        if (fill) ctx.fill();
        ctx.stroke();
      }
    } else {
      const tipAhead = p.tipAhead ?? 0, length = p.length ?? 0, thickness = p.thickness ?? 0;
      const heelLength = p.heelLength ?? 0, heelHeight = p.heelHeight ?? 0, back = tipAhead - length;
      ctx.beginPath(); ctx.rect(back, -thickness, length, thickness);
      if (heelLength > 0 && heelHeight > 0) ctx.rect(back - heelLength, -thickness, heelLength, thickness + heelHeight);
      if (fill) ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  };
  const flipped = input('flip').value === '1';
  const grasp = input('show-grasp').checked ? currentGrasp(viewing?.scene ?? scene) : null;
  if (grasp) {
    const px = X(grasp.x), pz = Z(grasp.z);
    const dx = Math.cos(grasp.angle), dz = -Math.sin(grasp.angle); // screen y points down
    const nx = (grasp.flip ? 1 : -1) * dz, nz = (grasp.flip ? -1 : 1) * dx; // top-face side: +90 deg of travel unless flipped
    ctx.strokeStyle = '#55e7c1'; ctx.fillStyle = '#55e7c1'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, pz); ctx.lineTo(px + dx * 34, pz + dz * 34); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px + dx * 34, pz + dz * 34); ctx.lineTo(px + dx * 26 - dz * 5, pz + dz * 26 + dx * 5); ctx.lineTo(px + dx * 26 + dz * 5, pz + dz * 26 - dx * 5); ctx.fill();
    ctx.beginPath(); ctx.moveTo(px, pz); ctx.lineTo(px + nx * 14, pz + nz * 14); ctx.stroke();
    ctx.beginPath(); ctx.arc(px, pz, 5, 0, Math.PI * 2); ctx.lineWidth = 1.5; ctx.stroke();
    ctx.font = '11px sans-serif'; ctx.fillText('파지점', px - 20, pz - 10);
  }
  if (viewing) {
    for (const plan of viewing.paths) {
      ctx.fillStyle = '#d9f'; ctx.strokeStyle = '#d9f5'; ctx.beginPath();
      plan.waypoints.forEach((w, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(w.tcp[0]), Z(w.tcp[1]))); ctx.stroke();
      for (const w of plan.waypoints) { ctx.beginPath(); ctx.arc(X(w.tcp[0]), Z(w.tcp[1]), 2, 0, Math.PI * 2); ctx.fill(); }
    }
  } else if (mode === 'draw') {
    if (input('snap').checked) {
      ctx.setLineDash([3, 4]); ctx.strokeStyle = '#ffd16655'; ctx.beginPath();
      ctx.arc(X(scene.target.center[0]), Z(scene.target.center[1]), Math.max(0, snapRadius()) * scale, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    }
    const waypoints = drawnWaypoints(points, stepping());
    ctx.strokeStyle = '#ffd166aa'; ctx.lineWidth = 1.5; ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(p.x), Z(p.z))); ctx.stroke();
    ctx.fillStyle = '#d9f';
    for (const w of waypoints) { ctx.beginPath(); ctx.arc(X(w.tcp[0]), Z(w.tcp[1]), 2, 0, Math.PI * 2); ctx.fill(); }
    const angles = pointAngles(points);
    if (!current && selected >= 0 && points[selected]) bladeShape({ x: points[selected].x, z: points[selected].z, angle: angles[selected] }, '#80efc533', '#80efc5aa', false, flipped);
    points.forEach((p, i) => {
      const hx = X(p.x) + Math.cos(angles[i]) * HANDLE_PX, hz = Z(p.z) - Math.sin(angles[i]) * HANDLE_PX;
      ctx.strokeStyle = p.angle === null ? '#ffd16677' : '#ffd166'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(p.x), Z(p.z)); ctx.lineTo(hx, hz); ctx.stroke();
      ctx.beginPath(); ctx.arc(hx, hz, 4, 0, Math.PI * 2); ctx.fillStyle = p.angle === null ? '#0e141b' : '#ffd166'; ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(X(p.x), Z(p.z), i === selected ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = i === selected ? '#fff3c4' : '#ffd166'; ctx.fill();
      ctx.fillStyle = '#dce5ee'; ctx.font = '11px sans-serif'; ctx.fillText(String(i + 1), X(p.x) + 8, Z(p.z) - 8);
    });
  } else if (autoPath) {
    for (const plan of autoPath.blades) {
      ctx.fillStyle = '#d9f'; ctx.strokeStyle = '#d9f5'; ctx.beginPath();
      plan.waypoints.forEach((w, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(w.tcp[0]), Z(w.tcp[1]))); ctx.stroke();
      for (const w of plan.waypoints) { ctx.beginPath(); ctx.arc(X(w.tcp[0]), Z(w.tcp[1]), 2, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  if (current) {
    current.blades.forEach((pose, index) => {
      const bladeFlip = !!simBlades[index]?.flip;
      const profile = simProfiles[index] ?? activeToolProfile();
      bladeShape(current.commands[index], null, '#d9f', true, bladeFlip, profile);
      bladeShape(pose, index ? '#6ec3ffaa' : '#80efc5aa', index ? '#6ec3ff' : '#80efc5', false, bladeFlip, profile);
    });
    ctx.fillStyle = '#dce5ee'; ctx.font = '12px sans-serif';
    ctx.fillText(`${current.t.toFixed(2)} s · ${current.stage} · 힘 ${current.forceN.toFixed(0)} N`, 12, 20);
    $('time').textContent = `${current.t.toFixed(2)} s`;
    input('scrub').value = String(frame);
    let elapsed = 1, index = 0;
    const waypoints = simBlades[0]?.waypoints ?? [];
    for (let i = 1; i < waypoints.length && current.t >= elapsed + waypoints[i].duration; i++) { elapsed += waypoints[i].duration; index = i; }
    document.querySelectorAll('#steps div').forEach(node => node.classList.toggle('now', Number((node as HTMLElement).dataset.i) === index + (current.t > 1 ? 1 : 0)));
  }
}

// ---------------------------------------------------------------- canvas input

type Drag = { kind: 'move' | 'angle' | 'pan'; index: number; startX: number; startY: number; cx: number; cz: number };
let drag: Drag | null = null;

function hit(px: number, py: number) {
  const { X, Z } = metrics();
  const angles = pointAngles(points);
  for (let i = points.length - 1; i >= 0; i--) {
    const hx = X(points[i].x) + Math.cos(angles[i]) * HANDLE_PX, hz = Z(points[i].z) - Math.sin(angles[i]) * HANDLE_PX;
    if (Math.hypot(px - hx, py - hz) < 7) return { kind: 'angle' as const, index: i };
  }
  for (let i = points.length - 1; i >= 0; i--) {
    if (Math.hypot(px - X(points[i].x), py - Z(points[i].z)) < 9) return { kind: 'move' as const, index: i };
  }
  return null;
}
/** The cylinder in the box under a canvas point, if any. */
function cylinderAt(px: number, py: number) {
  const { world } = metrics();
  const p = world(px, py);
  const circle = (pile ?? scene)?.circles.find(c => Math.hypot(c.center[0] - p.x, c.center[1] - p.z) <= c.radius);
  return circle?.id ?? null;
}

function placed(px: number, py: number) {
  const { world, scale } = metrics();
  const p = world(px, py);
  if (input('snap').checked && scene) {
    const snapped = snapToCircle(p.x, p.z, scene.target.center, snapRadius(), 12 / scale);
    if (snapped) return snapped;
  }
  return p;
}

function setupCanvas() {
  const canvas = $<HTMLCanvasElement>('view');
  const local = (event: MouseEvent) => { const r = canvas.getBoundingClientRect(); return [event.clientX - r.left, event.clientY - r.top]; };
  canvas.addEventListener('contextmenu', event => {
    event.preventDefault();
    if (mode !== 'draw') return;
    const [px, py] = local(event);
    const found = hit(px, py);
    if (found?.kind === 'move') { checkpoint(); points.splice(found.index, 1); selected = Math.min(selected, points.length - 1); changed(); }
  });
  canvas.addEventListener('dblclick', event => {
    if (mode !== 'draw') return;
    const [px, py] = local(event);
    const found = hit(px, py);
    if (found) { checkpoint(); points[found.index].angle = null; changed(); }
  });
  canvas.addEventListener('pointerdown', event => {
    if (event.button === 2) return;
    const [px, py] = local(event);
    canvas.setPointerCapture(event.pointerId);
    if (targetPickMode && !event.shiftKey && event.button === 0) {
      const id = cylinderAt(px, py);
      if (id) {
        hoverId = null;
        setTargetPickMode(false);
        if (id !== scene?.target_id) void loadScene(id, true).catch(error => status(error.message));
        else status('이미 선택된 원통입니다.');
      } else status('원통 위를 클릭하세요.');
      return;
    }
    if (pickMode && !event.shiftKey && event.button === 0) {
      const id = cylinderAt(px, py);
      if (id) pickCylinder(id);
      return;
    }
    if (event.shiftKey || event.button === 1 || mode !== 'draw') {
      drag = { kind: 'pan', index: -1, startX: px, startY: py, cx: view.cx, cz: view.cz };
      return;
    }
    const found = hit(px, py);
    if (found) { checkpoint(); selected = found.index; drag = { ...found, startX: px, startY: py, cx: 0, cz: 0 }; renderPoints(); draw(); return; }
    if (event.detail > 1) return; // second click of a double-click
    const p = placed(px, py);
    checkpoint();
    points.push({ x: p.x, z: p.z, angle: null, steps: null });
    selected = points.length - 1;
    drag = { kind: 'move', index: selected, startX: px, startY: py, cx: 0, cz: 0 };
    changed();
  });
  canvas.addEventListener('pointermove', event => {
    const [px, py] = local(event);
    if (!drag && (pickMode || targetPickMode)) {
      const id = cylinderAt(px, py);
      if (id !== hoverId) { hoverId = id; draw(); }
      canvas.style.cursor = id && !simulating ? 'pointer' : targetPickMode ? 'crosshair' : 'default';
      return;
    }
    if (!drag) { canvas.style.cursor = mode === 'draw' ? (hit(px, py) ? 'grab' : 'crosshair') : 'default'; return; }
    const { scale, X, Z } = metrics();
    if (drag.kind === 'pan') {
      view.cx = drag.cx - (px - drag.startX) / scale; view.cz = drag.cz + (py - drag.startY) / scale; draw(); return;
    }
    const point = points[drag.index];
    if (!point) return;
    if (drag.kind === 'move') { const p = placed(px, py); point.x = p.x; point.z = p.z; }
    else point.angle = Math.atan2(-(py - Z(point.z)), px - X(point.x));
    clearResult(); renderPoints(); draw();
  });
  const end = () => {
    if (drag && drag.kind !== 'pan') {
      // A click that only selected a point leaves nothing to undo.
      const last = undoStack[undoStack.length - 1];
      if (last && JSON.stringify(last.points) === JSON.stringify(points)) undoStack.pop();
      saveDrawing();
    }
    drag = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const [px, py] = local(event);
    const before = metrics().world(px, py);
    view.size = Math.min(3, Math.max(.08, view.size * Math.pow(1.0015, event.deltaY)));
    const after = metrics().world(px, py);
    view.cx += before.x - after.x; view.cz += before.z - after.z;
    draw();
  }, { passive: false });
}

// ---------------------------------------------------------------- misc actions

/** Project the selected tool's grasp point onto this section. */
function currentGrasp(of: LabScene | null = scene): SectionGrasp | null {
  if (!of) return null;
  const [cx, cz] = of.target.center, toward = of.toward_robot ?? [-1, 0];
  const planeY = of.plane_y ?? 0;
  const robotPosition = of.robot_position ?? [cx + toward[0], planeY, cz + toward[1]];
  const pair = mountedTools ?? [activeToolId, activeToolId];
  const coefficients: RamenGraspCoefficients = { left: toolCoefficients(pair[0]), right: toolCoefficients(pair[1]) };
  return sectionGrasps(of.target, planeY, robotPosition, of.robot_quaternion_xyzw ?? [0, 0, 0, 1], coefficients,
    of.box ? of.frame.up : null)[activeSide()];
}
const handIndex = (side: 'left' | 'right') => side === 'left' ? 0 : 1;
const preferredSide = (toolId: string): 'left' | 'right' => toolId === 'end_support' || toolId === 'nse_spatula' ? 'right' : 'left';
const activeSide = (): 'left' | 'right' => {
  const preferred = preferredSide(activeToolId);
  if (mountedTools?.[handIndex(preferred)] === activeToolId) return preferred;
  return mountedTools?.[handIndex(preferred === 'left' ? 'right' : 'left')] === activeToolId
    ? (preferred === 'left' ? 'right' : 'left') : preferred;
};
const toolLabel = (id: string) => toolOptions.find(option => option.id === id)?.label ?? id;
function migrateLegacyGrasps() {
  try {
    if (localStorage.getItem(TOOL_GRASP_KEY) !== null) return;
    const pair = stored().tools ?? mountedTools ?? ['scoop', 'end_support'];
    const editor = editorGraspCoefficients();
    const migrated: Record<string, GraspPointCoefficients> = {};
    for (const side of ['left', 'right'] as const) migrated[pair[handIndex(side)]] ??= { ...editor[side] };
    localStorage.setItem(TOOL_GRASP_KEY, JSON.stringify(migrated));
  } catch { /* Browser storage is optional. */ }
}
function toolCoefficients(toolId: string): GraspPointCoefficients {
  const saved = storedMap<GraspPointCoefficients>(TOOL_GRASP_KEY)[toolId];
  if (saved && ['lateral', 'clearance', 'height', 'roll', 'pitch', 'yaw'].every(key => Number.isFinite(saved[key as keyof GraspPointCoefficients]))) return saved;
  return DEFAULT_RAMEN_GRASP_COEFFICIENTS[preferredSide(toolId)];
}
const activeToolProfile = (): BladeProfile => toolProfiles[activeSide()] ?? NSE_SCOOP_PROFILE;
// Clamp planning emits robot-side then far-side paths; those use the left and right mounted tools.
const profilesForBlades = (blades: BladePath[]): BladeProfile[] => blades.length > 1
  ? [toolProfiles.left ?? NSE_SCOOP_PROFILE, toolProfiles.right ?? NSE_SCOOP_PROFILE]
  : [activeToolProfile()];
const profilesReadyForBlades = (blades: BladePath[]) => toolModelId === 'g1-tools' && (blades.length > 1
  ? Boolean(toolProfiles.left && toolProfiles.right) : Boolean(toolProfiles[activeSide()]));

function renderMountedTool() {
  const side = activeSide(), select = $<HTMLSelectElement>('mounted-tool');
  select.innerHTML = `<option value="">툴 선택…</option>`
    + toolOptions.map(option => `<option value="${option.id}">${option.label}</option>`).join('');
  if (toolOptions.some(option => option.id === activeToolId)) select.value = activeToolId;
  const pair = mountedTools?.map(toolLabel);
  const sectionStatus = toolProfiles[side] ? '단면 준비됨' : toolModelId === 'g1-tools' ? '단면을 가져오지 못함' : '툴 모델 선택 후 단면 사용';
  $('mounted-tool-info').textContent = toolModelId === 'g1-tools'
    ? `${toolLabel(activeToolId)} 설정 · ${sectionStatus} · 장착: 왼쪽 ${pair?.[0] ?? '—'} / 오른쪽 ${pair?.[1] ?? '—'}`
    : '툴을 고르면 G1 그리퍼 모델로 전환하고 해당 툴의 설정과 실제 충돌 단면을 불러옵니다.';
}

async function refreshToolProfiles() {
  await Promise.all((['left', 'right'] as const).map(async side => {
    delete toolProfiles[side];
    try {
      const profile = await api<ToolSection>(`cylinder-lab/tool-profile?side=${side}`);
      if (mountedTools?.[handIndex(side)] === profile.tool_id) {
        toolProfiles[side] = { outlines: profile.outlines, fixtures: profile.fixtures, area_m2: profile.area_m2, friction: .5 };
      }
    } catch {
      delete toolProfiles[side];
    }
  }));
}

async function changeMountedTool(toolId: string) {
  const select = $<HTMLSelectElement>('mounted-tool');
  select.disabled = true;
  try {
    if (toolModelId !== 'g1-tools') {
      await api('model', { model_id: 'g1-tools' });
      const current = await api<ToolState>('tools');
      toolModelId = current.model_id;
      mountedTools = current.tools ?? ['scoop', 'end_support'];
    }
    if (!mountedTools?.includes(toolId)) {
      const next: [string, string] = [...(mountedTools ?? ['scoop', 'end_support'])];
      next[handIndex(preferredSide(toolId))] = toolId;
      const updated = await api<{ model_id: string; tools: [string, string] }>('tools', { left: next[0], right: next[1] });
      toolModelId = updated.model_id;
      mountedTools = updated.tools;
      await refreshToolProfiles();
      resetPicks();
    }
    activeToolId = toolId;
    remember({ tools: mountedTools ?? undefined, selectedTool: toolId });
    restoreSections(); loadDrawing(); resetHistory(); clearResult(); renderGraspInfo(); renderPoints(); renderMountedTool(); draw();
    status(`${toolLabel(toolId)}의 파지점·경로·설정을 불러왔습니다.`);
  } catch (error) {
    // A failed change can still have switched the robot model; read the server's actual state back.
    try {
      const current = await api<ToolState>('tools');
      toolModelId = current.model_id; mountedTools = current.tools;
    } catch { /* keep the last known state */ }
    renderMountedTool();
    status(error instanceof Error ? error.message : String(error));
  } finally {
    select.disabled = false;
  }
}

const COEFFICIENT_FIELDS: [keyof RamenGraspCoefficients['left'], string, number, number | null][] = [
  ['lateral', 'lateral (m)', .005, 0], ['clearance', 'clearance (m)', .005, 0], ['height', 'height (m)', .005, null],
  ['roll', 'roll (°)', 1, null], ['pitch', 'pitch (°)', 1, null], ['yaw', 'yaw (°)', 1, null],
];

/** The active tool's grasp coefficients; the editor still receives the mounted side's values. */
function renderGraspInfo() {
  const c = toolCoefficients(activeToolId);
  const focused = (document.activeElement as HTMLElement | null)?.dataset?.coefficient;
  $('grasp-coefficients').innerHTML = COEFFICIENT_FIELDS.map(([key, label, step, min]) =>
    `<label class="row">${label}<input data-coefficient="${key}" type="number" step="${step}" ${min === null ? '' : `min="${min}"`} value="${c[key]}"/></label>`).join('');
  decorateTerms($('grasp-coefficients'));
  if (focused) (document.querySelector(`[data-coefficient="${focused}"]`) as HTMLInputElement | null)?.focus();
  // The planner's grasp fields mirror these coefficients.
  const mirror: Record<string, number> = { clearance_m: c.clearance, height_m: c.height, pitch_deg: c.pitch };
  document.querySelectorAll<HTMLInputElement>('#params input').forEach(field => {
    if (field.dataset.key! in mirror) field.value = String(mirror[field.dataset.key!]);
  });
  const grasp = currentGrasp();
  const boxPitch = scene ? boxPitchDeg(scene) : null;
  const entry = boxPitch === null ? '상자가 없어 수평 진입(아래로 0°)'
    : `상자 입구에 수직 진입: 아래로 90 + (${boxPitch.toFixed(0)}°) = ${(90 + boxPitch).toFixed(0)}°`;
  const final = grasp ? ` · pitch 계수 ${c.pitch}° 적용 후 아래로 ${(-grasp.angle * 180 / Math.PI).toFixed(0)}°` : '';
  $('grasp-info').textContent = `${entry}${final}. 계수는 이 툴에 바로 저장됩니다(pitch는 +가 위쪽).`
    + (grasp && grasp.inPlane < .5 ? ' yaw/roll 때문에 날이 단면 밖을 크게 향합니다(단면에서는 짧게 보임).' : '');
}

function writeCoefficients(next: GraspPointCoefficients) {
  try {
    const map = storedMap<GraspPointCoefficients>(TOOL_GRASP_KEY);
    map[activeToolId] = next;
    localStorage.setItem(TOOL_GRASP_KEY, JSON.stringify(map));
    // editorGraspCoefficients may return the shared default object; never mutate it.
    const editor = structuredClone(editorGraspCoefficients());
    for (const side of ['left', 'right'] as const) {
      if (mountedTools?.[handIndex(side)] === activeToolId) editor[side] = next;
    }
    localStorage.setItem(EDITOR_GRASP_KEY, JSON.stringify(editor));
  } catch { status('브라우저 저장소를 쓸 수 없어 계수를 저장하지 못했습니다.'); }
  renderGraspInfo(); draw();
}
function editCoefficient(key: keyof RamenGraspCoefficients['left'], raw: string) {
  const value = Number(raw);
  const field = COEFFICIENT_FIELDS.find(([k]) => k === key)!;
  if (raw.trim() === '' || !Number.isFinite(value) || (field[3] !== null && value < field[3])) { renderGraspInfo(); return; }
  writeCoefficients({ ...toolCoefficients(activeToolId), [key]: value });
}

function graspPoint(): DrawnPoint | null {
  const grasp = currentGrasp();
  if (!grasp) return null;
  input('flip').value = grasp.flip ? '1' : '0';
  return { x: grasp.x, z: grasp.z, angle: grasp.angle, steps: null };
}

function exportDrawing() {
  if (!scene) return;
  const data = { format: 'motioncreator.cylinder-drawn-path.v1', project: projectName, target_id: scene.target_id, frame: scene.frame,
    flip: input('flip').value === '1', stepping: stepping(), points, waypoints: drawnWaypoints(points, stepping()) };
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  link.download = `drawn-path-${scene.target_id.slice(-8)}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}
function applyStepping(value: DrawnStepping | undefined) {
  if (!value) return;
  input('step-m').value = String(value.stepM); input('step-deg').value = String(value.stepDeg); input('speed').value = String(value.speedMS);
  input('turn').value = String(value.turnDegS); input('min-step').value = String(value.minStepS);
}
async function importDrawing(file: File) {
  const data = JSON.parse(await file.text());
  if (data?.format !== 'motioncreator.cylinder-drawn-path.v1' || !Array.isArray(data.points)) throw new Error('그린 경로 JSON이 아닙니다.');
  checkpoint();
  points = data.points.map((p: DrawnPoint) => ({ x: Number(p.x), z: Number(p.z), angle: p.angle === null ? null : Number(p.angle),
    steps: p.steps === null ? null : Number(p.steps) }));
  input('flip').value = data.flip ? '1' : '0';
  applyStepping(data.stepping);
  selected = points.length - 1;
  changed();
  if (data.target_id !== scene?.target_id) status('다른 대상 원통에서 그린 경로입니다. 좌표는 그대로 불러왔습니다.');
}

function setMode(next: 'draw' | 'auto') {
  mode = next;
  $('tab-draw').setAttribute('aria-pressed', String(mode === 'draw'));
  $('tab-auto').setAttribute('aria-pressed', String(mode === 'auto'));
  $('draw-panel').hidden = mode !== 'draw'; $('auto-panel').hidden = mode !== 'auto'; $('points-panel').hidden = mode !== 'draw';
  remember({ mode }); clearResult(); renderPoints(); draw();
}

// ---------------------------------------------------------------- box tilt

/** Show the loaded box pitch and remember it for "원래 기울기로". */
function syncPitch(loadedPitch: number | null = baseScene ? boxPitchDeg(baseScene) : null) {
  originalPitch = loadedPitch;
  const current = baseScene ? boxPitchDeg(baseScene) : null;
  input('box-pitch').value = current === null ? '' : current.toFixed(1);
  input('box-pitch').disabled = current === null;
  $<HTMLButtonElement>('box-pitch-reset').disabled = loadedPitch === null;
}

/** Turn the box and its contents to `pitch`, settle again, and carry the drawn points along. */
async function applyPitch(pitch: number) {
  if (!baseScene?.box || !Number.isFinite(pitch)) return;
  const clamped = Math.max(-90, Math.min(90, pitch));
  const { scene: tilted, delta, pivot } = tiltScene(baseScene, clamped);
  if (Math.abs(delta) < 1e-9) return;
  if (pickMode) setPickMode(false);
  pile = null; attempts = []; renderAttempts(); autoPath = null;
  resetHistory(); // earlier drawings belong to the old tilt
  const c = Math.cos(delta), s = Math.sin(delta);
  points = points.map(p => ({ ...p, x: pivot[0] + (p.x - pivot[0]) * c - (p.z - pivot[1]) * s,
    z: pivot[1] + (p.x - pivot[0]) * s + (p.z - pivot[1]) * c, angle: p.angle === null ? null : p.angle + delta }));
  baseScene = tilted;
  input('box-pitch').value = clamped.toFixed(1);
  status('기울인 상자를 정착시키는 중…');
  await new Promise(resolve => setTimeout(resolve, 10));
  scene = settled(tilted);
  view.cx = scene.frame.origin[0]; view.cz = scene.frame.origin[1] + .05;
  clearResult();
  if (usingSharedPath) loadDrawing(); else saveDrawing();
  renderPoints(); draw(); renderGraspInfo();
  status(`상자 기울기 ${clamped.toFixed(1)}° · 상자와 용기를 함께 돌리고 다시 정착시켰습니다.`);
}

// ---------------------------------------------------------------- glossary (i icons)

const TERM_TARGETS = 'label.row, label.check, h2, .points-head span, .metric span, .legend span';

/** Put an (i) button after every label that has a glossary entry, inside `root`. */
function decorateTerms(root: unknown = document) {
  (root as Document).querySelectorAll<HTMLElement>(TERM_TARGETS).forEach(element => {
    if (element.querySelector(':scope > .term-icon, :scope > .term-label')) return;
    const node = [...element.childNodes].find(child => child.nodeType === Node.TEXT_NODE && child.textContent!.trim());
    const label = node?.textContent?.trim();
    const term = label ? termFor(label) : undefined;
    if (!node || !label || !term) return;
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'term-icon'; button.textContent = 'i';
    button.dataset.term = label; button.setAttribute('aria-label', `용어 설명: ${label}`); button.setAttribute('aria-expanded', 'false');
    if (element.matches('label.row')) {
      // Keep the label's grid to two cells: [text + icon + note] [control].
      const wrapper = document.createElement('span');
      wrapper.className = 'term-label';
      element.insertBefore(wrapper, node);
      wrapper.appendChild(node); wrapper.appendChild(button);
      const note = element.querySelector(':scope > small');
      if (note) wrapper.appendChild(note);
    } else node.after(button);
  });
}

let openTerm: HTMLElement | null = null;
function closeTerm() {
  $('term-pop').hidden = true;
  openTerm?.setAttribute('aria-expanded', 'false');
  openTerm = null;
}
function showTerm(button: HTMLElement) {
  if (openTerm === button) { closeTerm(); return; }
  closeTerm();
  const label = button.dataset.term!, term = termFor(label)!;
  const pop = $('term-pop');
  const escape = (value: string) => value.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
  pop.innerHTML = `<b>${escape(term.title ?? label)}</b><div>${escape(term.text)}</div>`;
  pop.hidden = false;
  const icon = button.getBoundingClientRect(), box = pop.getBoundingClientRect();
  const left = Math.max(8, Math.min(window.innerWidth - box.width - 8, icon.left - 12));
  const below = icon.bottom + 6 + box.height < window.innerHeight;
  pop.style.left = `${left}px`;
  pop.style.top = `${below ? icon.bottom + 6 : Math.max(8, icon.top - box.height - 6)}px`;
  button.setAttribute('aria-expanded', 'true');
  openTerm = button;
}
function setupTerms() {
  document.addEventListener('click', event => {
    const button = (event.target as HTMLElement).closest('.term-icon') as HTMLElement | null;
    if (button) { event.preventDefault(); event.stopPropagation(); showTerm(button); return; }
    if (!(event.target as HTMLElement).closest('#term-pop')) closeTerm();
  }, true);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeTerm(); });
  document.querySelectorAll('aside').forEach(panel => panel.addEventListener('scroll', closeTerm));
  decorateTerms();
}

// ---------------------------------------------------------------- demos

function renderDemos(active?: string) {
  const select = $<HTMLSelectElement>('demo-select');
  select.innerHTML = `<option value="">데모 선택…</option>` + (DEMOS as unknown as Demo[])
    .map(demo => `<option value="${demo.id}" title="${demo.description}">${demo.name}</option>`).join('');
  select.value = active ?? '';
}

/** Load a verified case: its scene, planner parameters, physics settings and planned path, then replay it. */
async function loadDemo(id: string) {
  const demo = (DEMOS as unknown as Demo[]).find(item => item.id === id);
  if (!demo) return;
  projectName = `데모:${demo.id}`;
  const projectSelect = $<HTMLSelectElement>('project');
  projectSelect.querySelector('option[data-demo]')?.remove();
  projectSelect.insertAdjacentHTML('afterbegin', `<option data-demo value="">데모 · ${demo.source}</option>`);
  projectSelect.value = '';
  baseScene = demo.base_scene;
  syncCylinderDiameter();
  $<HTMLSelectElement>('target').innerHTML = `<option value="${baseScene.target_id}">${baseScene.target.id === baseScene.target_id ? (baseScene.circles.find(c => c.id === baseScene!.target_id)?.name ?? '대상') : '대상'}</option>`;
  syncPitch();
  status('정착 중…');
  await new Promise(resolve => setTimeout(resolve, 10));
  scene = settled(baseScene);
  view.cx = scene.frame.origin[0]; view.cz = scene.frame.origin[1] + .05; view.size = .9;
  restoreSections(); loadDrawing(); resetHistory(); if (pickMode) setPickMode(false); pile = null; attempts = []; renderAttempts();
  renderParams({ ...defaults, ...demo.params });
  const soft = demo.options.soft, friction = demo.options.friction ?? {};
  $<HTMLSelectElement>('material').value = soft ? 'soft' : 'rigid';
  if (soft) { input('soft-mm').value = String(soft.mmPer10N); $<HTMLSelectElement>('soft-scope').value = soft.scope; $<HTMLSelectElement>('soft-shape').value = soft.shape; }
  input('mu-cylinder').value = String(friction.cylinder ?? .7); input('mu-blade').value = String(friction.blade ?? .5);
  input('mu-wall').value = String(friction.wall ?? .7); input('rolling').value = String(friction.rolling ?? 0);
  input('force').value = '60'; input('tracking').value = '10'; input('mass').value = '';
  $('material').dispatchEvent(new Event('change'));
  setMode('auto');
  autoPath = { waypoints: demo.blades[0].waypoints, stage_steps: demo.stage_steps, blades: demo.blades };
  renderDemos(demo.id); renderGraspInfo(); renderPoints(); draw();
  status(demo.description);
  await simulate(demo.blades);
}

function copyToDrawing() {
  if (!autoPath?.blades.length) { status('먼저 자동 계획을 실행하거나 데모를 불러오세요.'); return; }
  const blade = autoPath.blades[0];
  checkpoint();
  points = waypointsToPoints(blade.waypoints);
  input('flip').value = blade.flip ? '1' : '0';
  selected = points.length - 1;
  setMode('draw');
  changed();
  status(autoPath.blades.length > 1 ? '왼손 날의 경로만 옮겼습니다(그리기는 날 하나를 씁니다).' : '계획된 경로를 그린 점으로 옮겼습니다. 점을 고친 뒤 시뮬레이션하세요.');
}

// ---------------------------------------------------------------- path generation by picking cylinders

/** The path to carry over: the drawn path in the draw tab, the planned one in the auto tab. */
function currentPaths(): BladePath[] | null {
  if (mode === 'draw') return points.length >= 2 ? [{ waypoints: drawnWaypoints(points, stepping()), flip: input('flip').value === '1', side: 'drawn' }] : null;
  return autoPath?.blades ?? null;
}
const pickStatus = (text: string) => { $('pick-status').textContent = text; };

function setPickMode(on: boolean) {
  if (on) {
    if (!scene) return;
    if (mode === 'draw') {
      const paths = currentPaths();
      if (!paths) { pickStatus('먼저 궤적을 그리세요(점 2개 이상).'); return; }
      template = { kind: 'drawn', paths, referenceCenter: [...scene.target.center], referenceRimUp: scene.frame.rim_up_m };
      pickStatus(`그린 궤적(기준 ${scene.circles.find(c => c.id === scene!.target_id)?.name ?? '대상 원통'})을 옮겨 쓰고, 깊은 원통은 인양을 늘립니다. 꺼낼 라면용기를 클릭하세요.`);
    } else {
      template = { kind: 'auto', params: plannerParams() };
      pickStatus('클릭한 라면용기마다 지금 파라미터로 다시 계획합니다. 꺼낼 라면용기를 클릭하세요.');
    }
    if (!pile) { pile = scene; pileNeedsSettle = false; }
  } else {
    hoverId = null;
    pickStatus('꺼져 있습니다. 궤적을 고친 뒤 다시 켜면 남은 상자에서 이어서 합니다.');
  }
  pickMode = on;
  $('pick-mode').setAttribute('aria-pressed', String(on));
  $('pick-mode').textContent = on ? '경로 생성 중 (끄기)' : '경로 생성';
  draw();
}

function resetPicks() {
  pile = pickMode ? scene : null; pileNeedsSettle = false; attempts = []; viewing = null; hoverId = null;
  clearResult(); renderAttempts(); draw();
  pickStatus(pickMode ? '상자를 처음 상태로 되돌렸습니다. 꺼낼 라면용기를 클릭하세요.' : '상자를 처음 상태로 되돌렸습니다.');
}

function pickCylinder(id: string) {
  if (simulating || !template) return;
  pile ??= scene;
  if (!pile) return;
  const name = pile.circles.find(c => c.id === id)?.name ?? id, order = attempts.length + 1;
  simulating = true; hoverId = null;
  pickStatus(`#${order} ${name} · 파지점 계산과 시뮬레이션 중…`);
  worker ??= new Worker(new URL('./cylinder-batch.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (event: MessageEvent) => {
    simulating = false;
    if (event.data.type === 'error') { pickStatus(event.data.message); return; }
    const attempt: Attempt = event.data.attempt;
    attempts.push(attempt);
    if (attempt.result.success) {
      pile = pileAfter(attempt) as LabScene | null; pileNeedsSettle = true;
      if (!pile) { setPickMode(false); pickStatus(`#${order} ${name} 성공 · 모든 라면용기를 꺼냈습니다.`); }
      else pickStatus(`#${order} ${name} 성공 · 상자 밖으로 나왔습니다. 다음 라면용기를 클릭하세요.`);
    } else {
      pickStatus(`#${order} ${name} 실패 · 상자 안에 남았습니다. 다른 용기를 클릭하거나 궤적을 고친 뒤 다시 켜세요.`);
    }
    viewAttempt(attempt);
  };
  const options = physicsOptions();
  if (template.kind === 'drawn') {
    if (!profilesReadyForBlades(template.paths)) {
      simulating = false; pickStatus('경로에 필요한 손의 장착 툴 단면을 불러오지 못했습니다.'); return;
    }
    const profiles = profilesForBlades(template.paths);
    worker.postMessage({ pile, targetId: id, paths: template.paths, referenceCenter: template.referenceCenter, order,
      options, settle: pileNeedsSettle, profile: profiles, referenceRimUp: template.referenceRimUp });
    return;
  }
  // Auto plan: settle the pile around the chosen cylinder here, plan for it on the server, simulate in the worker.
  const aimed = retarget(pile, id);
  const ready = (pileNeedsSettle ? restingScene(aimed) : aimed) as LabScene;
  const send = worker;
  api<{ path: AutoPath }>('cylinder-lab/plan', { scene: ready, params: template.params })
    .then(planned => {
      if (!profilesReadyForBlades(planned.path.blades)) {
        simulating = false; pickStatus('경로에 필요한 손의 장착 툴 단면을 불러오지 못했습니다.'); return;
      }
      send.postMessage({ pile: ready, targetId: id, paths: planned.path.blades, referenceCenter: ready.target.center, order,
        options, settle: false, profile: profilesForBlades(planned.path.blades) });
    })
    .catch(error => { simulating = false; pickStatus(`#${order} ${name} 계획 실패: ${error instanceof Error ? error.message : error}`); });
}

function renderAttempts() {
  $('batch-panel').hidden = !attempts.length;
  $('batch-table').innerHTML = attempts.map((attempt, index) => {
    const r = attempt.result;
    const state = r.success ? '<b class="ok">성공</b>' : '<b class="fail">실패</b>';
    const detail = `들림 ${(r.liftM * 1000).toFixed(0)} mm${r.maxSquashMm > 0 ? ` · 찌그러짐 ${r.maxSquashMm.toFixed(0)} mm` : ''}`;
    const note = r.success ? r.warnings.join(' · ') : r.reason;
    return `<button class="batch-row ${viewing === attempt ? 'viewing' : ''}" data-index="${index}"><b>#${attempt.order}</b><span>${attempt.name}</span>${state}<span>${detail}</span>`
      + (note ? `<small>${note}</small>` : '') + '</button>';
  }).join('');
  const out = attempts.filter(a => a.result.success).length;
  $('batch-summary').textContent = attempts.length
    ? `${attempts.length}번 시도 · ${out}개 꺼냄 · 상자에 ${(pile ?? scene)?.circles.length ?? 0}개 남음` : '';
}

function viewAttempt(attempt: Attempt) {
  clearResult();
  viewing = attempt; simBlades = attempt.paths; simProfiles = attempt.profiles ?? profilesForBlades(attempt.paths); result = attempt.result;
  frame = 0; input('scrub').max = String(result.frames.length - 1);
  showResult(); renderAttempts(); playing = true;
  status(`#${attempt.order} ${attempt.name} 재생`);
}

function tick(time: number) {
  if (playing && result) {
    const period = result.frames.length > 1 ? (result.frames[1].t - result.frames[0].t) * 1000 : 16;
    if (time - lastTick >= period) {
      lastTick = time;
      frame = Math.min(result.frames.length - 1, frame + 1);
      if (frame === result.frames.length - 1) playing = false;
      draw();
    }
  }
  $('play').textContent = playing ? '❚❚' : '▶';
  requestAnimationFrame(tick);
}

async function init() {
  const saved = stored();
  const [names, tools] = await Promise.all([api<string[]>('saved'), api<ToolState>('tools')]);
  toolModelId = tools.model_id;
  mountedTools = tools.tools;
  toolOptions = tools.options;
  migrateLegacyGrasps();
  migrateLegacySections();
  activeToolId = saved.selectedTool && toolOptions.some(option => option.id === saved.selectedTool)
    ? saved.selectedTool : legacyToolId();
  for (const group of Object.keys(SAVE_GROUPS)) DEFAULT_GROUPS[group] = readSection(group);
  if (toolModelId === 'g1-tools') await refreshToolProfiles();
  renderMountedTool();
  const select = $<HTMLSelectElement>('project');
  select.innerHTML = names.map(name => `<option value="${name}">${name.replace(/\/project\.json$/, '')}</option>`).join('');
  select.value = saved.project && names.includes(saved.project) ? saved.project : names[0];
  select.onchange = () => { if (!select.value) return; renderDemos(); select.querySelector('option[data-demo]')?.remove(); void loadProject(select.value).catch(error => status(error.message)); };
  $<HTMLSelectElement>('target').onchange = event => void loadScene((event.target as HTMLSelectElement).value, true).catch(error => status(error.message));
  $('run').onclick = () => void run();
  $('select-sim-target').onclick = () => setTargetPickMode(!targetPickMode);
  $('reset').onclick = () => { renderParams(defaults); persistSection('auto'); renderGraspInfo(); };
  document.querySelectorAll<HTMLElement>('.save-section').forEach(button => button.addEventListener('click', event => {
    event.preventDefault(); event.stopPropagation(); saveSection(button.dataset.save!, button);
  }));
  $('to-draw').onclick = copyToDrawing;
  $('share-path').onclick = sharePath;
  $('detach-path').onclick = detachPath;
  $('restore-individual').onclick = restoreIndividualPath;
  $('reset-to-shared').onclick = resetToSharedPath;
  $('pick-mode').onclick = () => setPickMode(!pickMode);
  input('box-pitch').addEventListener('change', () => void applyPitch(num('box-pitch')));
  $('box-pitch-reset').onclick = () => { if (originalPitch !== null) void applyPitch(originalPitch); };
  input('cylinder-diameter').addEventListener('change', () => {
    const value = input('cylinder-diameter').value.trim();
    if (!value) return;
    applyCylinderDiameter(Number(value));
  });
  $('cylinder-diameter-reset').onclick = () => {
    if (loadedTargetDiameter !== null) applyCylinderDiameter(loadedTargetDiameter);
  };
  $('pick-reset').onclick = resetPicks;
  $('batch-table').addEventListener('click', event => {
    const row = (event.target as HTMLElement).closest('.batch-row') as HTMLElement | null;
    const attempt = row ? attempts[Number(row.dataset.index)] : undefined;
    if (attempt) viewAttempt(attempt);
  });
  renderDemos();
  $<HTMLSelectElement>('demo-select').addEventListener('change', event => {
    const id = (event.target as HTMLSelectElement).value;
    if (id) void loadDemo(id).catch(error => status(error.message));
  });
  $('tab-draw').onclick = () => setMode('draw');
  $('tab-auto').onclick = () => setMode('auto');
  $('from-grasp').onclick = () => { if (!scene) return; checkpoint(); const p = graspPoint(); if (p) { points.unshift(p); selected = 0; changed(); } };
  $('mounted-tool').addEventListener('change', event => {
    const selected = (event.target as HTMLSelectElement).value;
    if (selected) void changeMountedTool(selected);
  });
  $('grasp-coefficients').addEventListener('change', event => {
    const target = event.target as HTMLInputElement;
    if (target.dataset.coefficient) editCoefficient(target.dataset.coefficient as keyof RamenGraspCoefficients['left'], target.value);
  });
  $('grasp-defaults').onclick = () => {
    writeCoefficients({ ...DEFAULT_RAMEN_GRASP_COEFFICIENTS[preferredSide(activeToolId)] });
  };
  input('show-grasp').addEventListener('change', draw);
  window.addEventListener('storage', event => {
    if (event.key === TOOL_GRASP_KEY || event.key === EDITOR_GRASP_KEY) { renderGraspInfo(); draw(); }
    if (event.key === TOOL_SAVED_KEY) restoreSections();
  });
  $('undo-point').onclick = () => { if (!points.length) return; checkpoint(); points.pop(); selected = points.length - 1; changed(); };
  $('clear-points').onclick = () => { if (!points.length) return; checkpoint(); points = []; selected = -1; changed(); };
  let beforeFlip: Snapshot | null = null;
  input('flip').addEventListener('focus', () => { beforeFlip = snapshot(); });
  input('flip').addEventListener('change', () => {
    if (beforeFlip) { undoStack.push(beforeFlip); redoStack.length = 0; }
    beforeFlip = snapshot();
  });
  document.addEventListener('keydown', event => {
    if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
    const target = event.target;
    if (target instanceof Element && target.matches('input:not([type=checkbox]), textarea, select')) return; // text fields keep their own undo
    const key = event.key.toLowerCase();
    if (key === 'z' && !event.shiftKey) { event.preventDefault(); if (mode === 'draw') undo(); }
    else if ((key === 'z' && event.shiftKey) || key === 'y') { event.preventDefault(); if (mode === 'draw') redo(); }
  });
  $('export').onclick = exportDrawing;
  $('import').onclick = () => input('import-file').click();
  input('import-file').onchange = event => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) void importDrawing(file).catch(error => status(error.message));
    (event.target as HTMLInputElement).value = '';
  };
  for (const id of ['step-m', 'step-deg', 'speed', 'turn', 'min-step', 'snap-wedge', 'flip', 'snap']) input(id).addEventListener('change', changed);
  for (const [group, ids] of Object.entries(SAVE_GROUPS)) {
    for (const id of ids) input(id).addEventListener('change', () => persistSection(group));
  }
  $('params').addEventListener('change', event => {
    if ((event.target as HTMLElement).matches('input[data-key]')) persistSection('auto');
  });
  $('points').addEventListener('change', event => {
    const target = event.target as HTMLInputElement, row = target.closest('.point') as HTMLElement | null;
    if (row && target.dataset.field) editPoint(Number(row.dataset.i), target.dataset.field, target.value);
  });
  // Reorder drawn points by dragging a row's handle onto another row (above or below its middle).
  let dragFrom = -1;
  const clearDrop = () => document.querySelectorAll('.point.drop-before, .point.drop-after').forEach(row => row.classList.remove('drop-before', 'drop-after'));
  $('points').addEventListener('dragstart', event => {
    const handle = (event.target as HTMLElement).closest('.drag-handle'), row = handle?.closest('.point') as HTMLElement | null;
    if (!row || !event.dataTransfer) { event.preventDefault(); return; }
    dragFrom = Number(row.dataset.i);
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', String(dragFrom));
    event.dataTransfer.setDragImage(row, 12, 12);
    row.classList.add('dragging');
  });
  $('points').addEventListener('dragover', event => {
    const row = (event.target as HTMLElement).closest('.point') as HTMLElement | null;
    if (dragFrom < 0 || !row) return;
    event.preventDefault();
    const box = row.getBoundingClientRect(), after = event.clientY > box.top + box.height / 2;
    clearDrop(); row.classList.add(after ? 'drop-after' : 'drop-before');
  });
  $('points').addEventListener('drop', event => {
    const row = (event.target as HTMLElement).closest('.point') as HTMLElement | null;
    if (dragFrom < 0 || !row) return;
    event.preventDefault();
    const over = Number(row.dataset.i), box = row.getBoundingClientRect(), after = event.clientY > box.top + box.height / 2;
    const slot = over + (after ? 1 : 0), to = slot > dragFrom ? slot - 1 : slot;
    if (to !== dragFrom) {
      checkpoint();
      const moving = points[dragFrom];
      points = movePoint(points, dragFrom, to);
      selected = points.indexOf(moving);
      changed();
      status(`${dragFrom + 1}번 점을 ${to + 1}번 자리로 옮겼습니다.`);
    }
  });
  $('points').addEventListener('dragend', () => {
    dragFrom = -1; clearDrop();
    document.querySelectorAll('.point.dragging').forEach(row => row.classList.remove('dragging'));
  });
  $('points').addEventListener('click', event => {
    const target = event.target as HTMLElement, row = target.closest('.point') as HTMLElement | null;
    if (!row) return;
    if (target.dataset.field === 'delete') editPoint(Number(row.dataset.i), 'delete', '');
    else if (target.tagName !== 'INPUT') { selected = Number(row.dataset.i); renderPoints(); draw(); }
  });
  $('play').onclick = () => { if (!result) return; if (frame >= result.frames.length - 1) frame = 0; playing = !playing; };
  input('scrub').oninput = event => { frame = Number((event.target as HTMLInputElement).value); playing = false; draw(); };
  const syncMaterial = () => {
    const soft = $<HTMLSelectElement>('material').value === 'soft';
    for (const id of ['soft-shape', 'soft-mm', 'soft-scope']) $<HTMLSelectElement>(id).disabled = !soft;
    input('rolling').disabled = soft && $<HTMLSelectElement>('soft-scope').value === 'all';
  };
  for (const id of ['material', 'soft-scope']) $(id).addEventListener('change', syncMaterial);
  syncMaterial();
  window.addEventListener('resize', draw);
  setupTerms();
  setupCanvas();
  setMode(saved.mode === 'auto' ? 'auto' : 'draw');
  requestAnimationFrame(tick);
  if (toolModelId !== 'g1-tools' || !mountedTools?.includes(activeToolId)) await changeMountedTool(activeToolId);
  if (select.value) await loadProject(select.value);
  else if ((DEMOS as unknown as Demo[]).length) await loadDemo((DEMOS as unknown as Demo[])[0].id);
}
init().catch(error => status(error instanceof Error ? error.message : String(error)));
