// 2-D lab page: draw (or auto-plan) a TCP path in the cylinder cross-section and check it with Box2D.
import { settleScene, simulateScoop, NSE_SCOOP_PROFILE, DEFAULT_SOFTNESS, type BladePath, type Scene, type SimResult, type SoftShape, type Waypoint } from '../lib/cylinder-sim';
import { autoSteps, drawnWaypoints, pointAngles, snapToCircle, type DrawnPoint, type DrawnStepping } from '../lib/drawn-path';

type LabScene = Scene & { toward_robot?: number[] };
type AutoPath = { waypoints: Waypoint[]; stage_steps: Record<string, number>; blades: BladePath[] };
type Stored = { project?: string; mode?: string; params?: Record<string, number>; stepping?: DrawnStepping;
  drawings?: Record<string, { points: DrawnPoint[]; flip: boolean }>;
  physics?: { material?: string; softMm?: number; scope?: string; shape?: string; force?: number; tracking?: number;
    muCylinder?: number; muBlade?: number; muWall?: number; rolling?: number } };
const $ = <T = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
const input = (id: string) => $<HTMLInputElement>(id);
const num = (id: string) => Number(input(id).value);
const STORAGE_KEY = 'motioncreator-cylinder-lab-v2';
const EDITOR_GRASP_KEY = 'motioncreator-editor-ramen-grasp-v1';
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
let scene: LabScene | null = null;
let defaults: Record<string, number> = {};
let limits: Record<string, [number, number]> = {};
let mode: 'draw' | 'auto' = 'draw';
let points: DrawnPoint[] = [];
let selected = -1;
let autoPath: AutoPath | null = null;
let simBlades: BladePath[] = [];
let result: SimResult | null = null;
let frame = 0, playing = false, lastTick = 0;
const view = { cx: 0, cz: 0, size: .9 };

// ---------------------------------------------------------------- storage

function stored(): Stored { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { return {}; } }
function remember(values: Stored) { try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...stored(), ...values })); } catch { /* optional */ } }
const drawKey = () => `${projectName}|${scene?.target_id ?? ''}`;
function saveDrawing() {
  const drawings = stored().drawings ?? {};
  drawings[drawKey()] = { points, flip: input('flip').value === '1' };
  remember({ drawings, stepping: stepping() });
}
function loadDrawing() {
  const saved = (stored().drawings ?? {})[drawKey()];
  points = Array.isArray(saved?.points) ? saved.points : [];
  input('flip').value = saved?.flip ? '1' : '0';
  selected = points.length - 1;
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

/** Resting scene: settle, then move the target frame with the target (like with_positions on the server). */
function settled(source: LabScene): LabScene {
  const rest = settleScene(source) as LabScene;
  const [ox, oz] = source.frame.origin, [nx, nz] = rest.target.center, [ux, uz] = source.frame.up;
  return { ...rest, toward_robot: source.toward_robot,
    frame: { ...source.frame, origin: [nx, nz], rim_up_m: source.frame.rim_up_m - ((nx - ox) * ux + (nz - oz) * uz) } };
}

async function loadProject(name: string) {
  projectName = name;
  project = await api(`project/${name.split('/').map(encodeURIComponent).join('/')}`);
  remember({ project: name });
  await loadScene(undefined);
}
async function loadScene(targetId: string | undefined) {
  const response = await api<{ scene: LabScene; defaults: Record<string, number>; limits: Record<string, [number, number]>; targets: { id: string; name: string }[] }>(
    'cylinder-lab/scene', { project, target_id: targetId ?? null });
  baseScene = response.scene; defaults = response.defaults; limits = response.limits;
  $<HTMLSelectElement>('target').innerHTML = response.targets.map(t => `<option value="${t.id}">${t.name}</option>`).join('');
  $<HTMLSelectElement>('target').value = baseScene.target_id;
  renderParams({ ...defaults, ...Object.fromEntries(Object.entries(stored().params ?? {}).filter(([key]) => key in defaults)) });
  status('정착 중…');
  await new Promise(resolve => setTimeout(resolve, 10));
  scene = settled(baseScene);
  view.cx = scene.frame.origin[0]; view.cz = scene.frame.origin[1] + .05; view.size = .9;
  loadDrawing();
  autoPath = null; clearResult();
  renderPoints(); draw(); status('');
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
    return `<label class="row">${label}${note ? `<small>${note}</small>` : ''}<input data-key="${key}" type="number" step="any" min="${low}" max="${high}" value="${values[key]}"/></label>`;
  }).join('');
}
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
    return `<div class="point ${i === selected ? 'selected' : ''}" data-i="${i}"><b>${i + 1}</b><span class="xy">${(a * 1000).toFixed(0)} / ${(b * 1000).toFixed(0)}</span>`
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
  if (field === 'delete') { points.splice(index, 1); selected = Math.min(selected, points.length - 1); }
  else if (field === 'angle') point.angle = value === '' ? null : wrap(Number(value) * Math.PI / 180 + frameAxes().farAngle);
  else if (field === 'steps') point.steps = value === '' ? null : Math.max(1, Math.round(Number(value)));
  changed();
}
function changed() { clearResult(); saveDrawing(); renderPoints(); draw(); }
function clearResult() {
  result = null; frame = 0; playing = false;
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
      const params = readParams();
      remember({ params });
      status('궤적 계획 중…');
      const planned = await api<{ scene: LabScene; path: AutoPath }>('cylinder-lab/plan', {
        scene: baseScene, params, positions: Object.fromEntries(scene.circles.map(c => [c.id, c.center])) });
      autoPath = planned.path; simBlades = planned.path.blades;
      renderPoints();
    }
    const soft = $<HTMLSelectElement>('material').value === 'soft'
      ? { ...DEFAULT_SOFTNESS, mmPer10N: Math.max(.5, num('soft-mm')), scope: $<HTMLSelectElement>('soft-scope').value as 'neighbors' | 'all',
        shape: $<HTMLSelectElement>('soft-shape').value as SoftShape } : null;
    const friction = { cylinder: Math.max(0, num('mu-cylinder')), blade: Math.max(0, num('mu-blade')), wall: Math.max(0, num('mu-wall')),
      rolling: Math.max(0, num('rolling')) };
    remember({ physics: { material: $<HTMLSelectElement>('material').value, softMm: num('soft-mm'), scope: $<HTMLSelectElement>('soft-scope').value,
      shape: $<HTMLSelectElement>('soft-shape').value, force: num('force'), tracking: num('tracking'),
      muCylinder: friction.cylinder, muBlade: friction.blade, muWall: friction.wall, rolling: friction.rolling } });
    status(soft ? '말랑한 원통으로 물리 계산 중… (몇 초 걸립니다)' : '물리 계산 중…');
    await new Promise(resolve => setTimeout(resolve, 20));
    result = simulateScoop(scene, simBlades, NSE_SCOOP_PROFILE,
      { forceLimitN: num('force'), trackingLimitM: num('tracking') / 1000, frameEvery: 4, soft, friction });
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
  for (const wall of scene.walls) quad(wall.center, wall.half, wall.angle, '#8a6b4a');
  for (const circle of scene.circles) {
    const fill = circle.id === scene.target_id ? '#e8a33d' : '#56708a';
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
  for (const r of scene.rectangles) {
    const pose = current?.bodies[r.id];
    quad(pose ? [pose.x, pose.z] : r.center, r.half, pose ? pose.angle : r.angle, '#56708a');
  }
  const bladeShape = (pose: { x: number; z: number; angle: number }, fill: string | null, stroke: string, dashed: boolean, flipped: boolean) => {
    const p = NSE_SCOOP_PROFILE, flip = flipped ? -1 : 1, back = p.tipAhead - p.length;
    ctx.save(); ctx.translate(X(pose.x), Z(pose.z)); ctx.rotate(-pose.angle); ctx.scale(scale, -scale * flip);
    ctx.beginPath(); ctx.rect(back, -p.thickness, p.length, p.thickness); ctx.rect(back - p.heelLength, -p.thickness, p.heelLength, p.thickness + p.heelHeight);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.lineWidth = 1.5 / scale; ctx.setLineDash(dashed ? [4 / scale, 3 / scale] : []); ctx.strokeStyle = stroke; ctx.stroke(); ctx.restore();
  };
  const flipped = input('flip').value === '1';
  if (mode === 'draw') {
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
      bladeShape(current.commands[index], null, '#d9f', true, bladeFlip);
      bladeShape(pose, index ? '#6ec3ffaa' : '#80efc5aa', index ? '#6ec3ff' : '#80efc5', false, bladeFlip);
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
    if (found?.kind === 'move') { points.splice(found.index, 1); selected = Math.min(selected, points.length - 1); changed(); }
  });
  canvas.addEventListener('dblclick', event => {
    if (mode !== 'draw') return;
    const [px, py] = local(event);
    const found = hit(px, py);
    if (found) { points[found.index].angle = null; changed(); }
  });
  canvas.addEventListener('pointerdown', event => {
    if (event.button === 2) return;
    const [px, py] = local(event);
    canvas.setPointerCapture(event.pointerId);
    if (event.shiftKey || event.button === 1 || mode !== 'draw') {
      drag = { kind: 'pan', index: -1, startX: px, startY: py, cx: view.cx, cz: view.cz };
      return;
    }
    const found = hit(px, py);
    if (found) { selected = found.index; drag = { ...found, startX: px, startY: py, cx: 0, cz: 0 }; renderPoints(); draw(); return; }
    if (event.detail > 1) return; // second click of a double-click
    const p = placed(px, py);
    points.push({ x: p.x, z: p.z, angle: null, steps: null });
    selected = points.length - 1;
    drag = { kind: 'move', index: selected, startX: px, startY: py, cx: 0, cz: 0 };
    changed();
  });
  canvas.addEventListener('pointermove', event => {
    const [px, py] = local(event);
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
  const end = () => { if (drag && drag.kind !== 'pan') saveDrawing(); drag = null; };
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

function graspPoint(): DrawnPoint | null {
  if (!scene) return null;
  let coefficients = { clearance: .1, height: 0, pitch: 0 };
  try {
    const saved = JSON.parse(localStorage.getItem(EDITOR_GRASP_KEY) ?? 'null');
    if (saved?.left) coefficients = { ...coefficients, ...saved.left };
  } catch { /* use defaults */ }
  const toward = scene.toward_robot ?? [-1, 0];
  const [cx, cz] = scene.target.center, r = scene.target.radius;
  return { x: cx + toward[0] * (r + coefficients.clearance), z: cz + toward[1] * (r + coefficients.clearance) + coefficients.height,
    angle: Math.atan2(-toward[1], -toward[0]) + coefficients.pitch * Math.PI / 180, steps: null };
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
  applyStepping(saved.stepping);
  if (saved.physics) {
    $<HTMLSelectElement>('material').value = saved.physics.material ?? 'rigid'; input('soft-mm').value = String(saved.physics.softMm ?? 5);
    $<HTMLSelectElement>('soft-scope').value = saved.physics.scope ?? 'neighbors'; input('force').value = String(saved.physics.force ?? 60);
    input('tracking').value = String(saved.physics.tracking ?? 10);
    $<HTMLSelectElement>('soft-shape').value = saved.physics.shape ?? 'smooth';
    input('mu-cylinder').value = String(saved.physics.muCylinder ?? .7); input('mu-blade').value = String(saved.physics.muBlade ?? .5);
    input('mu-wall').value = String(saved.physics.muWall ?? .7); input('rolling').value = String(saved.physics.rolling ?? 0);
  }
  const names = await api<string[]>('saved');
  const select = $<HTMLSelectElement>('project');
  select.innerHTML = names.map(name => `<option value="${name}">${name.replace(/\/project\.json$/, '')}</option>`).join('');
  select.value = saved.project && names.includes(saved.project) ? saved.project : names[0];
  select.onchange = () => void loadProject(select.value).catch(error => status(error.message));
  $<HTMLSelectElement>('target').onchange = event => void loadScene((event.target as HTMLSelectElement).value).catch(error => status(error.message));
  $('run').onclick = () => void run();
  $('reset').onclick = () => { remember({ params: {} }); renderParams(defaults); };
  $('tab-draw').onclick = () => setMode('draw');
  $('tab-auto').onclick = () => setMode('auto');
  $('from-grasp').onclick = () => { const p = graspPoint(); if (p) { points.unshift(p); selected = 0; changed(); } };
  $('undo-point').onclick = () => { points.pop(); selected = points.length - 1; changed(); };
  $('clear-points').onclick = () => { points = []; selected = -1; changed(); };
  $('export').onclick = exportDrawing;
  $('import').onclick = () => input('import-file').click();
  input('import-file').onchange = event => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) void importDrawing(file).catch(error => status(error.message));
    (event.target as HTMLInputElement).value = '';
  };
  for (const id of ['step-m', 'step-deg', 'speed', 'turn', 'min-step', 'snap-wedge', 'flip', 'snap']) input(id).addEventListener('change', changed);
  $('points').addEventListener('change', event => {
    const target = event.target as HTMLInputElement, row = target.closest('.point') as HTMLElement | null;
    if (row && target.dataset.field) editPoint(Number(row.dataset.i), target.dataset.field, target.value);
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
  setupCanvas();
  setMode(saved.mode === 'auto' ? 'auto' : 'draw');
  requestAnimationFrame(tick);
  if (select.value) await loadProject(select.value);
}
init().catch(error => status(error instanceof Error ? error.message : String(error)));
