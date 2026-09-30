// 2-D lab page: plan a scoop path on the server (motioncreator.cylinder_extraction) and check it with Box2D.
import { settleScene, simulateScoop, NSE_SCOOP_PROFILE, type Scene, type SimResult, type Waypoint } from '../lib/cylinder-sim';

type Path = { waypoints: Waypoint[]; stage_steps: Record<string, number>; blade_flip: boolean; blades: { side: string; waypoints: Waypoint[]; flip: boolean }[] };
const $ = <T = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
const PARAM_LABELS: Record<string, [string, string?]> = {
  mode: ['방식', '0 = 둘레 감기, 1 = 지렛대, 2 = 양쪽 집기'], side: ['진입 쪽', '-1 = 로봇 쪽, +1 = 먼 쪽'],
  clearance_m: ['파지점 거리 (m)'], height_m: ['파지점 높이 (m)'], pitch_deg: ['파지점 피치 (°)'],
  approach_height_m: ['틈 위 대기 높이 (m)'], wedge_m: ['쐐기 깊이 (m)', '날 윗면이 원통 표면 안쪽으로'],
  insert_depth: ['삽입 깊이 (×r)', '원통 중심 아래로'], pivot_up: ['지렛대 받침점 (×r)', '원통 중심 위로'], pry_deg: ['젖히는 각도 (°)'],
  wrap_start_deg: ['감기 시작 (°)'], wrap_end_deg: ['감기 끝 (°)'], slide_extent: ['밀어 넣기 (×r)'],
  tilt_deg: ['날 끝 들기 (°)'], tilt_rise_m: ['들기 상승 (m)'], lift_margin_m: ['인양 여유 (m)'], carry_m: ['몸쪽 이동 (m)'],
  step_m: ['한 걸음 거리 (m)', '작을수록 단계가 많아짐'], step_deg: ['한 걸음 각도 (°)'],
  speed_m_s: ['속도 (m/s)'], turn_deg_s: ['회전 속도 (°/s)'], min_step_s: ['단계 최소 시간 (s)'],
};
const STORAGE_KEY = 'motioncreator-cylinder-lab-v1';

let project: Record<string, unknown> | null = null;
let baseScene: Scene | null = null;
let defaults: Record<string, number> = {};
let limits: Record<string, [number, number]> = {};
let scene: Scene | null = null;
let path: Path | null = null;
let result: SimResult | null = null;
let frame = 0, playing = false, lastTick = 0;

async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${url}`, body === undefined ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({})) as { detail?: unknown };
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `요청 실패 (${response.status})`);
  return data as T;
}
const status = (text: string) => { $('status').textContent = text; };
function saved(): Record<string, unknown> {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { return {}; }
}
function remember(values: Record<string, unknown>) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...saved(), ...values })); } catch { /* optional */ }
}

function renderParams(values: Record<string, number>) {
  $('params').innerHTML = Object.keys(defaults).map(key => {
    const [label, note] = PARAM_LABELS[key] ?? [key];
    const [low, high] = limits[key] ?? [-1e9, 1e9];
    return `<label class="row">${label}${note ? `<small>${note}</small>` : ''}<input data-key="${key}" type="number" step="any" min="${low}" max="${high}" value="${values[key]}"/></label>`;
  }).join('');
}
function readParams() {
  return Object.fromEntries([...document.querySelectorAll<HTMLInputElement>('#params input')].map(input => [input.dataset.key!, Number(input.value)]));
}

async function loadProject(name: string) {
  project = await api(`project/${name.split('/').map(encodeURIComponent).join('/')}`);
  remember({ project: name });
  await loadScene(undefined);
}
async function loadScene(targetId: string | undefined) {
  const response = await api<{ scene: Scene; defaults: Record<string, number>; limits: Record<string, [number, number]>; targets: { id: string; name: string }[] }>(
    'cylinder-lab/scene', { project, target_id: targetId ?? null });
  baseScene = response.scene; defaults = response.defaults; limits = response.limits;
  $<HTMLSelectElement>('target').innerHTML = response.targets.map(t => `<option value="${t.id}">${t.name}</option>`).join('');
  $<HTMLSelectElement>('target').value = baseScene.target_id;
  const stored = saved().params as Record<string, number> | undefined;
  renderParams({ ...defaults, ...Object.fromEntries(Object.entries(stored ?? {}).filter(([key]) => key in defaults)) });
  scene = baseScene; path = null; result = null; frame = 0;
  draw(); status('장면을 불러왔습니다. 계획 + 시뮬레이션을 누르세요.');
}

async function run() {
  if (!baseScene) return;
  $<HTMLButtonElement>('run').disabled = true;
  try {
    status('정착 중…');
    const rest = settleScene(baseScene);
    const params = readParams();
    remember({ params });
    status('궤적 계획 중…');
    const planned = await api<{ scene: Scene; path: Path }>('cylinder-lab/plan', {
      scene: baseScene, params, positions: Object.fromEntries(rest.circles.map(c => [c.id, c.center])) });
    scene = planned.scene; path = planned.path;
    status('물리 계산 중…');
    await new Promise(resolve => setTimeout(resolve, 20));
    result = simulateScoop(scene, path.blades, NSE_SCOOP_PROFILE, {
      forceLimitN: Number($<HTMLInputElement>('force').value), trackingLimitM: Number($<HTMLInputElement>('tracking').value) / 1000, frameEvery: 4,
    });
    frame = 0; $<HTMLInputElement>('scrub').max = String(result.frames.length - 1);
    showResult(); status(''); playing = true;
  } catch (error) {
    status(error instanceof Error ? error.message : String(error));
  } finally { $<HTMLButtonElement>('run').disabled = false; }
}

function showResult() {
  if (!result || !path) return;
  const box = $('result');
  box.className = `result ${result.success ? 'ok' : 'fail'}`;
  const row = (label: string, value: string) => `<div class="metric"><span>${label}</span><b>${value}</b></div>`;
  box.innerHTML = `<b>${result.success ? '성공' : '실패'}</b><p class="hint">${result.reason}</p>`
    + row('단계 수', `${result.steps}`) + row('시간', `${result.seconds.toFixed(1)} s`)
    + row('들린 높이', `${(result.liftM * 1000).toFixed(0)} / ${(result.requiredLiftM * 1000).toFixed(0)} mm`)
    + row('최대 힘', `${result.peakForceN.toFixed(0)} N`) + row('이동 중 최대 지연', `${(result.trackingMaxM * 1000).toFixed(0)} mm / ${result.trackingMaxDeg.toFixed(0)}°`)
    + row('접촉 단계에서 덜 간 거리', `${(result.stalledM * 1000).toFixed(0)} mm`) + row('끝 자세 오차', `${(result.finalErrorM * 1000).toFixed(0)} mm`)
    + row('이웃 최대 이동', `${(result.neighborMaxM * 1000).toFixed(0)} mm`)
    + `<p class="hint">${Object.entries(path.stage_steps).map(([k, v]) => `${k} ${v}`).join(' · ')}</p>`;
  $('steps').innerHTML = path.waypoints.map((w, i) => `<div data-i="${i}"><span>${i}. ${w.label}</span><span>${w.duration.toFixed(2)}s</span></div>`).join('');
}

function draw() {
  const canvas = $<HTMLCanvasElement>('view');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0e141b'; ctx.fillRect(0, 0, rect.width, rect.height);
  const shown = scene ?? baseScene;
  if (!shown) return;
  const [cx, cz] = shown.frame.origin;
  const view = 0.9; // metres across
  const scale = Math.min(rect.width, rect.height) / view;
  const X = (x: number) => rect.width / 2 + (x - cx) * scale;
  const Z = (z: number) => rect.height * .55 - (z - cz) * scale;
  const current = result?.frames[frame];
  ctx.strokeStyle = '#1f2a35'; ctx.lineWidth = 1;
  for (let g = -1; g <= 1.001; g += .1) {
    ctx.beginPath(); ctx.moveTo(X(cx + g), 0); ctx.lineTo(X(cx + g), rect.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, Z(cz + g)); ctx.lineTo(rect.width, Z(cz + g)); ctx.stroke();
  }
  const quad = (center: number[], half: number[], angle: number, fill: string) => {
    ctx.save(); ctx.translate(X(center[0]), Z(center[1])); ctx.rotate(-angle);
    ctx.fillStyle = fill; ctx.fillRect(-half[0] * scale, -half[1] * scale, half[0] * 2 * scale, half[1] * 2 * scale); ctx.restore();
  };
  for (const wall of shown.walls) quad(wall.center, wall.half, wall.angle, '#8a6b4a');
  for (const circle of shown.circles) {
    const pose = current?.bodies[circle.id];
    const [x, z, a] = pose ? [pose.x, pose.z, pose.angle] : [circle.center[0], circle.center[1], 0];
    ctx.beginPath(); ctx.arc(X(x), Z(z), circle.radius * scale, 0, Math.PI * 2);
    ctx.fillStyle = circle.id === shown.target_id ? '#e8a33d' : '#56708a'; ctx.fill();
    ctx.strokeStyle = '#0e141b'; ctx.beginPath(); ctx.moveTo(X(x), Z(z)); ctx.lineTo(X(x + Math.cos(a) * circle.radius), Z(z + Math.sin(a) * circle.radius)); ctx.stroke();
  }
  for (const r of shown.rectangles) {
    const pose = current?.bodies[r.id];
    quad(pose ? [pose.x, pose.z] : r.center, r.half, pose ? pose.angle : r.angle, '#56708a');
  }
  if (path) {
    for (const plan of path.blades) {
      ctx.fillStyle = '#d9f'; ctx.strokeStyle = '#d9f5'; ctx.beginPath();
      plan.waypoints.forEach((w, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(w.tcp[0]), Z(w.tcp[1])));
      ctx.stroke();
      for (const w of plan.waypoints) { ctx.beginPath(); ctx.arc(X(w.tcp[0]), Z(w.tcp[1]), 2, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  const blade = (pose: { x: number; z: number; angle: number }, fill: string | null, stroke: string, dashed = false, flipped = false) => {
    const p = NSE_SCOOP_PROFILE, flip = flipped ? -1 : 1;
    const back = p.tipAhead - p.length;
    ctx.save(); ctx.translate(X(pose.x), Z(pose.z)); ctx.rotate(-pose.angle); ctx.scale(scale, -scale * flip);
    ctx.beginPath(); ctx.rect(back, -p.thickness, p.length, p.thickness); ctx.rect(back - p.heelLength, -p.thickness, p.heelLength, p.thickness + p.heelHeight);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.lineWidth = 1.5 / scale; ctx.setLineDash(dashed ? [4 / scale, 3 / scale] : []); ctx.strokeStyle = stroke; ctx.stroke(); ctx.restore();
  };
  if (current) {
    current.blades.forEach((pose, index) => {
      const flipped = !!path?.blades[index]?.flip;
      blade(current.commands[index], null, '#d9f', true, flipped);
      blade(pose, index ? '#6ec3ffaa' : '#80efc5aa', index ? '#6ec3ff' : '#80efc5', false, flipped);
    });
    ctx.fillStyle = '#dce5ee'; ctx.font = '12px sans-serif';
    ctx.fillText(`${current.t.toFixed(2)} s · ${current.stage} · 힘 ${current.forceN.toFixed(0)} N`, 12, 20);
    $('time').textContent = `${current.t.toFixed(2)} s`;
    $<HTMLInputElement>('scrub').value = String(frame);
    const settle = 1; let elapsed = settle, index = 0;
    if (path) for (let i = 1; i < path.waypoints.length && current.t >= elapsed + path.waypoints[i].duration; i++) { elapsed += path.waypoints[i].duration; index = i; }
    document.querySelectorAll('#steps div').forEach(node => node.classList.toggle('now', Number((node as HTMLElement).dataset.i) === index + (current.t > settle ? 1 : 0)));
  } else if (path) {
    for (const plan of path.blades) blade({ x: plan.waypoints[0].tcp[0], z: plan.waypoints[0].tcp[1], angle: plan.waypoints[0].angle }, '#80efc566', '#80efc5', false, plan.flip);
  }
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
  const names = await api<string[]>('saved');
  const select = $<HTMLSelectElement>('project');
  select.innerHTML = names.map(name => `<option value="${name}">${name.replace(/\/project\.json$/, '')}</option>`).join('');
  const preferred = saved().project as string | undefined;
  select.value = preferred && names.includes(preferred) ? preferred : names[0];
  select.onchange = () => void loadProject(select.value).catch(error => status(error.message));
  $<HTMLSelectElement>('target').onchange = event => void loadScene((event.target as HTMLSelectElement).value).catch(error => status(error.message));
  $('run').onclick = () => void run();
  $('reset').onclick = () => { remember({ params: {} }); renderParams(defaults); };
  $('play').onclick = () => { if (!result) return; if (frame >= result.frames.length - 1) frame = 0; playing = !playing; };
  $<HTMLInputElement>('scrub').oninput = event => { frame = Number((event.target as HTMLInputElement).value); playing = false; draw(); };
  window.addEventListener('resize', draw);
  requestAnimationFrame(tick);
  if (select.value) await loadProject(select.value);
}
init().catch(error => status(error instanceof Error ? error.message : String(error)));
