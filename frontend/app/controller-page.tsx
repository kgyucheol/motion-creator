import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowLeft, Check, Eye, EyeOff, FolderOpen, Play, RotateCcw, Square } from 'lucide-react';
import { RobotScene, type PoseState } from '../lib/robot-scene';
import type { SceneObject, SceneObjectPose } from '../lib/scene-objects';

type StageName = 'ready' | 'grasp' | 'transfer' | 'place' | 'return';
type Stages = { target_object: string; authored_root_at_s?: number; grasp_tolerance_m?: number } & Record<StageName, { start_s: number; end_s: number }>;
type Runtime = { assets: { available: boolean; missing: string[]; invalid: string[]; setup_command: string }; default_stages: Stages };
type TaskEvent = { t: number; kind: string; state: string; reason: string };
type Task = {
  state: string; cycles: number; retries: number; recovery_reason: string; task_time: number; stalled: boolean;
  robot: null | { root_xyz: number[]; tilt_deg: number; fallen: boolean };
  command: null | { nav: number[]; waist_ypr: number[]; arms_max_abs: number };
  target: null | { position: number[]; source: string };
  events: TaskEvent[]; held: boolean; grip_forces_n: Record<'left' | 'right', number>;
  options: { proxy_grasp: boolean; vision_visible: boolean; adaptive_grasp: boolean }; target_object: string;
  planner: { name: string; last_plan_s: number | null; last_error: string };
  authored_target_base: number[]; controller_error: string;
};
type Snapshot = {
  revision: number; phase: string; playing: boolean; error: string; control_mode: string;
  nav: number[]; height: number; torso_rpy: number[]; state: PoseState; reference_state: PoseState;
  scene_objects: SceneObject[]; object_states: Record<string, SceneObjectPose>; task: Task;
};

const STATES = ['INITIALIZE', 'MANUAL_POSITION', 'OBSERVE', 'PLAN_GRASP', 'EXECUTE_GRASP', 'TRANSFER', 'PLACE', 'RETURN'];
const stateLabel: Record<string, string> = {
  INITIALIZE: '준비 자세', MANUAL_POSITION: '위치 잡기', OBSERVE: '관측', PLAN_GRASP: '파지 계획', EXECUTE_GRASP: '파지',
  TRANSFER: '이송', PLACE: '내려놓기', RETURN: '복귀', RECOVERY: '복구·정지',
};
const stageNames: [StageName, string][] = [['ready', '준비'], ['grasp', '파지'], ['transfer', '이송'], ['place', '배치'], ['return', '복귀']];
const filePath = (name: string) => name.split('/').map(encodeURIComponent).join('/');
const fixed = (values: number[] | undefined, digits = 2) => (values ?? []).map(value => value.toFixed(digits)).join(', ');

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(body.detail || `요청 실패 (${response.status})`);
  }
  return response.json();
}

export default function ControllerPage() {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<RobotScene | null>(null);
  const pollTimer = useRef<number | null>(null);
  const polling = useRef(false);
  const sessionId = useRef('');
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [saved, setSaved] = useState<string[]>([]);
  const [choice, setChoice] = useState('');
  const [stages, setStages] = useState<Stages | null>(null);
  const [stagesSource, setStagesSource] = useState('');
  const [objectNames, setObjectNames] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [sourceName, setSourceName] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('런타임을 확인하는 중…');
  const [error, setError] = useState('');
  const [showReference, setShowReference] = useState(false);
  const [options, setOptions] = useState({ proxy_grasp: false, vision_visible: true, adaptive_grasp: true });

  const renderSnapshot = useCallback((next: Snapshot) => {
    setSnapshot(next);
    scene.current?.setSceneObjects(next.scene_objects);
    scene.current?.setObjectPoses(next.object_states);
    scene.current?.update(next.state);
    scene.current?.updateReference(next.reference_state);
  }, []);

  useEffect(() => {
    if (!host.current) return;
    let mounted = true;
    let viewer: RobotScene | null = null;
    Promise.all([
      fetch('/api/init').then(responseJson<{ model_id: string; visual_revision: string }>),
      fetch('/api/controller/runtime').then(responseJson<Runtime>),
      fetch('/api/saved').then(responseJson<string[]>),
    ]).then(([model, nextRuntime, projects]) => {
      if (!mounted || !host.current) return;
      viewer = new RobotScene(host.current, {
        select: () => {}, begin: () => {}, move: () => {}, rotate: () => {}, jointAngle: () => {},
        transformMode: () => {}, end: () => {}, error: value => setError(value),
      }, model.model_id, model.visual_revision);
      viewer.setEditable(false); viewer.showHandles(false); viewer.setReferenceVisible(false); viewer.keyboardEnabled = false;
      scene.current = viewer;
      // Only saved bundles (`<folder>/project.json`) carry the motion files the task stages are cut from.
      const bundles = projects.filter(name => name.includes('/'));
      setRuntime(nextRuntime); setSaved(bundles); setChoice(bundles[0] || '');
      setMessage(nextRuntime.assets.available ? '프로젝트를 선택해 작업을 준비하세요.' : 'Decoupled WBC 자산 설치가 필요합니다.');
    }).catch(failure => setError((failure as Error).message));
    return () => { mounted = false; viewer?.dispose(); scene.current = null; };
  }, []);

  useEffect(() => { scene.current?.setReferenceVisible(showReference); }, [showReference]);

  useEffect(() => {
    const close = () => {
      if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
      if (sessionId.current) void fetch(`/api/controller/sessions/${sessionId.current}`, { method: 'DELETE', keepalive: true });
    };
    window.addEventListener('beforeunload', close);
    return () => { window.removeEventListener('beforeunload', close); close(); };
  }, []);

  // Load the stage mapping and the object names whenever another project is chosen.
  useEffect(() => {
    if (!choice || !runtime) return;
    let cancelled = false;
    Promise.all([
      fetch(`/api/controller/stages?bundle=${encodeURIComponent(choice)}`).then(responseJson<{ stages: Stages; source: string }>),
      fetch(`/api/project/${filePath(choice)}`).then(responseJson<{ scene_objects?: { name: string }[] }>),
    ]).then(([mapping, project]) => {
      if (cancelled) return;
      setStages(mapping.stages); setStagesSource(mapping.source);
      setObjectNames((project.scene_objects || []).map(item => item.name));
      setError('');
    }).catch(failure => { if (!cancelled) { setStages(null); setError((failure as Error).message); } });
    return () => { cancelled = true; };
  }, [choice, runtime]);

  const sendCommand = useCallback(async (action: 'start' | 'stop' | 'reset' | 'operator-reset' | 'options', optionValues?: Record<string, boolean>) => {
    if (!sessionId.current) { setError('먼저 작업을 준비하세요.'); return; }
    try {
      const next = await fetch(`/api/controller/sessions/${sessionId.current}/command`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, options: optionValues }),
      }).then(responseJson<Snapshot>);
      renderSnapshot(next);
      setError(next.error || next.task.controller_error || '');
    } catch (failure) { setError((failure as Error).message); }
  }, [renderSnapshot]);

  function connect(identifier: string) {
    if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
    const poll = async () => {
      if (polling.current || sessionId.current !== identifier) return;
      polling.current = true;
      try {
        const next = await fetch(`/api/controller/sessions/${identifier}`).then(responseJson<Snapshot>);
        renderSnapshot(next);
        if (next.error || next.task.controller_error) setError(next.error || next.task.controller_error);
      } catch (failure) { setError((failure as Error).message); }
      finally { polling.current = false; }
    };
    void poll();
    pollTimer.current = window.setInterval(() => void poll(), 80);
  }

  async function prepare() {
    if (!choice || !stages) return;
    setBusy(true); setError('');
    try {
      const project = await fetch(`/api/project/${filePath(choice)}`).then(responseJson<unknown>);
      if (sessionId.current) await fetch(`/api/controller/sessions/${sessionId.current}`, { method: 'DELETE' });
      const initial = await fetch('/api/controller/sessions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ project, bundle: choice, stages, options }),
      }).then(responseJson<Snapshot & { id: string }>);
      sessionId.current = initial.id; renderSnapshot(initial); setSourceName(choice);
      connect(initial.id); setMessage('준비 완료. Start를 누르면 FSM이 준비 자세부터 작업을 시작합니다.');
    } catch (failure) { setError((failure as Error).message); }
    finally { setBusy(false); }
  }

  function toggleOption(key: 'proxy_grasp' | 'vision_visible' | 'adaptive_grasp') {
    const next = { ...options, [key]: !options[key] };
    setOptions(next);
    if (sessionId.current) void sendCommand('options', { [key]: next[key] });
  }

  function editStage(name: StageName, edge: 'start_s' | 'end_s', value: string) {
    if (!stages) return;
    const number = Number(value);
    if (Number.isNaN(number)) return;
    setStages({ ...stages, [name]: { ...stages[name], [edge]: number } });
  }

  const task = snapshot?.task;
  const ready = runtime?.assets.available && !!snapshot;
  const running = !!snapshot?.playing;
  const current = task?.state ?? 'INITIALIZE';
  const currentIndex = STATES.indexOf(current);
  return <main className="wbc-page">
    <header className="wbc-header">
      <button className="wbc-back" onClick={() => location.assign('/')}><ArrowLeft size={16}/> 모션 편집기</button>
      <div><strong>TASK CONTROLLER</strong><span>작업 FSM · 비전 목업 · decoupled WBC sim2sim</span></div>
      <span className={`wbc-runtime ${runtime?.assets.available ? 'ok' : ''}`}><Activity size={14}/>{runtime?.assets.available ? 'Policy ready' : 'Setup required'}</span>
    </header>
    <aside className="wbc-sidebar">
      {task && <section className="wbc-command-card">
        <h2>작업 상태</h2>
        <div className={`task-state ${current}`}>{stateLabel[current] || current}<small>{current}</small></div>
        <ol className="task-stepper">{STATES.map((name, index) => <li key={name} className={name === current ? 'now' : index < currentIndex ? 'done' : ''}>{stateLabel[name]}</li>)}</ol>
        {current === 'RECOVERY' && <p className="error">{task.recovery_reason}</p>}
        <dl><dt>완료 사이클</dt><dd>{task.cycles}</dd><dt>자동 재시도</dt><dd>{task.retries}</dd>
          <dt>로봇 위치</dt><dd>{fixed(task.robot?.root_xyz)}</dd><dt>기울기</dt><dd>{task.robot ? `${task.robot.tilt_deg}°` : '-'}{task.robot?.fallen ? ' · 낙상' : ''}</dd>
          <dt>인식 pose (base)</dt><dd>{task.target ? fixed(task.target.position, 3) : '-'}</dd>
          <dt>저작 기준 (base)</dt><dd>{fixed(task.authored_target_base, 3)}</dd>
          <dt>파지 계획</dt><dd>{task.planner.name}{task.planner.last_plan_s !== null ? ` · ${task.planner.last_plan_s.toFixed(1)} s` : ''}</dd>
          <dt>파지 접촉력 L / R</dt><dd>{task.grip_forces_n.left.toFixed(1)} / {task.grip_forces_n.right.toFixed(1)} N{task.held ? ' · 잡음' : ''}</dd></dl>
        {task.planner.last_error && <p className="error">{task.planner.last_error}</p>}
        {task.stalled && <p className="error">로봇 상태가 끊겨 작업이 일시정지되었습니다.</p>}
      </section>}
      {task && <section>
        <h2>이벤트</h2>
        <div className="task-events">{task.events.slice().reverse().map((event, index) =>
          <div key={`${event.t}-${index}`} className={event.kind}><span>{event.t.toFixed(1)}s</span><b>{event.state}</b><i>{event.reason}</i></div>)}</div>
      </section>}
      <section>
        <h2>1. 프로젝트와 작업 단계</h2>
        <label>저장 프로젝트<select value={choice} disabled={busy || !runtime?.assets.available} onChange={event => setChoice(event.target.value)}>
          <option value="">프로젝트 선택</option>{saved.map(name => <option key={name} value={name}>{name}</option>)}
        </select></label>
        {stages && <>
          <label className="task-field">집을 물체<select value={stages.target_object} onChange={event => setStages({ ...stages, target_object: event.target.value })}>
            {[...new Set([stages.target_object, ...objectNames])].map(name => <option key={name} value={name}>{name}</option>)}
          </select></label>
          <table className="task-stages"><thead><tr><th>단계</th><th>시작 s</th><th>끝 s</th></tr></thead><tbody>
            {stageNames.map(([name, label]) => <tr key={name}><td>{label}</td>
              {(['start_s', 'end_s'] as const).map(edge => <td key={edge}><input type="number" step="0.1" min="0" value={stages[name][edge]}
                onChange={event => editStage(name, edge, event.target.value)}/></td>)}</tr>)}
          </tbody></table>
          <p>구간은 저장된 모션 번들의 시간입니다. {stagesSource === 'bundle' ? '이 프로젝트의 task_stages.json을 읽었습니다.' : '기본값(configs/sim_motions.json)입니다. 프로젝트에 맞게 고치세요.'}</p>
        </>}
        <button className="wide" disabled={busy || !choice || !stages || !runtime?.assets.available} onClick={() => void prepare()}><FolderOpen size={15}/> 작업 준비</button>
        {sourceName && <p className="wbc-source">SOURCE<br/><b>{sourceName}</b></p>}
      </section>
      <section>
        <h2>2. 테스트 옵션</h2>
        <div className="group-actions">
          <button className={options.adaptive_grasp ? 'chosen' : ''} onClick={() => toggleOption('adaptive_grasp')}>포즈 적응 파지</button>
          <button className={options.proxy_grasp ? 'chosen' : ''} onClick={() => toggleOption('proxy_grasp')}>파지 가정(proxy)</button>
          <button className={!options.vision_visible ? 'chosen' : ''} onClick={() => toggleOption('vision_visible')}>목표 가림</button>
        </div>
        <p>{options.adaptive_grasp ? '관측된 물체 자세와 현재 로봇 자세에서 저작 파지 경로를 다시 계산합니다.' : '저작된 파지 한 가지를 그대로 재생합니다(저작 위치에서 벗어나면 거부).'} {options.proxy_grasp ? '파지 모션이 끝나면 잡힌 것으로 가정합니다. 파지 물리는 검증되지 않습니다.' : '양쪽 툴이 물체를 1 N 이상으로 누를 때만 잡힌 것으로 판정합니다.'} {options.vision_visible ? '' : '비전이 목표를 보지 못해 관측이 실패하는 경우를 시험합니다.'}</p>
      </section>
    </aside>
    <section className="wbc-stage">
      <div ref={host} className="wbc-canvas"/>
      <div className="wbc-stage-label">
        <span>TASK</span> FSM + 비전 목업 <i/> <span>LOWER</span> GR00T WBC · external input
        <button type="button" className="wbc-reference-toggle" aria-pressed={showReference} onClick={() => setShowReference(value => !value)}
          title="편집기에서 컴파일된 저작 모션의 첫 프레임을 반투명하게 겹쳐 표시합니다.">
          {showReference ? <Eye size={13}/> : <EyeOff size={13}/>} 편집기 기준
        </button>
      </div>
      <div className="wbc-hud task-hud">
        <span>작업 <b>{stateLabel[current] || current}</b></span>
        <span>시뮬 <b>{snapshot?.phase ?? 'ready'}</b></span>
        <span>이동 명령 <b>{fixed(task?.command?.nav)}</b></span>
        <span>허리 ypr <b>{fixed(task?.command?.waist_ypr)}</b></span>
        <span>높이 <b>{(snapshot?.height ?? .74).toFixed(2)} m</b></span>
        <span>작업 시간 <b>{(task?.task_time ?? 0).toFixed(1)} s</b></span>
        <span>환경 <b>{snapshot?.scene_objects.length ?? 0} objects</b></span>
      </div>
      {busy && <div className="busy-overlay"><span className="spinner"/> 준비하는 중…</div>}
    </section>
    <footer className="wbc-transport">
      <div className="wbc-controls">
        <button className="primary" disabled={!ready || running} onClick={() => void sendCommand('start')}><Play size={17}/> Start</button>
        <button disabled={!ready || !running} onClick={() => void sendCommand('stop')}><Square size={15}/> Stop</button>
        <button disabled={!ready} onClick={() => void sendCommand('reset')}><RotateCcw size={15}/> Reset</button>
        <button disabled={!ready || current !== 'RECOVERY'} onClick={() => void sendCommand('operator-reset')}
          title="정지 원인을 확인한 뒤 준비 자세부터 다시 시작합니다. 낙상 후에는 Reset이 먼저 필요합니다."><Check size={15}/> 작업자 확인</button>
      </div>
      <div className="wbc-status"><span className={error ? 'error' : ''}>{error || message}</span></div>
    </footer>
    {runtime && !runtime.assets.available && <div className="wbc-setup"><h2>Decoupled WBC 자산 설치 필요</h2><p>GR00T 정책, WBC 모델과 메시를 검증된 로컬 폴더로 복사한 뒤 다시 실행하세요.</p><code>{runtime.assets.setup_command}</code><small>누락: {[...runtime.assets.missing, ...runtime.assets.invalid].join(', ')}</small></div>}
  </main>;
}
