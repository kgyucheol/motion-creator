import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowLeft, Boxes, Download, Eye, EyeOff, FolderOpen, Play, RotateCcw, Save, Square } from 'lucide-react';
import { RobotScene, type PoseState } from '../lib/robot-scene';
import type { SceneObject, SceneObjectPose } from '../lib/scene-objects';

type Runtime = { available: boolean; missing: string[]; invalid: string[]; source_revision: string; setup_command: string };
type ArmJoint = 'shoulder_pitch' | 'shoulder_roll' | 'shoulder_yaw' | 'elbow' | 'wrist_roll' | 'wrist_pitch' | 'wrist_yaw';
type JointTorque = { pose: number; bias: number; normal: number; orientation: number;
  vertical_support: number; feedback: number; command: number; limit: number; utilization_percent: number };
type Snapshot = {
  revision: number; phase: string; playing: boolean; error: string; policy: 'balance' | 'walk';
  control_mode: 'auto' | 'manual';
  nav: number[]; height: number; torso_rpy: number[]; upper_time: number; upper_duration: number;
  recording_frames: number; state: PoseState; scene_objects: SceneObject[];
  reference_state: PoseState;
  tracking: { lower_rmse_deg: number; upper_rmse_deg: number; upper_max_deg: number;
    root_error_m: number; yaw_error_deg: number };
  object_states: Record<string, SceneObjectPose>;
  grasp: null | { object_id: string; active: boolean; target_force_n: number;
    verification_force_n: number; object_mass_kg: number; effective_friction: number;
    bilateral_contact: boolean; normal_force_n: Record<'left' | 'right', number>;
    force_feedback: { enabled: boolean; engaged: boolean; torque_saturated: boolean;
      filtered_normal_force_n: Record<'left' | 'right', number>;
      correction_force_n: Record<'left' | 'right', number>;
      feedback_torque_nm: Record<'left' | 'right', number>;
      orientation_error_deg: Record<'left' | 'right', number>;
      orientation_error_rpy_deg: Record<'left' | 'right', Record<'roll' | 'pitch' | 'yaw', number>>;
      orientation_feedback_torque_nm: Record<'left' | 'right', number>;
      vertical_support: { target_height_m: number; height_error_m: number; target_velocity_mps: number;
        object_velocity_mps: number; force_n_per_hand: number };
      joint_torque_nm: Record<'left' | 'right', Record<ArmJoint, JointTorque>> } };
};
type SaveResult = { folder: string; files: string[] };

const filePath = (name: string) => name.split('/').map(encodeURIComponent).join('/');
const phaseLabel: Record<string, string> = { ready: '준비', settling: '상체 준비', playing: '재생·녹화', paused: '일시정지', fallen: '낙상 감지' };
const armJoints: [ArmJoint, string][] = [['shoulder_pitch', '어깨 P'], ['shoulder_roll', '어깨 R'],
  ['shoulder_yaw', '어깨 Y'], ['elbow', '팔꿈치'], ['wrist_roll', '손목 R'],
  ['wrist_pitch', '손목 P'], ['wrist_yaw', '손목 Y']];
const signed = (value: number, digits = 1) => `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(body.detail || `요청 실패 (${response.status})`);
  }
  return response.json();
}

export default function DecoupledWbcPage() {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<RobotScene | null>(null);
  const pollTimer = useRef<number | null>(null);
  const polling = useRef(false);
  const controlMode = useRef<'auto' | 'manual'>('auto');
  const file = useRef<HTMLInputElement>(null);
  const sessionId = useRef('');
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [saved, setSaved] = useState<string[]>([]);
  const [choice, setChoice] = useState('');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [sourceName, setSourceName] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('런타임을 확인하는 중…');
  const [error, setError] = useState('');
  const [result, setResult] = useState<SaveResult | null>(null);
  const [showReference, setShowReference] = useState(true);
  const [showCollisions, setShowCollisions] = useState(false);

  const renderSnapshot = useCallback((next: Snapshot) => {
    controlMode.current = next.control_mode;
    setSnapshot(next);
    scene.current?.setSceneObjects(next.scene_objects);
    scene.current?.setObjectPoses(next.object_states);
    scene.current?.update(next.state);
    scene.current?.updateReference(next.reference_state);
  }, []);

  const sendCommand = useCallback(async (action: 'play' | 'stop' | 'reset' | 'key' | 'mode' | 'grasp-force', key = '', mode?: 'auto' | 'manual', enabled?: boolean) => {
    if (!sessionId.current) { setError('먼저 모션을 불러오세요.'); return; }
    if (action === 'reset') { setResult(null); setError(''); }
    try {
      const next = await fetch(`/api/decoupled-wbc/sessions/${sessionId.current}/command`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, key, mode, enabled }),
      }).then(responseJson<Snapshot>);
      renderSnapshot(next);
      if (next.error) setError(next.error);
    } catch (failure) { setError((failure as Error).message); }
  }, [renderSnapshot]);

  useEffect(() => {
    if (!host.current) return;
    let mounted = true;
    let viewer: RobotScene | null = null;
    Promise.all([
      fetch('/api/init').then(responseJson<{ model_id: string; visual_revision: string }>),
      fetch('/api/decoupled-wbc/runtime').then(responseJson<Runtime>),
      fetch('/api/saved').then(responseJson<string[]>),
    ]).then(([model, nextRuntime, projects]) => {
      if (!mounted || !host.current) return;
      viewer = new RobotScene(host.current, {
        select: () => {}, begin: () => {}, move: () => {}, rotate: () => {}, jointAngle: () => {},
        transformMode: () => {}, end: () => {}, error: value => setError(value),
      }, model.model_id, model.visual_revision);
      viewer.setEditable(false); viewer.showHandles(false); viewer.setReferenceVisible(true); viewer.keyboardEnabled = false;
      scene.current = viewer;
      setRuntime(nextRuntime); setSaved(projects); setChoice(projects[0] || '');
      setMessage(nextRuntime.available ? '모션을 선택해 시뮬레이션을 준비하세요.' : 'Decoupled WBC 자산 설치가 필요합니다.');
    }).catch(failure => setError((failure as Error).message));
    return () => { mounted = false; viewer?.dispose(); scene.current = null; };
  }, []);

  useEffect(() => { scene.current?.setReferenceVisible(showReference); }, [showReference]);
  useEffect(() => { scene.current?.setCollisionProxiesVisible(showCollisions); }, [showCollisions]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.repeat || event.isComposing || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const key = event.key.toLowerCase();
      if (!['w', 's', 'a', 'd', 'q', 'e', '1', '2', '3', '4', '5', '6', '7', '8', 'z'].includes(key)) return;
      if (controlMode.current !== 'manual') return;
      event.preventDefault();
      void sendCommand('key', key);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sendCommand]);

  useEffect(() => {
    const close = () => {
      if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
      if (sessionId.current) void fetch(`/api/decoupled-wbc/sessions/${sessionId.current}`, { method: 'DELETE', keepalive: true });
    };
    window.addEventListener('beforeunload', close);
    return () => { window.removeEventListener('beforeunload', close); close(); };
  }, []);

  function connect(identifier: string) {
    if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
    const poll = async () => {
      if (polling.current || sessionId.current !== identifier) return;
      polling.current = true;
      try {
        const next = await fetch(`/api/decoupled-wbc/sessions/${identifier}`).then(responseJson<Snapshot>);
        renderSnapshot(next);
        if (next.error) setError(next.error);
      } catch (failure) {
        setError((failure as Error).message);
      } finally {
        polling.current = false;
      }
    };
    void poll();
    pollTimer.current = window.setInterval(() => void poll(), 50);
  }

  async function create(project: unknown, name: string) {
    setBusy(true); setError(''); setResult(null);
    try {
      if (sessionId.current) await fetch(`/api/decoupled-wbc/sessions/${sessionId.current}`, { method: 'DELETE' });
      const response = await fetch('/api/decoupled-wbc/sessions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ project }),
      });
      const initial = await responseJson<Snapshot & { id: string }>(response);
      sessionId.current = initial.id; renderSnapshot(initial); setSourceName(name);
      connect(initial.id); setMessage('준비 완료. Auto는 고스트 루트 궤적을 WBC 입력 명령으로 추종합니다.');
    } catch (failure) { setError((failure as Error).message); }
    finally { setBusy(false); }
  }

  async function loadSaved() {
    if (!choice) return;
    setBusy(true); setError('');
    try {
      const project = await fetch(`/api/project/${filePath(choice)}`).then(responseJson<unknown>);
      await create(project, choice);
    } catch (failure) { setError((failure as Error).message); setBusy(false); }
  }

  async function loadFile(source: File) {
    setBusy(true); setError('');
    try {
      let project: unknown;
      if (source.name.toLowerCase().endsWith('.json')) project = JSON.parse(await source.text());
      else {
        const response = await fetch(`/api/import-motion?filename=${encodeURIComponent(source.name)}&fps=30`, { method: 'POST', body: source });
        project = await responseJson<unknown>(response);
      }
      await create(project, source.name);
    } catch (failure) { setError((failure as Error).message); setBusy(false); }
  }

  async function saveRecording() {
    if (!sessionId.current) return;
    setBusy(true); setError('');
    try {
      const savedResult = await fetch(`/api/decoupled-wbc/sessions/${sessionId.current}/save`, { method: 'POST' }).then(responseJson<SaveResult>);
      setResult(savedResult); setMessage(`녹화 결과를 motions/${savedResult.folder}에 저장했습니다.`);
    } catch (failure) { setError((failure as Error).message); }
    finally { setBusy(false); }
  }

  const ready = runtime?.available && !!snapshot;
  const nav = snapshot?.nav ?? [0, 0, 0];
  return <main className="wbc-page">
    <header className="wbc-header">
      <button className="wbc-back" onClick={() => location.assign('/')}><ArrowLeft size={16}/> 모션 편집기</button>
      <div><strong>DECOUPLED WBC</strong><span>작성 모션 상체 · 입력 기반 하체·허리 제어</span></div>
      <span className={`wbc-runtime ${runtime?.available ? 'ok' : ''}`}><Activity size={14}/>{runtime?.available ? 'Policy ready' : 'Setup required'}</span>
    </header>
    <aside className="wbc-sidebar">
      <section>
        <h2>1. 상체 모션 불러오기</h2>
        <label>저장 프로젝트<select value={choice} disabled={busy || !runtime?.available} onChange={event => setChoice(event.target.value)}>
          <option value="">프로젝트 선택</option>{saved.map(name => <option key={name} value={name}>{name}</option>)}
        </select></label>
        <button className="wide" disabled={busy || !choice || !runtime?.available} onClick={() => void loadSaved()}><FolderOpen size={15}/> 선택 모션 준비</button>
        <button className="wide" disabled={busy || !runtime?.available} onClick={() => file.current?.click()}><FolderOpen size={15}/> JSON / NPZ / CSV 열기</button>
        <input ref={file} hidden type="file" accept=".json,.npz,.csv" onChange={event => { const selected = event.target.files?.[0]; if (selected) void loadFile(selected); event.target.value = ''; }}/>
        {sourceName && <p className="wbc-source">SOURCE<br/><b>{sourceName}</b></p>}
      </section>
      <section>
        <h2>2. 제어 모드</h2>
        <div className="group-actions"><button className={snapshot?.control_mode === 'auto' ? 'chosen' : ''} disabled={!ready} onClick={() => void sendCommand('mode', '', 'auto')}>Auto · 고스트 추종</button><button className={snapshot?.control_mode === 'manual' ? 'chosen' : ''} disabled={!ready} onClick={() => void sendCommand('mode', '', 'manual')}>키보드</button></div>
        {snapshot?.control_mode === 'auto' ? <p>고스트의 이동·회전·높이를 WBC의 속도/높이 입력으로 변환합니다. 허리 자세도 몸통 RPY 입력으로 전달하며 관절 상태를 직접 덮어쓰지 않습니다. 다리의 정확한 프레임 자세보다 이동 궤적을 추종하며 보행 형태는 정책이 결정합니다.</p> : <>
          <div className="wbc-keys"><kbd>W / S</kbd><span>전진 / 후진</span><kbd>A / D</kbd><span>좌 / 우</span><kbd>Q / E</kbd><span>좌 / 우 회전</span><kbd>1 / 2</kbd><span>높이 ±</span><kbd>3 / 4</kbd><span>허리 롤 ±</span><kbd>5 / 6</kbd><span>허리 피치 ±</span><kbd>7 / 8</kbd><span>허리 요 ±</span><kbd>Z</kbd><span>명령 초기화</span></div>
          <p>허리 키는 작성 모션의 허리 자세에 정책 입력 오프셋을 더합니다.</p>
        </>}
      </section>
      <section className="wbc-command-card">
        <h2>현재 명령</h2>
        <dl><dt>전후</dt><dd>{nav[0].toFixed(2)} m/s</dd><dt>좌우</dt><dd>{nav[1].toFixed(2)} m/s</dd><dt>회전</dt><dd>{nav[2].toFixed(2)} rad/s</dd><dt>높이</dt><dd>{(snapshot?.height ?? .74).toFixed(2)} m</dd><dt>허리 R/P/Y</dt><dd>{(snapshot?.torso_rpy ?? [0, 0, 0]).map(value => `${(value * 180 / Math.PI).toFixed(0)}°`).join(' / ')}</dd></dl>
      </section>
      {snapshot?.grasp && <section className="wbc-command-card">
        <h2>파지 연결 제어</h2>
        <dl><dt>물체 질량</dt><dd>{snapshot.grasp.object_mass_kg.toFixed(2)} kg</dd><dt>보수 마찰계수</dt><dd>{snapshot.grasp.effective_friction.toFixed(2)}</dd><dt>자동 목표</dt><dd>{snapshot.grasp.target_force_n.toFixed(1)} N / 손</dd><dt>상태</dt><dd>{!snapshot.grasp.force_feedback.enabled ? '꺼짐' : snapshot.grasp.force_feedback.engaged ? '접촉 피드백' : '접촉 대기'}</dd></dl>
        {snapshot.grasp.force_feedback.engaged && <div className="wbc-feedback-summary">
          {(['left', 'right'] as const).map(side => { const error = snapshot.grasp!.force_feedback.orientation_error_rpy_deg[side]; return <div key={side}><b>{side === 'left' ? '왼손' : '오른손'} R/P/Y</b><span>{signed(error.roll)}° / {signed(error.pitch)}° / {signed(error.yaw)}°</span></div>; })}
          <div><b>수직 지지</b><span>{snapshot.grasp.force_feedback.vertical_support.force_n_per_hand.toFixed(1)} N/손 · 오차 {signed(snapshot.grasp.force_feedback.vertical_support.height_error_m * 1000, 0)} mm</span></div>
        </div>}
        <button className={`wide ${snapshot.grasp.force_feedback.enabled ? 'chosen' : ''}`} disabled={!ready} onClick={() => void sendCommand('grasp-force', '', undefined, !snapshot.grasp!.force_feedback.enabled)}>힘·파지면 방향 피드백 {snapshot.grasp.force_feedback.enabled ? 'ON' : 'OFF'}</button>
        <p>접촉 후 박스 면 방향과 편집기 손 궤적의 높이를 추종합니다. 물체 질량의 중력을 양팔에 분배하며 weld 제약은 사용하지 않습니다. 편집기의 검증 최소 힘 {snapshot.grasp.verification_force_n.toFixed(1)} N과 별개입니다.</p>
        {snapshot.grasp.force_feedback.engaged && <details className="wbc-feedback-details"><summary>관절별 토크 진단</summary>
          {(['left', 'right'] as const).map(side => <div className="wbc-torque-table-wrap" key={side}><b>{side === 'left' ? '왼팔' : '오른팔'}</b><table><thead><tr><th>관절</th><th>명령/한계</th><th>자세</th><th>방향</th><th>지지</th></tr></thead><tbody>
            {armJoints.map(([joint, label]) => { const value = snapshot.grasp!.force_feedback.joint_torque_nm[side][joint]; return <tr className={value.utilization_percent >= 95 ? 'saturated' : ''} key={joint}><td>{label}</td><td>{signed(value.command)} / {value.limit.toFixed(0)}<small>{value.utilization_percent.toFixed(0)}%</small></td><td>{signed(value.pose)}</td><td>{signed(value.orientation)}</td><td>{signed(value.vertical_support)}</td></tr>; })}
          </tbody></table></div>)}
          <p>단위 Nm. 명령은 자세·중력보상·파지 피드백을 합친 최종 모터 토크입니다.</p>
        </details>}
        {snapshot.grasp.force_feedback.torque_saturated && <p className="error">모터 토크 포화: 파지면 간격을 늘리거나 자세를 다시 맞추세요.</p>}
      </section>}
    </aside>
    <section className="wbc-stage">
      <div ref={host} className="wbc-canvas"/>
      <div className="wbc-stage-label">
        <span>UPPER</span> authored arms <i/> <span>LOWER</span> GR00T WBC · {snapshot?.control_mode === 'auto' ? 'auto ghost input' : 'keyboard input'}
        <button type="button" className="wbc-reference-toggle" aria-pressed={showReference}
          onClick={() => setShowReference(value => !value)} title="편집기에서 컴파일된 원본 모션을 반투명하게 겹쳐 표시합니다.">
          {showReference ? <Eye size={13}/> : <EyeOff size={13}/>} 편집기 기준
        </button>
        <button type="button" className="wbc-reference-toggle" aria-pressed={showCollisions}
          onClick={() => setShowCollisions(value => !value)} title="장면 오브젝트의 GLB 외형을 숨기고 WBC가 사용하는 물리 충돌체만 표시합니다.">
          <Boxes size={13}/> 오브젝트 콜리전
        </button>
      </div>
      <div className="wbc-hud">
        <span>상태 <b>{phaseLabel[snapshot?.phase ?? 'ready'] || snapshot?.phase}</b></span>
        <span>정책 <b>{snapshot?.policy ?? 'balance'}</b></span>
        <span>상체 <b>{(snapshot?.upper_time ?? 0).toFixed(2)} / {(snapshot?.upper_duration ?? 0).toFixed(2)} s</b></span>
        <span>상체 추종 오차 <b>{(snapshot?.tracking.upper_rmse_deg ?? 0).toFixed(1)}° RMS</b></span>
        <span>하체 정책 차이 <b>{(snapshot?.tracking.lower_rmse_deg ?? 0).toFixed(1)}° RMS</b></span>
        {snapshot?.control_mode === 'auto' && <span>루트 추종 <b>{((snapshot?.tracking.root_error_m ?? 0) * 100).toFixed(1)} cm · {(snapshot?.tracking.yaw_error_deg ?? 0).toFixed(1)}°</b></span>}
        <span>환경 <b>{snapshot?.scene_objects.length ?? 0} objects</b></span>
        {snapshot?.grasp && <span>파지 <b>{snapshot.grasp.bilateral_contact
          ? `L ${snapshot.grasp.normal_force_n.left.toFixed(1)} · R ${snapshot.grasp.normal_force_n.right.toFixed(1)} / 목표 ${snapshot.grasp.target_force_n.toFixed(1)} N`
          : snapshot.grasp.active ? '편집기 파지 자세 · 접촉 대기' : '대기'}</b></span>}
        <span>녹화 <b>{snapshot?.recording_frames ?? 0} frames</b></span>
      </div>
      {busy && <div className="busy-overlay"><span className="spinner"/> 준비하는 중…</div>}
    </section>
    <footer className="wbc-transport">
      <div className="wbc-controls">
        <button className="primary" disabled={!ready || snapshot?.playing} onClick={() => void sendCommand('play')}><Play size={17}/> Play · Record</button>
        <button disabled={!ready || !snapshot?.playing} onClick={() => void sendCommand('stop')}><Square size={15}/> Stop</button>
        <button disabled={!ready} onClick={() => void sendCommand('reset')}><RotateCcw size={15}/> Reset</button>
        <button disabled={!ready || !snapshot?.recording_frames || busy} onClick={() => void saveRecording()}><Save size={15}/> 녹화 저장</button>
      </div>
      <div className="wbc-status"><span className={error ? 'error' : ''}>{error || message}</span>{result && <span className="wbc-downloads">{result.files.map(name => <a key={name} href={`/api/files/${filePath(name)}`}><Download size={13}/>{name.split('/').at(-1)}</a>)}</span>}</div>
    </footer>
    {runtime && !runtime.available && <div className="wbc-setup"><h2>Decoupled WBC 자산 설치 필요</h2><p>GR00T 정책, WBC 모델과 메시를 검증된 로컬 폴더로 복사한 뒤 다시 실행하세요.</p><code>{runtime.setup_command}</code><small>누락: {[...runtime.missing, ...runtime.invalid].join(', ')}</small></div>}
  </main>;
}
