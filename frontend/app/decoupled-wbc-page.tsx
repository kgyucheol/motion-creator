import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowLeft, Download, FolderOpen, Play, RotateCcw, Save, Square } from 'lucide-react';
import { RobotScene, type PoseState } from '../lib/robot-scene';

type Runtime = { available: boolean; missing: string[]; invalid: string[]; source_revision: string; setup_command: string };
type Snapshot = {
  revision: number; phase: string; playing: boolean; error: string; policy: 'balance' | 'walk';
  nav: number[]; height: number; torso_rpy: number[]; upper_time: number; upper_duration: number;
  recording_frames: number; state: PoseState;
};
type SaveResult = { folder: string; files: string[] };

const filePath = (name: string) => name.split('/').map(encodeURIComponent).join('/');
const phaseLabel: Record<string, string> = { ready: '준비', settling: '상체 준비', playing: '재생·녹화', paused: '일시정지', fallen: '낙상 감지' };

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

  const sendCommand = useCallback(async (action: 'play' | 'stop' | 'reset' | 'key', key = '') => {
    if (!sessionId.current) { setError('먼저 모션을 불러오세요.'); return; }
    if (action === 'reset') { setResult(null); setError(''); }
    try {
      const next = await fetch(`/api/decoupled-wbc/sessions/${sessionId.current}/command`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, key }),
      }).then(responseJson<Snapshot>);
      setSnapshot(next); scene.current?.update(next.state);
      if (next.error) setError(next.error);
    } catch (failure) { setError((failure as Error).message); }
  }, []);

  useEffect(() => {
    if (!host.current) return;
    const viewer = new RobotScene(host.current, {
      select: () => {}, begin: () => {}, move: () => {}, rotate: () => {}, jointAngle: () => {},
      transformMode: () => {}, end: () => {}, error: value => setError(value),
    });
    viewer.setEditable(false); viewer.showHandles(false); viewer.keyboardEnabled = false;
    scene.current = viewer;
    Promise.all([
      fetch('/api/decoupled-wbc/runtime').then(responseJson<Runtime>),
      fetch('/api/saved').then(responseJson<string[]>),
    ]).then(([nextRuntime, projects]) => {
      setRuntime(nextRuntime); setSaved(projects); setChoice(projects[0] || '');
      setMessage(nextRuntime.available ? '모션을 선택해 시뮬레이션을 준비하세요.' : 'Decoupled WBC 자산 설치가 필요합니다.');
    }).catch(failure => setError((failure as Error).message));
    return () => { viewer.dispose(); scene.current = null; };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.repeat || event.isComposing || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const key = event.key.toLowerCase();
      if (!['w', 's', 'a', 'd', 'q', 'e', '1', '2', 'z'].includes(key)) return;
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
        setSnapshot(next); scene.current?.update(next.state);
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
      sessionId.current = initial.id; setSnapshot(initial); setSourceName(name); scene.current?.update(initial.state);
      connect(initial.id); setMessage('준비 완료. Play를 누르면 2초 동안 상체를 준비한 뒤 모션 재생과 녹화를 시작합니다.');
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
      <div><strong>DECOUPLED WBC</strong><span>작성 모션 상체 · 키보드 텔레옵 하체</span></div>
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
        <h2>2. 하체 키보드 제어</h2>
        <div className="wbc-keys"><kbd>W</kbd><span>전진</span><kbd>S</kbd><span>후진</span><kbd>A / D</kbd><span>좌 / 우</span><kbd>Q / E</kbd><span>좌 / 우 회전</span><kbd>1 / 2</kbd><span>높이 ±</span><kbd>Z</kbd><span>명령 초기화</span></div>
        <p>키를 누를 때마다 속도가 단계적으로 변합니다. Stop을 누르면 이동 속도는 즉시 0이 됩니다.</p>
      </section>
      <section className="wbc-command-card">
        <h2>현재 명령</h2>
        <dl><dt>전후</dt><dd>{nav[0].toFixed(2)} m/s</dd><dt>좌우</dt><dd>{nav[1].toFixed(2)} m/s</dd><dt>회전</dt><dd>{nav[2].toFixed(2)} rad/s</dd><dt>높이</dt><dd>{(snapshot?.height ?? .74).toFixed(2)} m</dd></dl>
      </section>
    </aside>
    <section className="wbc-stage">
      <div ref={host} className="wbc-canvas"/>
      <div className="wbc-stage-label"><span>UPPER</span> authored arms <i/> <span>LOWER</span> GR00T WBC</div>
      <div className="wbc-hud">
        <span>상태 <b>{phaseLabel[snapshot?.phase ?? 'ready'] || snapshot?.phase}</b></span>
        <span>정책 <b>{snapshot?.policy ?? 'balance'}</b></span>
        <span>상체 <b>{(snapshot?.upper_time ?? 0).toFixed(2)} / {(snapshot?.upper_duration ?? 0).toFixed(2)} s</b></span>
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
