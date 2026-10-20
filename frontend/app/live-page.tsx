import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowLeft } from 'lucide-react';
import { RobotScene, type PoseState } from '../lib/robot-scene';
import { ReplayPanel } from './ReplayPanel';

type Project = { name?: string; keyframes: { name: string }[] };
type Status = { mode: 'real' | 'mock' | null; connected: boolean; robot: { age: number; tilt_deg: number; arms: number[] } | null };

const ARM_NAMES = ['어깨 피치', '어깨 롤', '어깨 요', '팔꿈치', '손목 롤', '손목 피치', '손목 요'];
const ARM_QPOS = 7 + 15;                     // left 7 then right 7 arm joints in qpos (Unitree order)
const filePath = (name: string) => name.split('/').map(encodeURIComponent).join('/');

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(body.detail || `요청 실패 (${response.status})`);
  }
  return response.json();
}

/**
 * Live robot: the real G1 (or a simulated one) stands and walks on Unitree's own controller while a saved motion drives the
 * arms over rt/arm_sdk. The run controls are the editor's replay panel; this page adds the motion picker and a live view of
 * the measured arms, to check that commands reach the robot.
 */
export default function LivePage() {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<RobotScene | null>(null);
  const home = useRef<number[] | null>(null);
  const [saved, setSaved] = useState<string[]>([]);
  const [choice, setChoice] = useState('');
  const [project, setProject] = useState<Project | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [checked, setChecked] = useState('');          // the project the plan on the server was checked from (this page only)
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    let viewer: RobotScene | null = null;
    Promise.all([
      fetch('/api/init').then(responseJson<{ model_id: string; visual_revision: string; state: PoseState }>),
      fetch('/api/saved').then(responseJson<string[]>),
    ]).then(([model, projects]) => {
      if (!mounted || !host.current) return;
      viewer = new RobotScene(host.current, {
        select: () => {}, begin: () => {}, move: () => {}, rotate: () => {}, jointAngle: () => {},
        transformMode: () => {}, end: () => {}, error: value => setError(value),
      }, model.model_id, model.visual_revision);
      viewer.setEditable(false); viewer.showHandles(false); viewer.setReferenceVisible(false); viewer.keyboardEnabled = false;
      viewer.update(model.state);
      scene.current = viewer;
      home.current = model.state.qpos;
      setSaved(projects); setChoice(projects[0] || '');
    }).catch(failure => setError((failure as Error).message));
    return () => { mounted = false; viewer?.dispose(); scene.current = null; };
  }, []);

  useEffect(() => {
    if (!choice) return;
    let cancelled = false;
    fetch(`/api/project/${filePath(choice)}`).then(responseJson<Project>)
      .then(next => { if (!cancelled) { setProject(next); setError(''); } })
      .catch(failure => { if (!cancelled) { setProject(null); setError((failure as Error).message); } });
    return () => { cancelled = true; };
  }, [choice]);

  // The measured arms on the robot model (legs and waist at the default stance: the robot's controller owns them).
  useEffect(() => {
    let busy = false;
    const timer = window.setInterval(async () => {
      if (busy) return;
      busy = true;
      try {
        const next = await fetch('/api/replay/status').then(responseJson<Status>);
        setStatus(next);
        if (next.robot && home.current && scene.current) {
          const qpos = [...home.current];
          next.robot.arms.forEach((value, index) => { qpos[ARM_QPOS + index] = value; });
          const pose = await fetch('/api/pose', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ qpos }) })
            .then(responseJson<PoseState>);
          scene.current?.update(pose);
        }
      } catch { /* the server may be restarting: keep the last picture */ }
      finally { busy = false; }
    }, 250);
    return () => window.clearInterval(timer);
  }, []);

  const arms = status?.robot?.arms;
  const live = !!status?.connected && !!status.robot;
  return <main className="wbc-page live-page">
    <header className="wbc-header">
      <button className="wbc-back" onClick={() => location.assign('/')}><ArrowLeft size={16}/> 모션 편집기</button>
      <div><strong>LIVE ROBOT</strong><span>하체 Unitree 기본 컨트롤러 · 상체 rt/arm_sdk · unitree_sdk2</span></div>
      <span className={`wbc-runtime ${live ? 'ok' : ''}`}><Activity size={14}/>{!status?.connected ? '연결 안 됨' : status.mode === 'real' ? '실제 로봇' : '가상 로봇'}</span>
    </header>
    <aside className="wbc-sidebar">
      <section>
        <h2>모션</h2>
        <label>저장된 프로젝트
          <select aria-label="저장된 프로젝트" value={choice} onChange={event => setChoice(event.target.value)}>
            {saved.map(name => <option key={name} value={name}>{name.split('/')[0]}</option>)}
            {!saved.length && <option value="">저장된 프로젝트 없음</option>}</select></label>
        {project && <ol className="live-cards">{project.keyframes.map((frame, index) => <li key={index}>{frame.name}</li>)}</ol>}
        <p>여기서 고른 모션을 오른쪽 패널의 <b>모션 점검</b>으로 확인한 뒤 실행합니다. 편집 중인 모션은 편집기에서 저장해야 목록에 나타납니다.</p>
      </section>
      <section>
        <h2>측정된 팔 관절 <small>°</small></h2>
        {arms ? <table className="live-joints"><thead><tr><th>관절</th><th>왼팔</th><th>오른팔</th></tr></thead>
          <tbody>{ARM_NAMES.map((name, index) => <tr key={name}><th>{name}</th>
            <td>{(arms[index] * 180 / Math.PI).toFixed(1)}</td><td>{(arms[7 + index] * 180 / Math.PI).toFixed(1)}</td></tr>)}</tbody></table>
          : <p>연결하면 rt/lowstate의 팔 관절 측정값이 여기에 표시됩니다.</p>}
        {status?.robot && <p>상태 지연 {status.robot.age.toFixed(2)} s · 몸통 기울기 {status.robot.tilt_deg.toFixed(1)}°</p>}
      </section>
      {error && <p className="error">{error}</p>}
    </aside>
    <section className="wbc-stage">
      <div ref={host} className="wbc-canvas"/>
      <div className="wbc-stage-label"><span>VIEW</span> 측정된 팔 자세 <i/> 다리·허리는 기본 자세로 표시 (로봇 컨트롤러가 담당)</div>
    </section>
    <aside className="live-control">
      {project
        ? <ReplayPanel inline fps={30} frameNames={project.keyframes.map(frame => frame.name)} getProject={() => project}
            onPrepared={() => setChecked(choice)}
            startBlocked={checked === choice ? '' : `지금 고른 모션(${choice.split('/')[0]})을 아직 점검하지 않았습니다. "현재 프로젝트 점검"을 먼저 누르세요.`}/>
        : <p className="live-empty">모션을 고르면 연결·점검·실행 패널이 나타납니다.</p>}
    </aside>
  </main>;
}
