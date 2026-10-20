'use client';
import { useCallback, useEffect, useRef, useState } from 'react';

type Item = { level: 'ok' | 'warn' | 'bad'; name: string; text: string };
type Summary = { duration: number; frames: number; fps: number; waist: boolean; items: Item[]; ok: boolean; speed_cap: number; speed_hard: number; warnings: number };
type PlayerStatus = {
  state: 'idle' | 'engaging' | 'approach' | 'playing' | 'paused' | 'stopped' | 'finished' | 'returning' | 'moving' | 'holding';
  time: number; duration: number; speed: number; phase: number | null; keyframe_times: number[]; fault: string | null; fast: boolean; backend: string | null;
};
type Battery = { soc: number; voltage: number; current: number; temp: number | null; cycle: number };
type Status = {
  mode: 'real' | 'mock' | null; interface: string | null; connected: boolean; read_only: boolean;
  robot: { age: number; tilt_deg: number; q?: number[]; temp?: number[] | null; tau?: number[] | null; battery?: Battery | null } | null;
  player: PlayerStatus; plan: Summary | null; temperature_limits?: { warning: number; critical: number };
  battery_estimate?: { percent_per_hour: number; remaining_s: number; measured_s: number } | null;
};
type Interface = { name: string; ip: string; wifi: boolean; robot: boolean };

const STATE_TEXT: Record<PlayerStatus['state'], string> = {
  idle: '대기', engaging: '제어권 넘겨받는 중', approach: '시작 자세로 이동 중', playing: '재생 중', paused: '일시정지',
  stopped: '정지 (마지막 목표를 유지 중)', finished: '재생 끝', returning: '시작 자세로 복귀 중',
  moving: '자세로 이동 중', holding: '보낸 자세 유지 중',
};
const ACTIVE = ['engaging', 'approach', 'playing', 'paused', 'stopped', 'finished', 'returning', 'moving', 'holding'];
const SETTLED = ['stopped', 'finished', 'holding'];          // engaged, holding a fixed target: a pose or a replay may follow
/** A pose the editor can send: the arms (and waist) of `qpos`, or why it cannot be sent now. */
export type PoseSource = { label: string; qpos: number[] | null; reason?: string };

async function call<T>(path: string, body?: unknown, method = body === undefined ? 'GET' : 'POST'): Promise<T> {
  const response = await fetch(`/api/replay/${path}`, { method, ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
  if (!response.ok) {
    const failure = await response.json().catch(() => ({})) as { detail?: unknown };
    throw new Error(typeof failure.detail === 'string' ? failure.detail : `요청 실패 (${response.status})`);
  }
  return response.json();
}

/**
 * Replay of the current project on a real G1 under Unitree's default policy (arms over rt/arm_sdk), or on a simulated robot.
 * Nothing is sent to a real robot until a connection exists, the preflight passed and the operator confirmed.
 */
export function ReplayPanel({ getProject, fps, frameNames, onClose, inline = false, onPrepared, startBlocked = '', poses }: {
  getProject: () => unknown; fps: number; frameNames: string[]; onClose?: () => void; inline?: boolean;
  /** called after the current project was checked (the plan on the server is now this project's) */
  onPrepared?: () => void;
  /** a reason the start button stays off (e.g. the checked plan belongs to another project) */
  startBlocked?: string;
  /** the editor: poses that can be sent to the robot one at a time (the edited pose, the selected card's) */
  poses?: () => PoseSource[];
}) {
  const [status, setStatus] = useState<Status | null>(null);
  const [interfaces, setInterfaces] = useState<Interface[]>([]);
  const [target, setTarget] = useState<'mock' | 'real'>('mock');
  const [nic, setNic] = useState('');
  const [speed, setSpeed] = useState(1);
  const [autoReturn, setAutoReturn] = useState(true);
  const [waist, setWaist] = useState(false);
  const [fast, setFast] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [sending, setSending] = useState<PoseSource | null>(null);      // a pose waiting for the real-robot confirmation
  const [safe, setSafe] = useState(false);
  const [error, setError] = useState('');
  const [working, setWorking] = useState(false);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try { const next = await call<Status>('status'); if (alive.current) setStatus(next); } catch { /* the server may be restarting */ }
  }, []);
  useEffect(() => {
    alive.current = true;
    void refresh();
    call<Interface[]>('interfaces').then(list => {
      if (!alive.current) return;
      setInterfaces(list);
      setNic(current => current || (list.find(item => item.robot) ?? list[0])?.name || '');
    }).catch(() => undefined);
    const timer = setInterval(() => void refresh(), 400);
    return () => { alive.current = false; clearInterval(timer); };
  }, [refresh]);

  async function act(task: () => Promise<unknown>) {
    setError(''); setWorking(true);
    try { await task(); } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setWorking(false); void refresh(); }
  }

  const player = status?.player;
  const running = !!player && ACTIVE.includes(player.state);
  const plan = status?.plan ?? null;
  const real = status?.mode === 'real';
  const blocked = startBlocked || (plan && plan.waist !== waist ? `허리 포함 설정이 점검 때(${plan.waist ? '켬' : '끔'})와 다릅니다. 다시 점검하세요.` : '');
  const stage = player && player.keyframe_times.length
    ? Math.max(0, player.keyframe_times.filter(start => start <= player.time + 1e-6).length - 1) : null;

  async function exportEpisode() {
    const response = await fetch('/api/export-lerobot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ project: getProject(), fps }) });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({})) as { detail?: unknown };
      throw new Error(typeof failure.detail === 'string' ? failure.detail : `내보내기 실패 (${response.status})`);
    }
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([await response.text()], { type: 'application/json' }));
    link.download = 'data.json';
    link.click(); URL.revokeObjectURL(link.href);
  }

  const panel = <div className={`replay-panel ${inline ? 'inline' : ''}`}>
      <header><strong>로봇 재생 · Unitree 기본 정책</strong>
        {onClose && <button aria-label="닫기" onClick={onClose} disabled={running} title={running ? '재생 중에는 닫을 수 없습니다' : '닫기'}>×</button>}</header>
      <p className="replay-note">로봇이 자체 모션 모드로 서 있는 상태에서 <b>팔 14축</b>(허리 포함 시 <b>허리 3축</b>도)을 <code>rt/arm_sdk</code>로 구동합니다. 다리·골반은 로봇이 정하고, 허리를 보내지 않을 때는 제어를 시작한 자세로 허리를 붙잡아 둡니다.
        {status?.read_only ? ' (읽기 전용 모드: 가상 로봇만 사용할 수 있습니다)' : ''}</p>
      {status?.connected && <RobotHealth status={status}/>}

      <section>
        <h4>1. 연결</h4>
        <div className="replay-row">
          <label><input type="radio" checked={target === 'mock'} disabled={running || status?.connected} onChange={() => setTarget('mock')}/> 가상 로봇</label>
          <label><input type="radio" checked={target === 'real'} disabled={running || status?.connected || status?.read_only} onChange={() => setTarget('real')}/> 실제 로봇</label>
          {target === 'real' && <select aria-label="네트워크 인터페이스" value={nic} disabled={running || status?.connected} onChange={e => setNic(e.target.value)}>
            {interfaces.map(item => <option key={item.name} value={item.name}>{item.name} · {item.ip}{item.robot ? ' · 로봇망' : ''}{item.wifi ? ' · Wi-Fi' : ''}</option>)}
            {!interfaces.length && <option value="">인터페이스 없음</option>}</select>}
          {!status?.connected
            ? <button className="primary" disabled={working || (target === 'real' && !nic)} onClick={() => void act(() => call('connect', { interface: target === 'real' ? nic : null, mock: target === 'mock' }))}>연결</button>
            : <button disabled={working || running} onClick={() => void act(() => call('disconnect', {}))}>연결 해제</button>}
          <span className={`replay-chip ${status?.connected ? 'ok' : ''}`}>{status?.connected ? `${real ? `실제 로봇 · ${status.interface}` : '가상 로봇'} 연결됨` : '연결 안 됨'}</span>
          {status?.robot && <span className="replay-chip" title="로봇 상태 수신 지연 / 몸통 기울기">{status.robot.age.toFixed(2)}s · 기울기 {status.robot.tilt_deg.toFixed(1)}°</span>}
        </div>
        {real && <p className="replay-warn">실제 로봇 연결은 읽기 전용입니다. 아래 확인을 거쳐 시작하기 전에는 명령을 보내지 않습니다.</p>}
      </section>

      <section>
        <h4>2. 모션 점검</h4>
        <div className="replay-row">
          <button disabled={working || (running && !SETTLED.includes(player.state))} title={running && !SETTLED.includes(player.state) ? '재생이나 이동이 끝난 뒤 점검할 수 있습니다' : '지금 장면으로 모션을 계산하고 점검합니다. 자세를 유지하는 중이면 그 자세에서 이어서 재생합니다.'} onClick={() => void act(async () => { await call('prepare', { project: getProject(), fps, waist }); onPrepared?.(); })}>현재 프로젝트 점검</button>
          <label title="모션의 허리(요·롤·피치)도 같은 rt/arm_sdk 메시지로 보냅니다(robot-studio의 허리 회전과 같은 방식). 바꾸면 다시 점검하세요."><input type="checkbox" checked={waist} disabled={working || running} onChange={e => setWaist(e.target.checked)}/> 허리 포함</label>
          <button disabled={working} onClick={() => void act(exportEpisode)} title="팔·허리를 LeRobot data.json으로 내보내 robot-studio 등에서 재생할 수 있습니다">data.json 내보내기</button>
          {plan && <span className="replay-chip">{plan.duration.toFixed(1)}초 · {plan.frames}프레임 · {plan.fps} Hz · {plan.waist ? '허리+팔 17축' : '팔 14축'}</span>}
        </div>
        {plan && <ul className="replay-items">{plan.items.map((item, index) => <li key={index} className={item.level}><b>{item.name}</b> {item.text}</li>)}</ul>}
      </section>

      <section>
        <h4>3. 실행</h4>
        <div className="replay-row">
          <label>속도 <select aria-label="재생 속도" value={speed} disabled={working} onChange={e => { const value = +e.target.value; setSpeed(value); if (running) void act(() => call(`speed?speed=${value}`, {})); }}>
            {[0.25, 0.5, 0.75, 1, 1.5, 2].map(value => <option key={value} value={value}>{value}×</option>)}</select></label>
          <label title="재생이 끝나면 시작했던 자세로 돌아간 뒤 제어권을 돌려줍니다"><input type="checkbox" checked={autoReturn} disabled={running} onChange={e => setAutoReturn(e.target.checked)}/> 끝나면 자동 복귀</label>
          {plan && speed > plan.speed_cap + 1e-9 && <label className="replay-warn" title="권장 속도 상한을 넘는 재생입니다"><input type="checkbox" checked={fast} disabled={running} onChange={e => setFast(e.target.checked)}/> 권장 속도 초과 허용</label>}
        </div>
        {(!running || player.state === 'holding') && <div className="replay-row">
          {!confirming
            ? <button className="primary" disabled={working || !status?.connected || !plan?.ok || !!blocked} onClick={() => real ? setConfirming(true) : void act(() => call('start', { speed, auto_return: autoReturn, allow_fast: fast, confirm: true }))}
                title={!status?.connected ? '먼저 연결하세요' : !plan ? '먼저 모션을 점검하세요' : !plan.ok ? '점검을 통과하지 못했습니다' : blocked}>시작</button>
            : <div className="replay-confirm">
                <p><b>실제 로봇이 움직입니다{plan?.waist ? ' (허리 포함)' : ''}.</b> {plan?.duration.toFixed(1)}초 분량을 {speed}× 로 재생하고{autoReturn ? ' 끝나면 시작 자세로 복귀합니다' : ' 끝난 자세를 유지합니다'}. 경고 {plan?.warnings ?? 0}건.</p>
                <label><input type="checkbox" checked={safe} onChange={e => setSafe(e.target.checked)}/> 로봇을 거치대나 지지 상태로 두었고, 팔 주변이 비어 있으며, 비상정지를 손이 닿는 곳에 두었습니다</label>
                <div className="replay-row">
                  <button className="primary" disabled={!safe || working} onClick={() => void act(async () => { await call('start', { speed, auto_return: autoReturn, allow_fast: fast, confirm: true }); setConfirming(false); setSafe(false); })}>실행</button>
                  <button onClick={() => { setConfirming(false); setSafe(false); }}>취소</button>
                </div>
              </div>}
        </div>}
        {running && <div className="replay-row">
          {player.state === 'paused'
            ? <button onClick={() => void act(() => call('pause?on=false', {}))}>이어서</button>
            : <button disabled={player.state !== 'playing'} onClick={() => void act(() => call('pause?on=true', {}))}>일시정지</button>}
          <button className="danger" onClick={() => void act(() => call('stop', {}))} title="그 자리에서 멈추고 마지막 목표를 계속 보냅니다">정지</button>
          <button disabled={![...SETTLED, 'paused'].includes(player.state)} onClick={() => void act(() => call('home', {}))} title="제어를 시작했던 자세로 돌아간 뒤 제어권을 돌려줍니다">복귀·해제</button>
        </div>}
        {player && <div className="replay-progress">
          <div className={`replay-bar ${player.state}`}><i style={{ width: `${player.duration ? Math.min(100, player.time / player.duration * 100) : 0}%` }}/></div>
          <span>{STATE_TEXT[player.state]}{player.phase !== null ? ` ${(player.phase * 100).toFixed(0)}%` : ''} · {player.time.toFixed(1)} / {player.duration.toFixed(1)}s{stage !== null && frameNames[stage] ? ` · ${frameNames[stage]}` : ''}</span>
        </div>}
        {blocked && !running && <p className="replay-warn">{blocked}</p>}
        {player?.fault && <p className="replay-error">{player.fault}</p>}
        {error && <p className="replay-error">{error}</p>}
      </section>

      {poses && <section>
        <h4>4. 자세 보내기</h4>
        <p className="replay-note">편집기의 자세 하나를 로봇에 보냅니다. 천천히(최대 0.6 rad/s) 이동한 뒤 그 자세를 계속 유지하고, 다음 자세나 모션 재생을 이어서 할 수 있습니다. {waist ? '허리 3축도 보냅니다.' : '팔 14축만 보냅니다(허리 포함을 켜면 허리도).'}</p>
        <div className="replay-row">
          {poses().map(source => <button key={source.label} disabled={working || !status?.connected || !source.qpos || (running && !SETTLED.includes(player.state))}
            title={source.reason ?? (running && !SETTLED.includes(player.state) ? '재생이나 이동이 끝난 뒤 보낼 수 있습니다' : '')}
            onClick={() => real && !SETTLED.includes(player?.state ?? 'idle') ? setSending(source) : void act(() => call('move', { qpos: source.qpos, waist, confirm: true }))}>{source.label}</button>)}
        </div>
        {sending && <div className="replay-confirm">
          <p><b>실제 로봇이 움직입니다.</b> “{sending.label}”로 {waist ? '허리와 ' : ''}팔을 천천히 옮긴 뒤 그 자세를 유지합니다.</p>
          <label><input type="checkbox" checked={safe} onChange={e => setSafe(e.target.checked)}/> 로봇을 거치대나 지지 상태로 두었고, 팔 주변이 비어 있으며, 비상정지를 손이 닿는 곳에 두었습니다</label>
          <div className="replay-row">
            <button className="primary" disabled={!safe || working} onClick={() => void act(async () => { await call('move', { qpos: sending.qpos, waist, confirm: true }); setSending(null); setSafe(false); })}>보내기</button>
            <button onClick={() => { setSending(null); setSafe(false); }}>취소</button>
          </div>
        </div>}
      </section>}
    </div>;
  return inline ? panel : <div className="replay-backdrop" role="dialog" aria-label="로봇 재생">{panel}</div>;
}

/** The robot connection while the robot panel is closed: a chip that opens it, and a stop button whenever the arms are being driven. */
export function RobotStatusChip({ onOpen }: { onOpen: () => void }) {
  const [status, setStatus] = useState<Status | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => call<Status>('status').then(next => { if (alive) setStatus(next); }).catch(() => undefined);
    void load();
    const timer = setInterval(() => void load(), 1000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  if (!status?.connected) return null;
  const running = ACTIVE.includes(status.player.state);
  return <span className="robot-chip">
    <button className={`replay-chip ok ${running ? 'running' : ''}`} onClick={onOpen} title="로봇 패널 열기">
      {status.mode === 'real' ? '실제 로봇' : '가상 로봇'} · {STATE_TEXT[status.player.state]}</button>
    {running && <button className="danger" onClick={() => void call('stop', {}).catch(() => undefined)} title="그 자리에서 멈추고 마지막 목표를 계속 보냅니다">정지</button>}
  </span>;
}

const JOINT_NAMES = ['왼 고관절 피치', '왼 고관절 롤', '왼 고관절 요', '왼 무릎', '왼 발목 피치', '왼 발목 롤',
  '오른 고관절 피치', '오른 고관절 롤', '오른 고관절 요', '오른 무릎', '오른 발목 피치', '오른 발목 롤', '허리 요', '허리 롤', '허리 피치',
  '왼 어깨 피치', '왼 어깨 롤', '왼 어깨 요', '왼 팔꿈치', '왼 손목 롤', '왼 손목 피치', '왼 손목 요',
  '오른 어깨 피치', '오른 어깨 롤', '오른 어깨 요', '오른 팔꿈치', '오른 손목 롤', '오른 손목 피치', '오른 손목 요'];
/** Body parts on the status figure (robot-studio's dashboard/core.py GROUPS): label, motor indices, x %, y %. The figure is a front
 * view, so the robot's left is on the right of the picture. */
const PARTS: [string, number[], number, number][] = [
  ['오른 어깨', [22, 23, 24], 33, 31], ['왼 어깨', [15, 16, 17], 67, 31], ['오른 팔꿈치', [25], 30, 45], ['왼 팔꿈치', [18], 70, 45],
  ['오른 손목', [26, 27, 28], 28, 55], ['왼 손목', [19, 20, 21], 72, 55], ['허리', [12, 13, 14], 50, 51],
  ['오른 고관절', [6, 7, 8], 37, 58], ['왼 고관절', [0, 1, 2], 63, 58], ['오른 무릎', [9], 37, 76], ['왼 무릎', [3], 63, 76],
  ['오른 발목', [10, 11], 37, 93], ['왼 발목', [4, 5], 63, 93]];

/** The robot at a glance (after robot-studio's status card): battery, the hottest joint, a figure with the temperature of each
 * body part, and a table of all 29 joints. */
function RobotHealth({ status }: { status: Status }) {
  const [details, setDetails] = useState(false);
  const robot = status.robot;
  const limits = status.temperature_limits ?? { warning: 55, critical: 70 };
  const level = (t: number | null | undefined) => t === null || t === undefined ? 'unknown' : t >= limits.critical ? 'hot' : t >= limits.warning ? 'warm' : 'cool';
  const temp = robot?.temp ?? null;
  const hottest = temp ? temp.reduce((best, value, index) => value > temp[best] ? index : best, 0) : null;
  const battery = robot?.battery ?? null;
  const estimate = status.battery_estimate ?? null;
  const duration = (seconds: number) => seconds >= 3600 ? `${Math.floor(seconds / 3600)}시간 ${Math.floor(seconds % 3600 / 60)}분` : `${Math.max(1, Math.round(seconds / 60))}분`;
  return <section className="robot-health">
    <h4>로봇 상태 <small>{robot ? `${status.mode === 'real' ? 'LIVE' : '가상'} · 지연 ${robot.age.toFixed(2)}s · 기울기 ${robot.tilt_deg.toFixed(1)}°` : '수신 대기'}</small></h4>
    <div className="robot-health-body">
      <div className="robot-figure">
        <img src="/robot-status/g1-status-cutout.png" alt="G1 정면 · 부위별 최고 관절 온도"/>
        {PARTS.map(([label, ids, x, y]) => {
          const value = temp ? Math.max(...ids.map(id => temp[id])) : null;
          return <i key={label} className={`heat ${level(value)}`} style={{ left: `${x}%`, top: `${y}%` }} title={`${label} ${value === null ? '미수신' : `${value.toFixed(0)}°C`}`}/>;
        })}
      </div>
      <div className="robot-metrics">
        <div><span>배터리</span><strong>{battery ? battery.soc.toFixed(0) : '—'}<small>%</small>{battery && <em className="battery-left" title={estimate
            ? `최근 ${Math.round(estimate.measured_s / 60)}분 동안 SOC가 줄어든 속도(시간당 ${estimate.percent_per_hour.toFixed(1)}%)로 0%까지 남은 시간을 추정합니다. 부하가 바뀌면 달라집니다.`
            : 'SOC가 두 번 줄어들 때까지 감소 속도를 재고 있습니다(1% 단위라 몇 분 걸립니다).'}>{estimate ? `약 ${duration(estimate.remaining_s)} 남음` : '남은 시간 측정 중'}</em>}</strong>
          <div className="battery-track"><i className={battery && battery.soc < 20 ? 'low' : ''} style={{ width: `${battery ? Math.max(0, Math.min(100, battery.soc)) : 0}%` }}/></div>
          <small>{battery ? `${battery.voltage.toFixed(1)} V · ${battery.current.toFixed(1)} A${battery.temp !== null ? ` · ${battery.temp.toFixed(0)}°C` : ''}` : 'rt/lf/bmsstate 미수신'}</small></div>
        <div><span>최고 관절 온도</span><strong className={level(hottest !== null && temp ? temp[hottest] : null)}>{hottest !== null && temp ? temp[hottest].toFixed(0) : '—'}<small>°C</small></strong>
          <small>{hottest !== null ? JOINT_NAMES[hottest] : '미수신'}</small></div>
        <div className="heat-legend"><span><i className="cool"/>{`<${limits.warning}`}</span><span><i className="warm"/>주의</span><span><i className="hot"/>{`≥${limits.critical}`}</span><span><i className="unknown"/>미수신</span></div>
        <button className="robot-details-toggle" onClick={() => setDetails(open => !open)}>{details ? '관절 상세 닫기' : '29개 관절 상세'}</button>
      </div>
    </div>
    {details && <table className="robot-joints"><thead><tr><th>관절</th><th>각도</th><th>온도</th><th>토크</th></tr></thead>
      <tbody>{JOINT_NAMES.map((name, index) => <tr key={name}><th>{index} · {name}</th>
        <td>{robot?.q ? `${(robot.q[index] * 180 / Math.PI).toFixed(1)}°` : '—'}</td>
        <td className={level(temp?.[index])}>{temp ? `${temp[index].toFixed(0)}°C` : '—'}</td>
        <td>{robot?.tau ? `${robot.tau[index].toFixed(1)}` : '—'}</td></tr>)}</tbody></table>}
  </section>;
}
