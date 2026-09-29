import NumericInput from './numeric-input';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RobotScene, type PoseState } from '../lib/robot-scene';

type Pose = { position: number[]; quaternion_xyzw: number[] };
type BoxSpec = { pose: Pose; size: number[] };
type Spec = {
  format: string;
  name: string;
  actual: BoxSpec;
  perceived: BoxSpec;
  destination: Pose;
  mass_kg: number;
  hand_friction: number;
  floor_friction: number;
  pallet_friction: number;
  pallet_height: number;
  squeeze_m: number;
  pregrasp_clearance_m: number;
  standoff_m: number;
  carry_height_m: number;
  phase_seconds: number;
  timeout_seconds: number;
  seed: number;
  diffusion_steps: number;
};
type Phase = {
  id: string;
  label: string;
  start: number;
  end: number;
  surface_targets: Record<string, number[]>;
  root_position: number[];
  guard: string;
};
type Plan = {
  spec: Spec;
  phases: Phase[];
  start_qpos: number[];
  target_surface_gap_m: number;
  task_sha256: string;
};
type Run = {
  id: string;
  task_id: string;
  kind: string;
  status: string;
  progress: number;
  message: string;
  files: string[];
  result?: {
    status?: string;
    reason?: string;
    validated?: boolean;
    wall_seconds?: number;
    sim_seconds?: number;
    reference?: { max_joint_limit_excess_rad: number };
    constraint_errors?: {
      phase: string;
      left_surface_error_m: number;
      right_surface_error_m: number;
    }[];
  };
};
type Hands = Record<
  string,
  {
    normal_n: number;
    tangent_n: number;
    slip_m_s: number;
    force_world_n: number[];
    wrist_contact_torque_world_nm: number[];
  }
>;
type Frame = {
  time: number;
  state: PoseState;
  box_position?: number[];
  box_quaternion_xyzw?: number[];
  metrics?: {
    phase: string;
    hands: Hands;
    torque_saturation_fraction: number;
    root_error_m: number;
    penetration_m: number;
  };
};
type Replay = { frames: Frame[]; plan: Plan; physics: boolean };
async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(
    '/api/tasks' + path,
    body === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
  );
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { detail?: unknown };
    throw new Error(
      typeof e.detail === 'string'
        ? e.detail
        : JSON.stringify(e.detail ?? r.status),
    );
  }
  return r.json();
}
function Vec({
  label,
  value,
  onChange,
  step = 0.01,
}: {
  label: string;
  value: number[];
  onChange: (v: number[]) => void;
  step?: number;
}) {
  return (
    <fieldset className="task-vector">
      <legend>{label}</legend>
      {value.map((v, i) => (
        <label key={i}>
          <span>{value.length === 4 ? 'XYZW'[i] : 'XYZ'[i]}</span>
          <NumericInput
            step={step}
            value={v}
            aria-label={`${label} ${'XYZW'[i]}`}
            onChange={(e) =>
              onChange(
                value.map((x, j) => (j === i ? Number(e.target.value) : x)),
              )
            }
          />
        </label>
      ))}
    </fieldset>
  );
}
function PoseFields({
  label,
  value,
  onChange,
}: {
  label: string;
  value: Pose;
  onChange: (v: Pose) => void;
}) {
  const yaw = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion().fromArray(value.quaternion_xyzw),
    'ZYX',
  );
  return (
    <>
      <Vec
        label={label + ' 위치 (m)'}
        value={value.position}
        onChange={(position) => onChange({ ...value, position })}
      />
      <label className="task-number">
        {label} 회전 Z (°)
        <NumericInput
          step="1"
          value={Math.round(THREE.MathUtils.radToDeg(yaw.z) * 100) / 100}
          onChange={(e) =>
            onChange({
              ...value,
              quaternion_xyzw: new THREE.Quaternion()
                .setFromEuler(
                  new THREE.Euler(
                    yaw.x,
                    yaw.y,
                    THREE.MathUtils.degToRad(+e.target.value),
                    'ZYX',
                  ),
                )
                .toArray(),
            })
          }
        />
      </label>
      <details>
        <summary>6D pose quaternion</summary>
        <Vec
          label={label + ' xyzw'}
          value={value.quaternion_xyzw}
          step={0.001}
          onChange={(quaternion_xyzw) =>
            onChange({ ...value, quaternion_xyzw })
          }
        />
      </details>
    </>
  );
}
const noop = () => {};
export default function TaskWorkbench({
  initialQ,
  onClose,
}: {
  initialQ: number[];
  onClose: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<RobotScene | null>(null);
  const extras = useRef<THREE.Group | null>(null);
  const [spec, setSpec] = useState<Spec | null>(null),
    [tid, setTid] = useState(''),
    [tasks, setTasks] = useState<{ id: string; name: string }[]>([]);
  const [startQ, setStartQ] = useState(initialQ),
    [plan, setPlan] = useState<Plan | null>(null),
    [runs, setRuns] = useState<Run[]>([]),
    [job, setJob] = useState<Run | null>(null);
  const [runtime, setRuntime] = useState<{
    installed_assets: boolean;
    missing: string[];
    reference_files: string[];
  } | null>(null);
  const [selectedRef, setSelectedRef] = useState(''),
    [controller, setController] = useState('sonic');
  const [error, setError] = useState(''),
    [busy, setBusy] = useState(false),
    [replay, setReplay] = useState<Replay | null>(null),
    [sample, setSample] = useState(0),
    [playing, setPlaying] = useState(false),
    [phase, setPhase] = useState(0);
  const active = job?.status === 'starting' || job?.status === 'running';
  const cancelledPoll = useRef(false);
  async function load(t: string) {
    const r = await api<{ id: string; plan: Plan }>('/' + t);
    setTid(t);
    setSpec(r.plan.spec);
    setStartQ(r.plan.start_qpos);
    setPlan(r.plan);
    const rr = await api<Run[]>('/' + t + '/runs');
    setRuns(rr);
    setJob(rr[0] ?? null);
    setReplay(null);
    setPlaying(false);
    setSelectedRef(
      rr.find(
        (x) =>
          x.kind !== 'simulate' &&
          x.status === 'completed' &&
          x.files.includes('reference.npz'),
      )?.id ?? '',
    );
    localStorage.setItem('g1-last-task', t);
    if (rr[0]?.status === 'completed' && rr[0].files.includes('reference.npz'))
      await loadReplay(rr[0]);
  }
  useEffect(() => {
    let mounted = true;
    let v: RobotScene | null = null;
    void (async () => {
      try {
        const [model, defaults, rt, list] = await Promise.all([
          fetch('/api/init').then(async response => {
            if (!response.ok) throw new Error(`모델 정보 요청 실패 (${response.status})`);
            return response.json() as Promise<{ model_id: string; visual_revision: string }>;
          }),
          api<Spec>('/defaults'),
          api<NonNullable<typeof runtime>>('/runtime'),
          api<typeof tasks>(''),
        ]);
        if (!mounted) return;
        v = new RobotScene(host.current!, {
          select: noop,
          begin: noop,
          move: noop,
          rotate: noop,
          jointAngle: noop,
          transformMode: noop,
          end: noop,
          error: setError,
        }, model.model_id, model.visual_revision);
        viewer.current = v;
        v.setEditable(false);
        v.showHandles(false);
        v.orbit.target.set(0.65, 0, 0.65);
        v.camera.position.set(2.8, -3.2, 2.1);
        v.orbit.update();
        extras.current = new THREE.Group();
        v.scene.add(extras.current);
        setRuntime(rt);
        setTasks(list);
        const last =
          new URLSearchParams(window.location.search).get('task') ||
          localStorage.getItem('g1-last-task');
        if (last && list.some((t) => t.id === last)) await load(last);
        else setSpec(defaults);
      } catch (e) {
        if (mounted) setError((e as Error).message);
      }
    })();
    return () => {
      mounted = false;
      v?.dispose();
      viewer.current = null;
    };
  }, []);
  useEffect(() => {
    if (replay) return;
    let current = true;
    void fetch('/api/pose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ qpos: startQ }),
    })
      .then(async (r) => {
        if (r.ok) {
          const pose = (await r.json()) as PoseState;
          if (current) viewer.current?.update(pose);
        }
      })
      .catch((e) => {
        if (current) setError(e.message);
      });
    return () => {
      current = false;
    };
  }, [startQ, !!replay]);
  useEffect(() => {
    if (!spec) return;
    let mounted = true;
    const timer = setTimeout(() => {
      void api<Plan>('/plan', { spec, start_qpos: startQ })
        .then((p) => {
          if (mounted) {
            setPlan(p);
            setError('');
          }
        })
        .catch((e) => {
          if (mounted) {
            setError(e.message);
            setPlan(null);
          }
        });
    }, 350);
    return () => {
      mounted = false;
      clearTimeout(timer);
    };
  }, [spec, startQ]);
  useEffect(() => {
    if (!spec || !viewer.current || !extras.current) return;
    const v = viewer.current,
      g = extras.current;
    while (g.children.length) {
      const child = g.children[0];
      g.remove(child);
      child.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.Line) {
          o.geometry.dispose();
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) =>
            m.dispose(),
          );
        }
      });
    }
    const shown = replay?.plan.spec ?? spec;
    const frame = replay?.frames[sample];
    v.setBox(
      frame?.box_position ?? shown.actual.pose.position,
      shown.actual.size,
      true,
    );
    v.box.quaternion.fromArray(
      frame?.box_quaternion_xyzw ?? shown.actual.pose.quaternion_xyzw,
    );
    const perceived = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(
          ...(shown.perceived.size as [number, number, number]),
        ),
      ),
      new THREE.LineBasicMaterial({ color: '#6fcef9' }),
    );
    perceived.position.fromArray(shown.perceived.pose.position);
    perceived.quaternion.fromArray(shown.perceived.pose.quaternion_xyzw);
    g.add(perceived);
    const pallet = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.8, shown.pallet_height),
      new THREE.MeshStandardMaterial({ color: '#594b38' }),
    );
    pallet.position.set(
      shown.destination.position[0],
      shown.destination.position[1],
      shown.pallet_height / 2,
    );
    const pr = new THREE.Euler().setFromQuaternion(
      new THREE.Quaternion().fromArray(shown.destination.quaternion_xyzw),
      'ZYX',
    );
    pallet.rotation.z = pr.z;
    g.add(pallet);
    const shownPhases = (replay?.plan ?? plan)?.phases;
    const p = frame?.metrics
      ? shownPhases?.find((p) => p.id === frame.metrics!.phase)
      : frame
        ? shownPhases?.find((p) => p.end >= frame.time)
        : shownPhases?.[phase];
    if (p)
      for (const [side, pos] of Object.entries(p.surface_targets)) {
        const target = new THREE.Mesh(
          new THREE.SphereGeometry(0.022, 12, 8),
          new THREE.MeshBasicMaterial({
            color: side === 'left' ? '#8be6be' : '#f3b979',
            wireframe: true,
          }),
        );
        target.position.fromArray(pos);
        g.add(target);
      }
    if (frame) {
      v.update(frame.state);
      for (const [side, h] of Object.entries(frame.metrics?.hands ?? {})) {
        const f = new THREE.Vector3().fromArray(h.force_world_n);
        if (f.length() > 0.05) {
          const arrow = new THREE.ArrowHelper(
            f.clone().normalize(),
            new THREE.Vector3().fromArray(
              frame.state.handles[side + '_hand'].position,
            ),
            Math.min(0.6, f.length() * 0.005),
            side === 'left' ? 0x8be6be : 0xf3b979,
            0.04,
            0.025,
          );
          g.add(arrow);
        }
      }
    }
    v.dirty = true;
  }, [spec, plan, phase, replay, sample]);
  async function loadReplay(run: Run) {
    const data = await api<Replay>(`/${run.task_id}/runs/${run.id}/replay`);
    setPlaying(false);
    setSample(0);
    setReplay(data);
  }
  useEffect(() => {
    if (!job || !active) return;
    cancelledPoll.current = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api<Run>(`/${job!.task_id}/runs/${job!.id}`);
        if (cancelledPoll.current) return;
        setJob(next);
        if (next.status === 'running' || next.status === 'starting')
          timer = setTimeout(poll, 1200);
        else {
          setRuns(await api<Run[]>(`/${next.task_id}/runs`));
          if (
            next.status === 'completed' &&
            next.files.includes('reference.npz')
          ) {
            if (next.kind !== 'simulate') setSelectedRef(next.id);
            await loadReplay(next);
          }
        }
      } catch (e) {
        if (!cancelledPoll.current) setError((e as Error).message);
      }
    }
    timer = setTimeout(poll, 1000);
    return () => {
      cancelledPoll.current = true;
      clearTimeout(timer);
    };
  }, [job?.id, active]);
  useEffect(() => {
    if (!playing || !replay?.frames.length) return;
    let id = 0;
    const start = performance.now(),
      base = replay.frames[sample].time;
    function tick() {
      const t = base + (performance.now() - start) / 1000;
      let i = replay!.frames.findIndex((f) => f.time > t);
      if (i < 0) {
        setSample(replay!.frames.length - 1);
        setPlaying(false);
        return;
      }
      setSample(Math.max(0, i - 1));
      id = requestAnimationFrame(tick);
    }
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, replay]);
  async function runAction(fn: () => Promise<void>) {
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    if (!spec) throw Error('작업 설정이 없습니다.');
    const r = await api<{ id: string; plan: Plan }>('', {
      spec,
      start_qpos: startQ,
      ...(tid ? { id: tid } : {}),
    });
    setTid(r.id);
    setPlan(r.plan);
    localStorage.setItem('g1-last-task', r.id);
    setTasks(await api<typeof tasks>(''));
    return r;
  }
  function launch(kind: string) {
    void runAction(async () => {
      setPlaying(false);
      const task = await save();
      const body: Record<string, string> = { kind, controller };
      if (kind === 'simulate') {
        if (selectedRef.startsWith('file:'))
          body.reference_file = selectedRef.slice(5);
        else body.reference_run = selectedRef;
      }
      const next = await api<Run>(`/${task.id}/runs`, body);
      setJob(next);
      setReplay(null);
      setRuns(await api<Run[]>(`/${task.id}/runs`));
    });
  }
  function number(
    key: keyof Spec,
    label: string,
    step = 0.01,
    min = 0,
    max = 100,
  ) {
    return (
      spec && (
        <label className="task-number">
          {label}
          <NumericInput
            min={min}
            max={max}
            step={step}
            value={spec[key] as number}
            onChange={(e) => setSpec({ ...spec, [key]: +e.target.value })}
          />
        </label>
      )
    );
  }
  const frame = replay?.frames[sample];
  const stale =
    !!replay && !!plan && replay.plan.task_sha256 !== plan.task_sha256;
  return (
    <div
      className="task-workbench"
      role="dialog"
      aria-modal="true"
      aria-label="상자 태스크 실험실"
    >
      <header>
        <div>
          <strong>상자 태스크 실험실</strong>
          <span>G1 29 DoF · CPU · 접촉 기반 파지</span>
        </div>
        <button onClick={onClose}>편집기로 돌아가기</button>
      </header>
      <aside className="task-settings">
        <label>
          저장된 태스크
          <select
            value={tid}
            disabled={active || busy}
            onChange={(e) => {
              if (e.target.value) void runAction(() => load(e.target.value));
              else {
                setTid('');
                setRuns([]);
                setJob(null);
                setReplay(null);
                setSelectedRef('');
              }
            }}
          >
            <option value="">새 태스크</option>
            {tasks.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        {spec && (
          <>
            <label>
              태스크 이름
              <input
                value={spec.name}
                onChange={(e) => setSpec({ ...spec, name: e.target.value })}
              />
            </label>
            <h3>실제 상자 · 물리 환경</h3>
            <PoseFields
              label="실제"
              value={spec.actual.pose}
              onChange={(pose) =>
                setSpec({ ...spec, actual: { ...spec.actual, pose } })
              }
            />
            <Vec
              label="실제 크기 (m) · Y가 파지 폭"
              value={spec.actual.size}
              onChange={(size) =>
                setSpec({ ...spec, actual: { ...spec.actual, size } })
              }
            />
            {number('mass_kg', '상자 질량 (kg)', 0.1, 0.1, 20)}
            {number('hand_friction', '손–상자 마찰계수', 0.05, 0, 2)}
            <h3>인식 결과 · 모션 생성 입력</h3>
            <button
              onClick={() =>
                setSpec({ ...spec, perceived: structuredClone(spec.actual) })
              }
            >
              실제 상자 값 복사
            </button>
            <PoseFields
              label="인식"
              value={spec.perceived.pose}
              onChange={(pose) =>
                setSpec({ ...spec, perceived: { ...spec.perceived, pose } })
              }
            />
            <Vec
              label="인식 크기 (m)"
              value={spec.perceived.size}
              onChange={(size) =>
                setSpec({ ...spec, perceived: { ...spec.perceived, size } })
              }
            />
            {number('squeeze_m', '추가 압착 간격 (m)', 0.002, 0, 0.06)}
            <p className="task-note">
              목표 접촉면 간격:{' '}
              <b>
                {plan ? (plan.target_surface_gap_m * 100).toFixed(1) : '—'} cm
              </b>
              <br />
              파지력은 실제 접촉·마찰·관절 토크로 결정됩니다.
            </p>
            <h3>내려놓을 위치</h3>
            <PoseFields
              label="목표 상자 중심"
              value={spec.destination}
              onChange={(destination) => setSpec({ ...spec, destination })}
            />
            {number('pallet_height', '팔레트 윗면 높이 (m)', 0.01, 0.01, 0.5)}
            <details>
              <summary>생성·환경 설정</summary>
              {number('floor_friction', '바닥 마찰계수', 0.05, 0, 2)}
              {number('pallet_friction', '팔레트 마찰계수', 0.05, 0, 2)}
              {number('standoff_m', '접근 거리 (m)', 0.01, 0.25, 0.8)}
              {number(
                'pregrasp_clearance_m',
                '손 접근 여유 (편측 m)',
                0.01,
                0.01,
                0.25,
              )}
              {number(
                'carry_height_m',
                '운반 상자 중심 높이 (m)',
                0.01,
                0.35,
                0.95,
              )}
              {number('phase_seconds', '단계별 참조 시간 (s)', 0.2, 0.4, 6)}
              {number('timeout_seconds', '조건 대기 제한 (s)', 0.5, 0.5, 10)}
              {number('diffusion_steps', 'ARDY 확산 단계', 1, 1, 10)}
              {number('seed', '생성 시드', 1, 0, 2147483647)}
              <button onClick={() => setStartQ(initialQ)}>
                편집기의 현재 자세를 시작점으로
              </button>
            </details>
            <button
              disabled={busy || !plan}
              onClick={() =>
                void runAction(async () => {
                  await save();
                })
              }
            >
              태스크 프리셋 저장
            </button>
          </>
        )}
      </aside>
      <main className="task-main">
        <div className="task-toolbar">
          <button
            disabled={busy || active || !plan || !runtime?.installed_assets}
            className="primary"
            onClick={() => launch('ardy')}
          >
            ARDY 모션 생성
          </button>
          <button
            disabled={busy || active || !plan}
            onClick={() => launch('ik_preview')}
          >
            IK 배치 미리보기
          </button>
          <span>
            {runtime?.installed_assets
              ? 'CPU 모델 설치됨'
              : `실행 준비 필요: ${runtime?.missing.join(', ') ?? '확인 중'}`}
          </span>
        </div>
        <div className="task-view" ref={host} />
        <div className="task-legend">
          <span>■ 실제 상자</span>
          <span>□ 인식된 상자</span>
          <span>○ 접촉면 목표</span>
          <span>→ 상자가 손에 가하는 힘 · 1 N = 5 mm, 최대 0.6 m</span>
        </div>
        <div className="task-phases">
          {(replay?.plan ?? plan)?.phases.map((p, i) => (
            <button
              key={p.id}
              className={
                frame?.metrics?.phase === p.id ||
                (!frame?.metrics && phase === i)
                  ? 'active'
                  : ''
              }
              onClick={() => {
                setPhase(i);
                if (replay) {
                  const index = replay.frames.findIndex((f) =>
                    f.metrics ? f.metrics.phase === p.id : f.time >= p.end,
                  );
                  if (index >= 0) {
                    setSample(index);
                    setPlaying(false);
                  }
                }
              }}
            >
              {i + 1}. {p.label}
            </button>
          ))}
        </div>
        <div className="task-playback">
          <button
            disabled={!replay?.frames.length}
            onClick={() => setPlaying(!playing)}
          >
            {playing ? '일시정지' : '재생'}
          </button>
          <input
            aria-label="태스크 모션 시간"
            type="range"
            min="0"
            max={Math.max(0, (replay?.frames.length ?? 1) - 1)}
            value={sample}
            onChange={(e) => {
              setPlaying(false);
              setSample(+e.target.value);
            }}
          />
          <span>{frame?.time.toFixed(2) ?? '0.00'} s</span>
          <b>
            {replay
              ? replay.physics
                ? 'MuJoCo 실제 실행'
                : '참조 모션 · 물리 미검증'
              : '상자 위치·목표 확인'}
          </b>
        </div>
        {stale && (
          <p className="task-warning">
            현재 설정과 다른 실행 기록을 보고 있습니다. 변경된 설정으로 다시
            생성·검증하세요.
          </p>
        )}
        <section className="task-results">
          <div className="task-sim-controls">
            <select
              aria-label="시뮬레이션 참조 모션"
              value={selectedRef}
              onChange={(e) => setSelectedRef(e.target.value)}
            >
              <option value="">검증할 참조 모션 선택</option>
              {runs
                .filter(
                  (r) =>
                    r.kind !== 'simulate' &&
                    r.status === 'completed' &&
                    r.files.includes('reference.npz'),
                )
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.kind === 'ardy' ? 'ARDY' : 'IK 배치'} · {r.id}
                  </option>
                ))}
              {runtime?.reference_files.map((f) => (
                <option key={f} value={'file:' + f}>
                  기존 NPZ · {f}
                </option>
              ))}
            </select>
            <select
              aria-label="전신 제어기"
              value={controller}
              onChange={(e) => setController(e.target.value)}
            >
              <option value="sonic">GEAR-SONIC / CPU</option>
              <option value="pd_diagnostic">PD 진단 (WBC 검증 아님)</option>
            </select>
            <button
              className="primary"
              disabled={busy || active || !selectedRef || !plan}
              onClick={() => launch('simulate')}
            >
              MuJoCo 검증 실행
            </button>
          </div>
          {job && (
            <div className="task-job">
              <strong>
                {job.kind} · {job.status}
              </strong>
              <span>{job.message}</span>
              {active && (
                <>
                  <progress max="1" value={job.progress} />
                  <button
                    onClick={() =>
                      void runAction(async () => {
                        await api(`/${job.task_id}/runs/${job.id}/cancel`, {});
                      })
                    }
                  >
                    작업 취소
                  </button>
                </>
              )}
              {job.result?.status && (
                <b
                  className={
                    job.result.validated ? 'task-pass' : 'task-warning'
                  }
                >
                  {job.result.validated
                    ? '태스크 성공'
                    : `${job.result.status} · ${job.result.reason || '물리 성공으로 검증되지 않음'}`}
                </b>
              )}
              {job.result?.wall_seconds !== undefined && (
                <span>
                  물리 {job.result.sim_seconds?.toFixed(2)} s / 계산{' '}
                  {job.result.wall_seconds.toFixed(2)} s
                </span>
              )}
              {job.result?.reference && (
                <span>
                  관절 범위 최대 초과:{' '}
                  {(
                    (job.result.reference.max_joint_limit_excess_rad * 180) /
                    Math.PI
                  ).toFixed(2)}
                  °
                </span>
              )}
            </div>
          )}
          {job?.result?.constraint_errors && (
            <details>
              <summary>생성된 모션의 접촉면 목표 오차</summary>
              <table>
                <thead>
                  <tr>
                    <th>단계</th>
                    <th>왼손 (mm)</th>
                    <th>오른손 (mm)</th>
                  </tr>
                </thead>
                <tbody>
                  {job.result.constraint_errors
                    .filter((e) => !['approach', 'retreat'].includes(e.phase))
                    .map((e) => (
                      <tr key={e.phase}>
                        <td>
                          {plan?.phases.find((p) => p.id === e.phase)?.label ??
                            e.phase}
                        </td>
                        <td>{(e.left_surface_error_m * 1000).toFixed(1)}</td>
                        <td>{(e.right_surface_error_m * 1000).toFixed(1)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </details>
          )}
          {frame?.metrics ? (
            <>
              <table>
                <thead>
                  <tr>
                    <th>페이크핸드</th>
                    <th>압착력 N</th>
                    <th>접선력 N</th>
                    <th>미끄럼 m/s</th>
                    <th>손목 접촉 모멘트 N·m</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(frame.metrics.hands).map(([side, h]) => (
                    <tr key={side}>
                      <td>{side === 'left' ? '왼손' : '오른손'}</td>
                      <td>{h.normal_n.toFixed(2)}</td>
                      <td>{h.tangent_n.toFixed(2)}</td>
                      <td>{h.slip_m_s.toFixed(3)}</td>
                      <td>
                        {h.wrist_contact_torque_world_nm
                          .map((x) => x.toFixed(2))
                          .join(' / ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="task-note">
                토크 포화 관절{' '}
                {(100 * frame.metrics.torque_saturation_fraction).toFixed(0)}% ·
                루트 XY 오차 {frame.metrics.root_error_m.toFixed(3)} m · 최대
                상자 접촉 침투 {(1000 * frame.metrics.penetration_m).toFixed(1)}{' '}
                mm
              </p>
            </>
          ) : (
            <p className="task-note">
              검증 실행 후 양손의 접촉력·미끄러짐·손목 모멘트를 확인할 수
              있습니다. 상자를 손에 구속하지 않습니다.
            </p>
          )}
          <details>
            <summary>저장된 실행 기록 · NPZ / 보고서</summary>
            {runs.map((r) => (
              <div className="task-run" key={r.id}>
                <button
                  disabled={
                    busy || active || !r.files.includes('reference.npz')
                  }
                  onClick={() =>
                    void runAction(async () => {
                      setJob(r);
                      await loadReplay(r);
                    })
                  }
                >
                  {r.kind} · {r.id} · {r.status}
                </button>
                {r.files
                  .filter((f) =>
                    [
                      'reference.npz',
                      'report.json',
                      'contact-log.json',
                      'simulation.npz',
                      'request.json',
                      'scene.xml',
                      'constraint-errors.json',
                      'worker.log',
                    ].includes(f),
                  )
                  .map((f) => (
                    <a
                      key={f}
                      href={`/api/tasks/${r.task_id}/runs/${r.id}/files/${f}`}
                      download
                    >
                      {f}
                    </a>
                  ))}
              </div>
            ))}
          </details>
        </section>
      </main>
      <footer className={error ? 'task-warning' : ''} role="status">
        {error ||
          '작업 단계는 측정 조건을 만족해야 전환됩니다. 참조 모션·설정·실행 결과는 각각 저장됩니다.'}
      </footer>
    </div>
  );
}
