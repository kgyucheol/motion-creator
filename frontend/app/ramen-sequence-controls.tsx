import { useState } from 'react';
import { Crosshair, Play, Route, ShieldCheck } from 'lucide-react';
import type { Keyframe } from '../lib/keyframes';
import type { SceneObject } from '../lib/scene-objects';

export type RamenSequenceSettings = {
  approach_clearance_m: number;
  lift_height_m: number;
  extraction_distance_m: number;
  extraction_cycles: number;
  carry_tilt_deg: number;
  phase_seconds: number;
  insertion_seconds: number;
  hold_seconds: number;
  force_limit_n: number;
  lateral_stiffness_n_per_m: number;
  insertion_stiffness_n_per_m: number;
  translation_damping_ns_per_m: number;
  orientation_stiffness_nm_per_rad: number;
  orientation_damping_nms_per_rad: number;
  maximum_feedback_torque_fraction: number;
};

const defaults: RamenSequenceSettings = {
  approach_clearance_m: .08,
  lift_height_m: .18,
  extraction_distance_m: .20,
  extraction_cycles: 3,
  carry_tilt_deg: 18,
  phase_seconds: 2,
  insertion_seconds: 3,
  hold_seconds: 1,
  force_limit_n: 35,
  lateral_stiffness_n_per_m: 420,
  insertion_stiffness_n_per_m: 90,
  translation_damping_ns_per_m: 28,
  orientation_stiffness_nm_per_rad: 38,
  orientation_damping_nms_per_rad: 4.5,
  maximum_feedback_torque_fraction: .35,
};

type Props = {
  objects: SceneObject[];
  selectedObjectId: string | null;
  keyframes: Keyframe[];
  disabled: boolean;
  onTargetChange: (id: string) => void;
  onGenerate: (objectId: string, settings: RamenSequenceSettings) => void;
  onSelectFrame: (index: number) => void;
  onRun: (index: number) => void;
};

export default function RamenSequenceControls(props: Props) {
  const [settings, setSettings] = useState<RamenSequenceSettings>(() => structuredClone(defaults));
  const candidates = props.objects.filter(object => object.visible && !object.fixed
    && (object.shape === 'cylinder' || object.collision_shape === 'cylinder'));
  const targetId = candidates.some(object => object.id === props.selectedObjectId)
    ? props.selectedObjectId! : candidates[0]?.id ?? '';
  const target = candidates.find(object => object.id === targetId);
  const targetSize = target?.collision_shape === 'cylinder' && target.collision_size?.length === 3
    ? target.collision_size : target?.size;
  const targetDiameter = targetSize ? Math.max(targetSize[0], targetSize[1]) : 0;
  const sequence = props.keyframes.map((frame, index) => ({ frame, index }))
    .filter(value => value.frame.interaction?.task === 'ramen_extract');
  const updateNumber = (key: keyof RamenSequenceSettings, value: number) => {
    if (!Number.isFinite(value)) return;
    setSettings(current => ({ ...current, [key]: value }));
  };
  return <section className="ramen-sequence-controls">
    <div className="panel-heading"><span>라면 꺼내기</span><small>물체 상대 TCP</small></div>
    <p className="hint">대상 중심·충돌 지름·회전으로 양쪽 파지점과 상자 기울기를 계산합니다. 현재 TCP는 각 도구의 장착 방향을 판별하는 기준으로만 사용하며, 생성 경로는 G1 기본자세에서 시작합니다.</p>
    <div className="inspector-label">대상 라면 묶음</div>
    <select aria-label="라면 꺼내기 대상" value={targetId} disabled={props.disabled || !candidates.length} onChange={event => props.onTargetChange(event.target.value)}>
      {!candidates.length && <option value="">움직일 수 있는 오브젝트 없음</option>}
      {candidates.map(object => <option key={object.id} value={object.id}>{object.name}</option>)}
    </select>
    {target && <p className="ramen-object-reference">물체 중심 <b>X {target.position[0].toFixed(3)} · Y {target.position[1].toFixed(3)} · Z {target.position[2].toFixed(3)} m</b><br/>충돌 지름 <b>{(targetDiameter * 1000).toFixed(0)} mm</b> · 현재 회전을 상자 기울기로 사용</p>}
    <div className="ramen-sequence-grid">
      <label>삽입 시작 높이 <span><input type="number" min="10" max="300" defaultValue={settings.approach_clearance_m * 1000} disabled={props.disabled} onChange={event => updateNumber('approach_clearance_m', +event.target.value / 1000)}/> mm</span></label>
      <label>총 인양 높이 <span><input type="number" min="20" max="600" defaultValue={settings.lift_height_m * 1000} disabled={props.disabled} onChange={event => updateNumber('lift_height_m', +event.target.value / 1000)}/> mm</span></label>
      <label>몸쪽 꺼내기 <span><input type="number" min="20" max="800" defaultValue={settings.extraction_distance_m * 1000} disabled={props.disabled} onChange={event => updateNumber('extraction_distance_m', +event.target.value / 1000)}/> mm</span></label>
      <label>반복 횟수 <span><input type="number" min="2" max="6" step="1" defaultValue={settings.extraction_cycles} disabled={props.disabled} onChange={event => updateNumber('extraction_cycles', Math.round(+event.target.value))}/> 회</span></label>
      <label>운반 기울기 <span><input type="number" min="0" max="45" step="1" defaultValue={settings.carry_tilt_deg} disabled={props.disabled} onChange={event => updateNumber('carry_tilt_deg', +event.target.value)}/> °</span></label>
      <label>삽입 시간 <span><input type="number" min=".2" max="15" step=".1" defaultValue={settings.insertion_seconds} disabled={props.disabled} onChange={event => updateNumber('insertion_seconds', +event.target.value)}/> s</span></label>
      <label>힘 상한 <span><input type="number" min="1" max="200" step="1" defaultValue={settings.force_limit_n} disabled={props.disabled} onChange={event => updateNumber('force_limit_n', +event.target.value)}/> N</span></label>
    </div>
    <details className="ramen-impedance-settings"><summary>임피던스 설정</summary>
      <p className="hint">삽입축은 부드럽게, 틈을 벗어나는 방향과 툴 각도는 강하게 유지합니다. 다른 장면 물체와 접촉해도 같은 TCP 자세 복원 토크가 적용됩니다.</p>
      <div className="ramen-sequence-grid">
        <label>횡방향 강성 <span><input type="number" min="1" max="3000" defaultValue={settings.lateral_stiffness_n_per_m} disabled={props.disabled} onChange={event => updateNumber('lateral_stiffness_n_per_m', +event.target.value)}/> N/m</span></label>
        <label>삽입축 강성 <span><input type="number" min="0" max="1000" defaultValue={settings.insertion_stiffness_n_per_m} disabled={props.disabled} onChange={event => updateNumber('insertion_stiffness_n_per_m', +event.target.value)}/> N/m</span></label>
        <label>회전 강성 <span><input type="number" min="0" max="300" step="1" defaultValue={settings.orientation_stiffness_nm_per_rad} disabled={props.disabled} onChange={event => updateNumber('orientation_stiffness_nm_per_rad', +event.target.value)}/> Nm/rad</span></label>
        <label>토크 사용률 <span><input type="number" min="5" max="80" defaultValue={settings.maximum_feedback_torque_fraction * 100} disabled={props.disabled} onChange={event => updateNumber('maximum_feedback_torque_fraction', +event.target.value / 100)}/> %</span></label>
      </div>
    </details>
    <button className="wide primary" disabled={props.disabled || !targetId} onClick={() => props.onGenerate(targetId, structuredClone(settings))}><Route size={15}/>현재 TCP로 시퀀스 생성</button>
    {!sequence.length ? <p className="hint"><Crosshair size={13}/> 기본자세에서 시작해 물체 중심 위 정렬 → 양쪽 수직 삽입 → 접촉 확인 → 각도 완화·소폭 인양·몸쪽 꺼내기 반복 → 운반자세가 생성됩니다. 삽입·인양 거리는 물체 지름 이상으로 자동 보정됩니다.</p> : <>
      <div className="ramen-sequence-steps">{sequence.map(({ frame, index }, order) => <button key={index} disabled={props.disabled} onClick={() => props.onSelectFrame(index)}><span>{order + 1}</span>{frame.name}</button>)}</div>
      <button className="wide" disabled={props.disabled} onClick={() => props.onRun(sequence[0].index)}><Play size={15}/>이 시퀀스 물리 실행</button>
      <p className="ramen-safety-note"><ShieldCheck size={14}/> 힘 상한이 10 ms 이상 계속되면 물리 실행을 중단합니다.</p>
    </>}
  </section>;
}
