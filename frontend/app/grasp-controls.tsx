'use client';
import { Crosshair, Hand, Play, RotateCcw } from 'lucide-react';
import type { SceneObject } from '../lib/scene-objects';
import type { TwoHandGrasp } from '../lib/keyframes';

export type GraspPickMode = 'left' | 'right' | null;

type Props = {
  objects: SceneObject[];
  selectedObjectId: string | null;
  grasp?: TwoHandGrasp;
  disabled: boolean;
  pickMode: GraspPickMode;
  onChange: (grasp?: TwoHandGrasp) => void;
  onPickMode: (mode: GraspPickMode) => void;
  onFit: () => void;
  onValidate: () => void;
};

const freshGrasp = (objectId: string): TwoHandGrasp => ({
  format: 'motioncreator.two-hand-grasp.v1', object_id: objectId,
  left_surface_uv: [0, 0], right_surface_uv: [0, 0],
  inward_offset_m: .02, closure_seconds: .4, target_force_n: 8, max_force_n: 60,
});

export default function GraspControls({ objects, selectedObjectId, grasp, disabled, pickMode, onChange, onPickMode, onFit, onValidate }: Props) {
  const boxes = objects.filter(object => object.shape === 'box' && object.visible);
  const selectedBox = boxes.find(object => object.id === (grasp?.object_id ?? selectedObjectId));
  const update = (patch: Partial<TwoHandGrasp>) => {
    if (!selectedBox) return;
    onChange({ ...(grasp ?? freshGrasp(selectedBox.id)), ...patch, object_id: patch.object_id ?? selectedBox.id,
      closure_qpos: undefined, object_signature: undefined, contact_points_world: undefined, hand_twist_deg: undefined });
  };
  return <section className="grasp-controls">
    <div className="panel-heading"><span>양손 파지</span><small>키프레임 상호작용</small></div>
    {!grasp ? <>
      <p className="hint">선택 키프레임의 자세를 기반으로 페이크 핸드의 손바닥 하단–손목 접촉부를 박스 양면에 맞춥니다.</p>
      <button className="wide" disabled={disabled || !selectedBox} onClick={() => selectedBox && onChange(freshGrasp(selectedBox.id))}><Hand size={15}/>선택 박스에 파지 설정</button>
      {!selectedBox && <p className="hint">먼저 장면에서 파지할 박스를 선택하세요.</p>}
    </> : <>
      <div className="inspector-label">대상 박스</div>
      <select value={grasp.object_id} disabled={disabled} onChange={event => update({ object_id: event.target.value })}>{boxes.map(object => <option key={object.id} value={object.id}>{object.name}</option>)}</select>
      <div className="grasp-pick-row">
        <button className={pickMode === 'left' ? 'chosen' : ''} disabled={disabled} onClick={() => onPickMode(pickMode === 'left' ? null : 'left')}><Crosshair size={14}/>왼손 표면 찍기</button>
        <button className={pickMode === 'right' ? 'chosen' : ''} disabled={disabled} onClick={() => onPickMode(pickMode === 'right' ? null : 'right')}><Crosshair size={14}/>오른손 표면 찍기</button>
      </div>
      {pickMode && <p className="grasp-pick-hint">박스의 {pickMode === 'left' ? '+Y 왼쪽' : '−Y 오른쪽'} 면을 클릭하세요. 중심선 근처는 자동 스냅됩니다.</p>}
      <div className="group-actions"><button disabled={disabled} onClick={() => update({ left_surface_uv: [0, 0], right_surface_uv: [0, 0] })}>양면 정가운데</button><button disabled={disabled} onClick={() => update({ right_surface_uv: [...grasp.left_surface_uv] })}>왼손 위치 대칭</button></div>
      <div className="grasp-coordinates"><span>왼손 <b>{Math.round(grasp.left_surface_uv[0] * 100)}%, {Math.round(grasp.left_surface_uv[1] * 100)}%</b></span><span>오른손 <b>{Math.round(grasp.right_surface_uv[0] * 100)}%, {Math.round(grasp.right_surface_uv[1] * 100)}%</b></span></div>
      <label className="range-label">가상 안쪽 오프셋 <span>{Math.round(grasp.inward_offset_m * 1000)} mm</span><input type="range" min="0" max={Math.min(60, Math.max(0, Math.round(((selectedBox?.size[1] ?? .09) - .03) * 1000)))} step="1" disabled={disabled} value={Math.round(grasp.inward_offset_m * 1000)} onChange={event => update({ inward_offset_m: +event.target.value / 1000 })}/></label>
      <label className="range-label">닫기 시간 <span>{grasp.closure_seconds.toFixed(1)} s</span><input type="range" min=".1" max="2" step=".1" disabled={disabled} value={grasp.closure_seconds} onChange={event => update({ closure_seconds: +event.target.value })}/></label>
      <div className="grasp-force-row"><label>검증 목표 힘 <input type="number" min="1" max="200" value={grasp.target_force_n} disabled={disabled} onChange={event => { const value = +event.target.value; update({ target_force_n: value, max_force_n: Math.max(value, grasp.max_force_n) }); }}/>N</label><label>안전 상한 <input type="number" min={grasp.target_force_n} max="400" value={grasp.max_force_n} disabled={disabled} onChange={event => update({ max_force_n: Math.max(grasp.target_force_n, +event.target.value) })}/>N</label></div>
      <button className="wide primary" disabled={disabled || !selectedBox} onClick={onFit}><Hand size={15}/>양손 파지 자세 맞추기</button>
      <button className="wide" disabled={disabled || !grasp.closure_qpos} onClick={onValidate}><Play size={15}/>이 키프레임부터 물리 검증</button>
      {grasp.closure_qpos && <p className="hint">접촉 자세 보정 완료 · 손목 비틀림 {grasp.hand_twist_deg?.toFixed(0) ?? 0}° · 물리 재생 시작 시 {grasp.closure_seconds.toFixed(1)}초 동안 닫습니다.</p>}
      <button className="wide danger-subtle" disabled={disabled} onClick={() => { onPickMode(null); onChange(undefined); }}><RotateCcw size={14}/>파지 설정 제거</button>
      <p className="hint">안쪽 오프셋은 힘이 아닌 위치 목표입니다. 접촉력은 MuJoCo 검증 결과에서 측정합니다.</p>
    </>}
  </section>;
}
