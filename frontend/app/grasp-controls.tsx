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
  inward_offset_m: 0, closure_seconds: .4, target_force_n: 8, max_force_n: 60,
});

export default function GraspControls({ objects, selectedObjectId, grasp, disabled, pickMode, onChange, onPickMode, onFit, onValidate }: Props) {
  const boxes = objects.filter(object => object.shape === 'box' && object.visible);
  const selectedBox = boxes.find(object => object.id === (grasp?.object_id ?? selectedObjectId));
  const maxOffset = Math.min(.06, Math.max(0, (selectedBox?.size[1] ?? .09) - .03));
  const fitted = !!grasp?.object_signature && !grasp.closure_qpos;
  const update = (patch: Partial<TwoHandGrasp>) => {
    if (!selectedBox) return;
    onChange({ ...(grasp ?? freshGrasp(selectedBox.id)), ...patch, object_id: patch.object_id ?? selectedBox.id,
      closure_qpos: undefined, object_signature: undefined, contact_points_world: undefined, hand_twist_deg: undefined });
  };
  return <section className="grasp-controls">
    <div className="panel-heading"><span>양손 파지</span><small>키프레임 상호작용</small></div>
    {!grasp ? <>
      <p className="hint">손가락 끝과 손목 요 링크 사이를 잇는 청록·주황 면은 자세를 맞추는 가상 기준면이며 충돌하지 않습니다. 물리 재생에서는 실제 손·손목 메시가 박스와 접촉합니다.</p>
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
      <label className="range-label">파지면 위치 오프셋 <span>{Math.round(grasp.inward_offset_m * 1000)} mm</span><input type="range" min="0" max={Math.round(maxOffset * 1000)} step="1" disabled={disabled} value={Math.round(grasp.inward_offset_m * 1000)} onChange={event => update({ inward_offset_m: +event.target.value / 1000 })}/></label>
      <div className="group-actions"><button disabled={disabled || grasp.inward_offset_m <= 0} onClick={() => update({ inward_offset_m: Math.max(0, grasp.inward_offset_m - .001) })}>−1 mm</button><button disabled={disabled || grasp.inward_offset_m >= maxOffset} onClick={() => update({ inward_offset_m: Math.min(maxOffset, grasp.inward_offset_m + .001) })}>+1 mm</button></div>
      <p className="hint">박스 양쪽 표면을 기준으로 두 손의 간격을 이 값만큼 줄입니다(손당 절반). 보조면 방향은 박스 옆면과 평행하게 유지되며, 값을 바꾼 뒤 자세 맞추기를 누르면 편집 자세 자체에 적용됩니다.</p>
      <div className="grasp-force-row"><label>검증 목표 힘 <input type="number" min="1" max="200" value={grasp.target_force_n} disabled={disabled} onChange={event => { const value = +event.target.value; update({ target_force_n: value, max_force_n: Math.max(value, grasp.max_force_n) }); }}/>N</label><label>안전 상한 <input type="number" min={grasp.target_force_n} max="400" value={grasp.max_force_n} disabled={disabled} onChange={event => update({ max_force_n: Math.max(grasp.target_force_n, +event.target.value) })}/>N</label></div>
      <button className="wide primary" disabled={disabled || !selectedBox} onClick={onFit}><Hand size={15}/>양손 파지 자세 맞추기</button>
      <button className="wide" disabled={disabled || !fitted} onClick={onValidate}><Play size={15}/>이 키프레임부터 물리 검증</button>
      {fitted && <p className="hint">박스 평행 파지 자세 적용 완료 · 손목 비틀림 {grasp.hand_twist_deg?.toFixed(0) ?? 0}° · 물리 재생에서도 이 편집 자세를 그대로 사용합니다.</p>}
      {grasp.closure_qpos && <p className="hint">이 키프레임은 이전 방식의 재생용 닫힘 보정값을 포함합니다. 현재 오프셋 방식으로 다시 맞추면 편집 자세에 직접 반영됩니다.</p>}
      <button className="wide danger-subtle" disabled={disabled} onClick={() => { onPickMode(null); onChange(undefined); }}><RotateCcw size={14}/>파지 설정 제거</button>
      <p className="hint">위치 오프셋은 힘 명령이 아닙니다. 실제 접촉력은 MuJoCo 검증 결과에서 측정합니다.</p>
    </>}
  </section>;
}
