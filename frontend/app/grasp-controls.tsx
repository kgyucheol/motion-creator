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

const freshGrasp = (object: SceneObject): TwoHandGrasp => ({
  format: 'motioncreator.two-hand-grasp.v1', object_id: object.id,
  left_surface_uv: [0, 0], right_surface_uv: [0, 0],
  hand_gap_m: object.size[1], closure_seconds: .4, target_force_n: 8, max_force_n: 60,
  follow_object: true,
});

export default function GraspControls({ objects, selectedObjectId, grasp, disabled, pickMode, onChange, onPickMode, onFit, onValidate }: Props) {
  const boxes = objects.filter(object => object.shape === 'box' && object.visible);
  const selectedBox = boxes.find(object => object.id === (grasp?.object_id ?? selectedObjectId));
  const boxWidth = selectedBox?.size[1] ?? .09;
  const handGap = grasp?.hand_gap_m ?? Math.max(0, boxWidth - (grasp?.inward_offset_m ?? 0));
  const maxGap = Math.min(1.2, Math.max(.6, boxWidth + .2));
  const clearance = handGap - boxWidth;
  const fitted = !!grasp?.object_signature && !grasp.closure_qpos;
  const update = (patch: Partial<TwoHandGrasp>) => {
    if (!selectedBox) return;
    const preservesFit = Object.keys(patch).every(key => ['follow_object', 'target_force_n', 'max_force_n'].includes(key));
    onChange({ ...(grasp ?? freshGrasp(selectedBox)), hand_gap_m: handGap, inward_offset_m: undefined,
      ...patch, object_id: patch.object_id ?? selectedBox.id,
      ...(preservesFit ? {} : { closure_qpos: undefined, object_signature: undefined,
        contact_points_world: undefined, hand_twist_deg: undefined }) });
  };
  return <section className="grasp-controls">
    <div className="panel-heading"><span>양손 파지</span><small>키프레임 상호작용</small></div>
    {!grasp ? <>
      <p className="hint">손가락 끝과 손목 요 링크 사이를 잇는 청록·주황 면은 자세를 맞추는 가상 기준면이며 충돌하지 않습니다. 물리 재생에서는 실제 손·손목 메시가 박스와 접촉합니다.</p>
      <button className="wide" disabled={disabled || !selectedBox} onClick={() => selectedBox && onChange(freshGrasp(selectedBox))}><Hand size={15}/>선택 박스에 파지 설정</button>
      {!selectedBox && <p className="hint">먼저 장면에서 파지할 박스를 선택하세요.</p>}
    </> : <>
      <div className="inspector-label">대상 박스</div>
      <select value={grasp.object_id} disabled={disabled} onChange={event => {
        const object = boxes.find(value => value.id === event.target.value);
        if (object) onChange(freshGrasp(object));
      }}>{boxes.map(object => <option key={object.id} value={object.id}>{object.name}</option>)}</select>
      <div className="grasp-pick-row">
        <button className={pickMode === 'left' ? 'chosen' : ''} disabled={disabled} onClick={() => onPickMode(pickMode === 'left' ? null : 'left')}><Crosshair size={14}/>왼손 표면 찍기</button>
        <button className={pickMode === 'right' ? 'chosen' : ''} disabled={disabled} onClick={() => onPickMode(pickMode === 'right' ? null : 'right')}><Crosshair size={14}/>오른손 표면 찍기</button>
      </div>
      {pickMode && <p className="grasp-pick-hint">박스의 {pickMode === 'left' ? '+Y 왼쪽' : '−Y 오른쪽'} 면을 클릭하세요. 중심선 근처는 자동 스냅됩니다.</p>}
      <div className="group-actions"><button disabled={disabled} onClick={() => update({ left_surface_uv: [0, 0], right_surface_uv: [0, 0] })}>양면 정가운데</button><button disabled={disabled} onClick={() => update({ right_surface_uv: [...grasp.left_surface_uv] })}>왼손 위치 대칭</button></div>
      <div className="grasp-coordinates"><span>왼손 <b>{Math.round(grasp.left_surface_uv[0] * 100)}%, {Math.round(grasp.left_surface_uv[1] * 100)}%</b></span><span>오른손 <b>{Math.round(grasp.right_surface_uv[0] * 100)}%, {Math.round(grasp.right_surface_uv[1] * 100)}%</b></span></div>
      <label className="range-label">양손 파지면 간격 <span>{Math.round(handGap * 1000)} mm</span><input type="range" min="0" max={Math.round(maxGap * 1000)} step="1" disabled={disabled} value={Math.round(handGap * 1000)} onChange={event => update({ hand_gap_m: +event.target.value / 1000 })}/></label>
      <div className="group-actions"><button disabled={disabled || handGap <= 0} onClick={() => update({ hand_gap_m: Math.max(0, handGap - .001) })}>−1 mm</button><button disabled={disabled} onClick={() => update({ hand_gap_m: boxWidth })}>박스 너비 {Math.round(boxWidth * 1000)} mm</button><button disabled={disabled || handGap >= maxGap} onClick={() => update({ hand_gap_m: Math.min(maxGap, handGap + .001) })}>+1 mm</button></div>
      <p className="hint">0 mm에서는 두 보조면이 맞닿고, 박스 너비에서는 양쪽 면과 일치합니다. {clearance > .0005 ? `현재는 면보다 ${Math.round(clearance * 1000)} mm 벌어진 접근 자세입니다.` : clearance < -.0005 ? `현재는 박스보다 ${Math.round(-clearance * 1000)} mm 작은, 접촉력을 만들기 위한 닫힘 목표입니다.` : '현재 보조면이 박스 양쪽 면에 맞춰집니다.'} 값을 바꾼 뒤 자세 맞추기를 누르세요.</p>
      <div className="grasp-force-row"><label>검증 최소 힘 <input type="number" min="1" max="200" value={grasp.target_force_n} disabled={disabled} onChange={event => { const value = +event.target.value; update({ target_force_n: value, max_force_n: Math.max(value, grasp.max_force_n) }); }}/>N</label><label>검증 안전 상한 <input type="number" min={grasp.target_force_n} max="400" value={grasp.max_force_n} disabled={disabled} onChange={event => update({ max_force_n: Math.max(grasp.target_force_n, +event.target.value) })}/>N</label></div>
      <button className="wide primary" disabled={disabled || !selectedBox} onClick={onFit}><Hand size={15}/>양손 파지 자세 맞추기</button>
      <label className="checkbox"><input type="checkbox" checked={grasp.follow_object !== false} disabled={disabled || !fitted} onChange={event => update({ follow_object: event.target.checked })}/>이 키프레임에서 박스 이동 시 양손 따라가기</label>
      <button className="wide" disabled={disabled || !fitted} onClick={onValidate}><Play size={15}/>이 키프레임부터 물리 검증</button>
      {fitted && <p className="hint">박스 평행 파지 자세 적용 완료 · 손목 비틀림 {grasp.hand_twist_deg?.toFixed(0) ?? 0}° · 물리 재생에서도 이 편집 자세를 그대로 사용합니다.</p>}
      {grasp.closure_qpos && <p className="hint">이 키프레임은 이전 방식의 재생용 닫힘 보정값을 포함합니다. 현재 오프셋 방식으로 다시 맞추면 편집 자세에 직접 반영됩니다.</p>}
      <button className="wide danger-subtle" disabled={disabled} onClick={() => { onPickMode(null); onChange(undefined); }}><RotateCcw size={14}/>파지 설정 제거</button>
      <p className="hint">간격은 힘 명령이 아니며, 박스보다 작은 간격은 물리가 막아야 하는 닫힘 목표입니다. 실제 접촉력은 MuJoCo 검증 결과에서 측정합니다.</p>
    </>}
  </section>;
}
