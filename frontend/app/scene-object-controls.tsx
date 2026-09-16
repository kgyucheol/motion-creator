import { Plus, Trash2 } from 'lucide-react';
import { eulerDegrees, quaternionFromDegrees } from '../lib/pose-transforms';
import { normalizedObjectSize, type ObjectTransformMode, type SceneObject, type SceneObjectShape } from '../lib/scene-objects';

type Props = {
  objects: SceneObject[];
  selectedId: string | null;
  mode: ObjectTransformMode;
  disabled: boolean;
  onSelect: (id: string | null) => void;
  onAdd: (shape: SceneObjectShape) => void;
  onRemove: (id: string) => void;
  onChange: (id: string, patch: Partial<SceneObject>) => void;
  onModeChange: (mode: ObjectTransformMode) => void;
};

const shapeLabels: Record<SceneObjectShape, string> = { box: '박스', cylinder: '원통', sphere: '구' };

export default function SceneObjectControls(props: Props) {
  const selected = props.objects.find(object => object.id === props.selectedId) ?? null;
  const patchVector = (field: 'position' | 'size', index: number, value: number) => {
    if (!selected || !Number.isFinite(value)) return;
    const vector = selected[field].map((current, i) => i === index ? value : current);
    props.onChange(selected.id, { [field]: field === 'size' ? normalizedObjectSize(selected.shape, vector, 'XYZ'[index]) : vector });
  };
  const rotation = selected ? eulerDegrees(selected.quaternion_xyzw) : [0, 0, 0];

  return <>
    <div className="panel-heading"><span>SCENE OBJECTS</span><small>{objectsLabel(props.objects.length)}</small></div>
    <div className="object-add-row">
      {(['box', 'cylinder', 'sphere'] as SceneObjectShape[]).map(shape => <button key={shape} disabled={props.disabled || props.objects.length >= 32} onClick={() => props.onAdd(shape)}><Plus size={13}/>{shapeLabels[shape]}</button>)}
    </div>
    {!props.objects.length && <p className="hint">장면 물체가 없습니다. 위 버튼으로 MuJoCo 기본 도형을 추가하세요.</p>}
    {!!props.objects.length && <select aria-label="장면 물체 선택" value={props.selectedId ?? ''} disabled={props.disabled} onChange={event => props.onSelect(event.target.value || null)}>
      <option value="">물체 선택</option>
      {props.objects.map(object => <option key={object.id} value={object.id}>{object.name} · {shapeLabels[object.shape]}</option>)}
    </select>}
    {selected && <div className="object-editor">
      <div className="object-title-row"><input aria-label="물체 이름" value={selected.name} maxLength={80} disabled={props.disabled} onChange={event => props.onChange(selected.id, { name: event.target.value || selected.name })}/><button title="물체 삭제" disabled={props.disabled} onClick={() => props.onRemove(selected.id)}><Trash2 size={14}/></button></div>
      <div className="segmented object-modes">{(['translate', 'rotate', 'scale'] as ObjectTransformMode[]).map((mode, index) => <button key={mode} className={props.mode === mode ? 'chosen' : ''} disabled={props.disabled} onClick={() => props.onModeChange(mode)}>{['이동 W', '회전 E', '크기 R'][index]}</button>)}</div>
      <div className="inspector-label">도형</div>
      <select aria-label="물체 도형" value={selected.shape} disabled={props.disabled} onChange={event => {
        const shape = event.target.value as SceneObjectShape;
        props.onChange(selected.id, { shape, size: normalizedObjectSize(shape, selected.size) });
      }}>{Object.entries(shapeLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
      <label className="checkbox"><input type="checkbox" checked={selected.visible} disabled={props.disabled} onChange={event => props.onChange(selected.id, { visible: event.target.checked })}/>화면에 표시</label>
      <div className="inspector-label">위치 · 월드 XYZ <small>m</small></div>
      <div className="xyz">{selected.position.map((value, index) => <label key={index}><span>{'XYZ'[index]}</span><input aria-label={`물체 위치 ${'XYZ'[index]}`} type="number" step=".01" value={Number(value.toFixed(4))} disabled={props.disabled} onChange={event => patchVector('position', index, +event.target.value)}/></label>)}</div>
      <div className="inspector-label">회전 · 월드 XYZ <small>°</small></div>
      <div className="xyz">{rotation.map((value, index) => <label key={index}><span>{'XYZ'[index]}</span><input aria-label={`물체 회전 ${'XYZ'[index]}`} type="number" step="1" value={Number(value.toFixed(2))} disabled={props.disabled} onChange={event => {
        const next = rotation.map((current, i) => i === index ? +event.target.value : current);
        if (next.every(Number.isFinite)) props.onChange(selected.id, { quaternion_xyzw: quaternionFromDegrees(next) });
      }}/></label>)}</div>
      <div className="inspector-label">{selected.shape === 'sphere' ? '지름' : selected.shape === 'cylinder' ? '지름 XY · 높이 Z' : '크기 XYZ'} <small>m</small></div>
      <div className="xyz">{selected.size.map((value, index) => <label key={index} className={selected.shape === 'sphere' && index > 0 || selected.shape === 'cylinder' && index === 1 ? 'linked-size' : ''}><span>{'XYZ'[index]}</span><input aria-label={`물체 크기 ${'XYZ'[index]}`} type="number" min=".01" step=".01" value={Number(value.toFixed(4))} disabled={props.disabled || selected.shape === 'sphere' && index > 0 || selected.shape === 'cylinder' && index === 1} onChange={event => patchVector('size', index, +event.target.value)}/></label>)}</div>
      <div className="object-physics-row"><label>질량 <span><input aria-label="물체 질량" type="number" min=".001" max="1000" step=".1" value={selected.mass_kg} disabled={props.disabled} onChange={event => props.onChange(selected.id, { mass_kg: Math.max(.001, Math.min(1000, +event.target.value || .001)) })}/> kg</span></label><label>마찰 <input aria-label="물체 마찰" type="number" min="0" max="2" step=".05" value={selected.friction} disabled={props.disabled} onChange={event => props.onChange(selected.id, { friction: Math.max(0, Math.min(2, +event.target.value || 0)) })}/></label></div>
      <div className="object-appearance-row"><label>색상 <input aria-label="물체 색상" type="color" value={selected.color} disabled={props.disabled} onChange={event => props.onChange(selected.id, { color: event.target.value })}/></label><label>불투명도 <span>{Math.round(selected.opacity * 100)}%</span><input aria-label="물체 불투명도" type="range" min=".05" max="1" step=".01" value={selected.opacity} disabled={props.disabled} onChange={event => props.onChange(selected.id, { opacity: +event.target.value })}/></label></div>
      <p className="hint">기즈모로 이동·회전·크기를 조절합니다. 구는 균일 지름, 원통은 원형 지름과 높이를 유지합니다. 물리 재생을 다시 시작하면 이 초기 위치와 방향에서 출발합니다.</p>
    </div>}
  </>;
}

function objectsLabel(count: number) {
  return count ? `${count}개 · 최대 32개` : '최대 32개';
}
