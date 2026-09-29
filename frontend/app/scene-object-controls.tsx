import { useState } from 'react';
import { Boxes, Crosshair, Plus, Trash2, Upload } from 'lucide-react';
import { eulerDegrees, quaternionFromDegrees } from '../lib/pose-transforms';
import { isRamenBundle, normalizedObjectSize, reparentSceneObjects, selectSceneObjectRows, type CandidateGraspPoints, type ObjectTransformMode, type SceneObject, type SceneObjectGroup, type SceneObjectShape, type ScenePlacementOptions } from '../lib/scene-objects';

type Props = {
  objects: SceneObject[];
  selectedId: string | null;
  mode: ObjectTransformMode;
  disabled: boolean;
  preventOverlap: boolean;
  surfaceSnap: boolean;
  groundLock: boolean;
  groups: SceneObjectGroup[];
  selectedGroupId: string | null;
  onSelect: (id: string | null) => void;
  onAdd: (shape: SceneObjectShape) => void;
  onRemove: (id: string) => void;
  onChange: (id: string, patch: Partial<SceneObject>) => void;
  onModeChange: (mode: ObjectTransformMode) => void;
  onPlacementChange: (patch: Partial<ScenePlacementOptions>) => void;
  onImport: () => void;
  onParentChange: (ids: string[], parentId: string | null) => void;
  onCreateGroup: (ids: string[], name: string) => void;
  onSelectGroup: (id: string | null) => void;
  onChangeGroup: (id: string, patch: Partial<SceneObjectGroup>) => void;
  onRemoveGroup: (id: string) => void;
  graspPreview: CandidateGraspPoints | null;
  onToggleGraspPreview: (id: string) => void;
};

const shapeLabels: Record<SceneObjectShape, string> = { box: '박스', open_box: '열린 상자', cylinder: '원통', sphere: '구' };
const SHOW_LEGACY_OBJECT_GROUPS = false;

export default function SceneObjectControls(props: Props) {
  const [groupMembers, setGroupMembers] = useState<string[]>([]);
  const [groupName, setGroupName] = useState('');
  const [treeSelection, setTreeSelection] = useState({ ids: props.selectedId ? [props.selectedId] : [] as string[],
    anchorId: props.selectedId, primaryId: props.selectedId });
  const [draggedIds, setDraggedIds] = useState<string[]>([]);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const selectedIds = props.selectedId === treeSelection.primaryId ? treeSelection.ids
    : props.selectedId ? [props.selectedId] : [];
  const anchorId = props.selectedId === treeSelection.primaryId ? treeSelection.anchorId : props.selectedId;
  const selected = props.objects.find(object => object.id === props.selectedId) ?? null;
  const selectedGroup = props.groups.find(group => group.id === props.selectedGroupId) ?? null;
  const grouped = new Set(props.groups.flatMap(group => group.member_ids));
  const patchVector = (field: 'position' | 'size', index: number, value: number) => {
    if (!selected || !Number.isFinite(value)) return;
    const vector = selected[field].map((current, i) => i === index ? value : current);
    props.onChange(selected.id, { [field]: field === 'size' ? normalizedObjectSize(selected.shape, vector, 'XYZ'[index]) : vector });
  };
  const rotation = selected ? eulerDegrees(selected.quaternion_xyzw) : [0, 0, 0];
  const roots = props.objects.filter(object => !object.parent_id || !props.objects.some(parent => parent.id === object.parent_id));
  const orderedIds: string[] = [];
  const collectIds = (object: SceneObject) => {
    orderedIds.push(object.id);
    props.objects.filter(child => child.parent_id === object.id).forEach(collectIds);
  };
  roots.forEach(collectIds);
  const activeIds = selectedIds.filter(id => props.objects.some(object => object.id === id));
  const canParent = (ids: string[], parentId: string | null) => ids.length > 0
    && reparentSceneObjects(props.objects, ids, parentId) !== props.objects;
  const selectRow = (id: string, toggle: boolean, range: boolean) => {
    const next = selectSceneObjectRows(activeIds, anchorId, id, orderedIds, toggle, range);
    const primaryId = next.ids.includes(id) ? id : next.ids.at(-1) ?? null;
    setTreeSelection({ ...next, primaryId });
    props.onSelect(primaryId);
  };
  const clearSelection = () => {
    setTreeSelection({ ids: [], anchorId: null, primaryId: null });
    setDraggedIds([]); setDropTarget(null);
    props.onSelect(null);
  };
  const renderObjectRow = (object: SceneObject, depth: number) => <div key={object.id}>
    <button className={`scene-tree-row ${activeIds.includes(object.id) ? 'selected' : ''} ${dropTarget === object.id ? 'drop-target' : ''}`}
      style={{ paddingLeft: `${8 + depth * 16}px` }} disabled={props.disabled} draggable={!props.disabled}
      aria-pressed={activeIds.includes(object.id)}
      onClick={event => selectRow(object.id, event.ctrlKey || event.metaKey, event.shiftKey)}
      onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); clearSelection(); } }}
      onDragStart={event => {
        const ids = activeIds.includes(object.id) ? activeIds : [object.id];
        if (!activeIds.includes(object.id)) selectRow(object.id, false, false);
        event.dataTransfer.setData('text/plain', object.name);
        event.dataTransfer.effectAllowed = 'move'; setDraggedIds(ids);
      }}
      onDragEnd={() => { setDraggedIds([]); setDropTarget(null); }}
      onDragOver={event => { if (canParent(draggedIds, object.id)) { event.preventDefault(); setDropTarget(object.id); } }}
      onDragLeave={() => setDropTarget(value => value === object.id ? null : value)}
      onDrop={event => { event.preventDefault(); if (canParent(draggedIds, object.id)) props.onParentChange(draggedIds, object.id); setDraggedIds([]); setDropTarget(null); }}>
      <span className="scene-tree-grip" aria-hidden="true">⋮⋮</span><span className="scene-tree-name">{object.name}</span><small>{shapeLabels[object.shape]}</small>
    </button>
    {props.objects.filter(child => child.parent_id === object.id).map(child => renderObjectRow(child, depth + 1))}
  </div>;

  return <>
    <div className="panel-heading"><span>SCENE OBJECTS</span><small>{objectsLabel(props.objects.length)}</small></div>
    <div className="object-add-row">
      {(['box', 'cylinder', 'sphere'] as SceneObjectShape[]).map(shape => <button key={shape} disabled={props.disabled || props.objects.length >= 32} onClick={() => props.onAdd(shape)}><Plus size={13}/>{shapeLabels[shape]}</button>)}
    </div>
    <button className="wide" disabled={props.disabled || props.objects.length >= 32} onClick={props.onImport}><Upload size={14}/> 3D 모델 가져오기 (.blend / .glb)</button>
    {!props.objects.length && <p className="hint">장면 물체가 없습니다. 위 버튼으로 MuJoCo 기본 도형을 추가하세요.</p>}
    {!!props.objects.length && <><p className="hint">Ctrl/⌘+클릭으로 다중 선택, Shift+클릭으로 범위 선택한 뒤 드래그해 부모에 넣을 수 있습니다. 현재 위치는 유지됩니다.</p>
      <div className="scene-tree-selection-bar"><small>{activeIds.length}개 선택</small><button type="button" disabled={props.disabled || !activeIds.length} onClick={clearSelection}>선택 해제 <kbd>Esc</kbd></button></div>
      <div className="scene-tree" aria-label="장면 오브젝트 계층">{roots.map(object => renderObjectRow(object, 0))}</div>
      <button className="scene-tree-root" disabled={props.disabled || !canParent(draggedIds.length ? draggedIds : activeIds, null)} onClick={() => props.onParentChange(activeIds, null)}
        onDragOver={event => { if (canParent(draggedIds, null)) event.preventDefault(); }}
        onDrop={event => { event.preventDefault(); if (canParent(draggedIds, null)) props.onParentChange(draggedIds, null); setDraggedIds([]); setDropTarget(null); }}>
        선택 물체 부모 해제 · 여기에 놓으면 최상위로
      </button></>}
    <div className="object-placement-options">
      <label className="checkbox"><input type="checkbox" checked={props.preventOverlap} disabled={props.disabled || !selected} onChange={event => props.onPlacementChange({ preventOverlap: event.target.checked })}/>겹침 방지</label>
      <label className="checkbox"><input type="checkbox" checked={props.surfaceSnap} disabled={props.disabled || !selected} onChange={event => props.onPlacementChange({ surfaceSnap: event.target.checked })}/>표면 스냅</label>
      <label className="checkbox"><input type="checkbox" checked={props.groundLock} disabled={props.disabled || !selected} onChange={event => props.onPlacementChange({ groundLock: event.target.checked })}/>지면 고정</label>
    </div>
    {selected && <div className="object-editor">
      <div className="object-title-row"><input aria-label="물체 이름" value={selected.name} maxLength={80} disabled={props.disabled} onChange={event => props.onChange(selected.id, { name: event.target.value || selected.name })}/><button title="물체와 하위 물체 삭제" disabled={props.disabled} onClick={() => props.onRemove(selected.id)}><Trash2 size={14}/></button></div>
      <div className="segmented object-modes">{(['translate', 'rotate', 'scale'] as ObjectTransformMode[]).map((mode, index) => <button key={mode} className={props.mode === mode ? 'chosen' : ''} disabled={props.disabled} onClick={() => props.onModeChange(mode)}>{['이동 W', '회전 E', '크기 R'][index]}</button>)}</div>
      {isRamenBundle(selected) && <div className="candidate-grasp-control"><button className="wide" type="button" aria-pressed={!!props.graspPreview} disabled={props.disabled} onClick={() => props.onToggleGraspPreview(selected.id)}><Crosshair size={14}/>{props.graspPreview ? '파지점 미리보기 끄기' : '파지점 미리보기'}</button>{props.graspPreview && <div className="candidate-grasp-coordinates">{(['left', 'right'] as const).map(side => <div key={side}><span>{side === 'left' ? '왼손' : '오른손'}</span><code>{props.graspPreview![side].map(value => value.toFixed(3)).join(', ')} m</code></div>)}</div>}</div>}
      <div className="inspector-label">{selected.asset_id ? '물리 충돌체' : '도형'}</div>
      <select aria-label="물체 도형" value={selected.shape} disabled={props.disabled || !!selected.asset_id} onChange={event => {
        const shape = event.target.value as SceneObjectShape;
        props.onChange(selected.id, { shape, size: normalizedObjectSize(shape, selected.size) });
      }}>{Object.entries(shapeLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
      {selected.asset_id && <p className="hint">화면에는 외부 3D 메시를 표시하고, 물리 계산은 이 단순 충돌체를 사용합니다.</p>}
      <label className="checkbox"><input type="checkbox" checked={selected.visible} disabled={props.disabled} onChange={event => props.onChange(selected.id, { visible: event.target.checked })}/>화면에 표시</label>
      <label className="checkbox"><input type="checkbox" checked={!!selected.fixed} disabled={props.disabled} onChange={event => props.onChange(selected.id, { fixed: event.target.checked })}/>물리에서 고정</label>
      <div className="inspector-label">위치 · 월드 XYZ <small>m</small></div>
      <div className="xyz">{selected.position.map((value, index) => <label key={index}><span>{'XYZ'[index]}</span><input aria-label={`물체 위치 ${'XYZ'[index]}`} type="number" step=".01" value={Number(value.toFixed(4))} disabled={props.disabled || props.groundLock && index === 2} onChange={event => patchVector('position', index, +event.target.value)}/></label>)}</div>
      <div className="inspector-label">회전 · 월드 XYZ <small>°</small></div>
      <div className="xyz">{rotation.map((value, index) => <label key={index}><span>{'XYZ'[index]}</span><input aria-label={`물체 회전 ${'XYZ'[index]}`} type="number" step="1" value={Number(value.toFixed(2))} disabled={props.disabled} onChange={event => {
        const next = rotation.map((current, i) => i === index ? +event.target.value : current);
        if (next.every(Number.isFinite)) props.onChange(selected.id, { quaternion_xyzw: quaternionFromDegrees(next) });
      }}/></label>)}</div>
      <div className="inspector-label">{selected.shape === 'sphere' ? '지름' : selected.shape === 'cylinder' ? '지름 XY · 높이 Z' : '크기 XYZ'} <small>m</small></div>
      <div className="xyz">{selected.size.map((value, index) => <label key={index} className={selected.shape === 'sphere' && index > 0 || selected.shape === 'cylinder' && index === 1 ? 'linked-size' : ''}><span>{'XYZ'[index]}</span><input aria-label={`물체 크기 ${'XYZ'[index]}`} type="number" min=".01" step=".01" value={Number(value.toFixed(4))} disabled={props.disabled || selected.shape === 'sphere' && index > 0 || selected.shape === 'cylinder' && index === 1} onChange={event => patchVector('size', index, +event.target.value)}/></label>)}</div>
      <div className="object-physics-row"><label>질량 <span><input aria-label="물체 질량" type="number" min=".001" max="1000" step=".1" value={selected.mass_kg} disabled={props.disabled} onChange={event => props.onChange(selected.id, { mass_kg: Math.max(.001, Math.min(1000, +event.target.value || .001)) })}/> kg</span></label><label>마찰 <input aria-label="물체 마찰" type="number" min="0" max="2" step=".05" value={selected.friction} disabled={props.disabled} onChange={event => props.onChange(selected.id, { friction: Math.max(0, Math.min(2, +event.target.value || 0)) })}/></label></div>
      <div className="object-appearance-row"><label>색상 <input aria-label="물체 색상" type="color" value={selected.color} disabled={props.disabled} onChange={event => props.onChange(selected.id, { color: event.target.value })}/></label><label>불투명도 <span>{Math.round(selected.opacity * 100)}%</span><input aria-label="물체 불투명도" type="range" min=".05" max="1" step=".01" value={selected.opacity} disabled={props.disabled} onChange={event => props.onChange(selected.id, { opacity: +event.target.value })}/></label></div>
      <p className="hint">기즈모로 이동·회전·크기를 조절하고 Delete 또는 Backspace로 선택 물체를 삭제합니다. 겹침 방지는 회전된 도형의 바깥 경계를 기준으로 물체를 가장 가까운 비충돌 위치에 둡니다. 표면 스냅은 2cm 이내의 물체 표면에 붙이고, 지면 고정은 최저점을 바닥에 유지합니다. 물리 재생을 다시 시작하면 이 초기 위치와 방향에서 출발합니다.</p>
    </div>}
    {SHOW_LEGACY_OBJECT_GROUPS && <><div className="section-divider"/>
    <div className="panel-heading"><span>OBJECT GROUPS</span><small>{props.groups.length}개</small></div>
    <p className="hint">상자와 내부 물체를 하나의 그룹으로 묶으면 중심 기준으로 함께 이동·회전합니다.</p>
    <input aria-label="오브젝트 그룹 이름" placeholder="그룹 이름 (선택)" value={groupName} disabled={props.disabled} onChange={event => setGroupName(event.target.value)}/>
    <div className="object-group-members">{props.objects.map(object => <label className="checkbox" key={object.id}><input type="checkbox" checked={groupMembers.includes(object.id)} disabled={props.disabled || grouped.has(object.id)} onChange={event => setGroupMembers(values => event.target.checked ? [...values, object.id] : values.filter(id => id !== object.id))}/>{object.name}{grouped.has(object.id) ? ' · 그룹됨' : ''}</label>)}</div>
    <button className="wide" disabled={props.disabled || groupMembers.length < 2} onClick={() => { props.onCreateGroup(groupMembers, groupName); setGroupMembers([]); setGroupName(''); }}><Boxes size={14}/> 선택 물체 그룹 만들기</button>
    {!!props.groups.length && <select aria-label="오브젝트 그룹 선택" value={props.selectedGroupId ?? ''} disabled={props.disabled} onChange={event => props.onSelectGroup(event.target.value || null)}><option value="">그룹 선택</option>{props.groups.map(group => <option key={group.id} value={group.id}>{group.name} · {group.member_ids.length}개</option>)}</select>}
    {selectedGroup && <div className="object-editor">
      <div className="object-title-row"><input aria-label="그룹 이름" value={selectedGroup.name} maxLength={80} disabled={props.disabled} onChange={event => props.onChangeGroup(selectedGroup.id, { name: event.target.value || selectedGroup.name })}/><button title="그룹 해제" disabled={props.disabled} onClick={() => props.onRemoveGroup(selectedGroup.id)}><Trash2 size={14}/></button></div>
      <div className="inspector-label">그룹 중심 · 월드 XYZ <small>m</small></div>
      <div className="xyz">{selectedGroup.position.map((value, index) => <label key={index}><span>{'XYZ'[index]}</span><input aria-label={`그룹 위치 ${'XYZ'[index]}`} type="number" step=".01" value={Number(value.toFixed(4))} disabled={props.disabled} onChange={event => { const position = selectedGroup.position.map((current, i) => i === index ? +event.target.value : current); if (position.every(Number.isFinite)) props.onChangeGroup(selectedGroup.id, { position }); }}/></label>)}</div>
      <div className="inspector-label">그룹 회전 · 월드 XYZ <small>°</small></div>
      <div className="xyz">{eulerDegrees(selectedGroup.quaternion_xyzw).map((value, index, rotation) => <label key={index}><span>{'XYZ'[index]}</span><input aria-label={`그룹 회전 ${'XYZ'[index]}`} type="number" step="1" value={Number(value.toFixed(2))} disabled={props.disabled} onChange={event => { const next = rotation.map((current, i) => i === index ? +event.target.value : current); if (next.every(Number.isFinite)) props.onChangeGroup(selectedGroup.id, { quaternion_xyzw: quaternionFromDegrees(next) }); }}/></label>)}</div>
      <p className="hint">그룹을 회전하면 각 물체의 위치와 방향이 중심을 기준으로 같이 바뀌며, 해제해도 결과 트랜스폼은 유지됩니다.</p>
    </div>}</>}
  </>;
}

function objectsLabel(count: number) {
  return count ? `${count}개 · 최대 32개` : '최대 32개';
}
