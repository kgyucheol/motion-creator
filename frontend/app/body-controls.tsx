import { ChevronDown, ChevronRight, LockKeyhole, Rotate3D } from 'lucide-react';
import { BODY_GROUPS, nodeMembers, selectionState, type BodyNode } from '../lib/body-groups';
import type { PoseState } from '../lib/robot-scene';

type Props = {
  pose: PoseState | null; selected: string[]; pins: string[]; anglePins: string[]; expanded: string[]; disabled: boolean;
  onExpand: (id: string) => void; onSelect: (members: string[], additive: boolean) => void;
  onPin: (key: string) => void; onAnglePin: (key: string) => void;
};
export default function BodyControls(props: Props) {
  function row(node: BodyNode) {
    const group = !!node.children;
    const open = props.expanded.includes(node.id);
    const status = selectionState(node, props.selected);
    const joint = node.handle ? props.pose?.hinges[node.handle] : undefined;
    return <li key={node.id}>
      <div className={`body-tree-row ${status.all ? 'selected' : status.partial ? 'partial' : ''}`}>
        {group ? <button className="tree-toggle" aria-label={`${node.label} ${open ? '접기' : '펼치기'}`} aria-expanded={open} aria-controls={`body-${node.id}`} onClick={() => props.onExpand(node.id)}>{open ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}</button> : <span className="tree-spacer"/>}
        <button className="tree-name" title={node.handle ?? `${node.label} 전체 선택`} disabled={props.disabled} aria-pressed={status.all} onClick={event => props.onSelect(nodeMembers(node), event.shiftKey)}>
          <span className={`part-dot ${status.all ? 'tree-selected-dot' : ''} ${node.handle && props.pins.includes(node.handle) ? 'locked' : ''}`}/>
          <span>{node.label}</span><small>{group ? `${status.count}/${status.total}` : joint ? `${(joint.angle*180/Math.PI).toFixed(1)}°` : ''}</small>
        </button>
        {node.handle && <button className={`pin-button ${props.pins.includes(node.handle) ? 'is-pinned' : ''}`} disabled={props.disabled} onClick={() => props.onPin(node.handle!)} aria-label={`${props.pose?.handles[node.handle]?.label ?? node.label} ${props.pins.includes(node.handle) ? '고정 해제' : '고정'}`} title="공간상 위치 고정"><LockKeyhole size={12}/></button>}
        {node.handle && <button className={`pin-button angle-pin-button ${props.anglePins.includes(node.handle) ? 'is-angle-pinned' : ''}`} disabled={props.disabled} onClick={() => props.onAnglePin(node.handle!)} aria-label={`${props.pose?.handles[node.handle]?.label ?? node.label} ${props.anglePins.includes(node.handle) ? '각도 고정 해제' : '각도 고정'}`} title="방향 또는 실제 관절각만 고정"><Rotate3D size={12}/></button>}
      </div>
      {group && <ul id={`body-${node.id}`} hidden={!open}>{node.children!.map(row)}</ul>}
    </li>;
  }
  return <ul className="body-tree" aria-label="신체 부위와 관절">{BODY_GROUPS.map(row)}</ul>;
}
