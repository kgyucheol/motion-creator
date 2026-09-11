export type BodyNode = { id: string; label: string; handle?: string; control?: string; children?: BodyNode[] };
const leaf = (handle: string, label: string): BodyNode => ({ id: handle, label, handle });
const axes = (prefix: string, names: string[]) => names.map(axis => leaf(`${prefix}_${axis}_joint`, axis));
const sides = [['left', '왼'], ['right', '오른']] as const;
export const BODY_GROUPS: BodyNode[] = [
  leaf('pelvis', '골반 · 루트'),
  { id: 'shoulders', label: '어깨', children: sides.map(([side, label]) => ({ id: `${side}_shoulders`, label: `${label} 어깨`, children: axes(`${side}_shoulder`, ['pitch', 'roll', 'yaw']) })) },
  { id: 'elbows', label: '팔꿈치', children: sides.map(([side, label]) => leaf(`${side}_elbow_joint`, `${label} 팔꿈치`)) },
  { id: 'hands', label: '손 · 손목 포함', children: sides.map(([side, label]) => ({ id: `${side}_hands`, label: `${label}손`, control: `${side}_hand`, children: [...axes(`${side}_wrist`, ['roll', 'pitch', 'yaw']), leaf(`${side}_hand`, '손끝 · IK 지점')] })) },
  { id: 'waist_group', label: '허리', control: 'waist', children: axes('waist', ['roll', 'pitch', 'yaw']) },
  { id: 'legs', label: '다리', children: sides.map(([side, label]) => ({ id: `${side}_leg`, label: `${label} 다리`, children: [
    { id: `${side}_hips`, label: '고관절', control: `${side}_hip`, children: axes(`${side}_hip`, ['roll', 'pitch', 'yaw']) }, leaf(`${side}_knee_joint`, '무릎'),
  ] })) },
  { id: 'feet', label: '발', children: sides.map(([side, label]) => ({ id: `${side}_feet`, label: `${label}발`, control: `${side}_ankle`, children: [...axes(`${side}_ankle`, ['roll', 'pitch']), leaf(`${side}_foot`, '발바닥 · IK 지점')] })) },
];
export const nodeMembers = (node: BodyNode): string[] => node.handle ? [node.handle] : (node.children ?? []).flatMap(nodeMembers);
export function allNodes(nodes = BODY_GROUPS): BodyNode[] { return nodes.flatMap(node => [node, ...allNodes(node.children ?? [])]); }
export const CONTROL_GROUPS = allNodes().filter(node => node.control);
export function groupForControl(key: string) { return CONTROL_GROUPS.find(node => node.control === key); }
export function expandVirtualControls(selected: string[]) {
  return [...new Set(selected.flatMap(key => {
    const node = groupForControl(key);
    return node && !nodeMembers(node).includes(key) ? nodeMembers(node) : [key];
  }))];
}
export function selectionState(node: BodyNode, selected: string[]) {
  const keys = nodeMembers(node); const count = keys.filter(key => selected.includes(key)).length;
  return { count, total: keys.length, all: count === keys.length, partial: count > 0 && count < keys.length };
}
export function selectMembers(current: string[], incoming: string[], additive: boolean) {
  if (!additive) return [...incoming];
  return incoming.every(key => current.includes(key)) ? current.filter(key => !incoming.includes(key)) : [...new Set([...current, ...incoming])];
}
// Selection stays at leaf granularity. Full anatomical groups retain their established IK controls.
export function controlSelection(selected: string[]) {
  return [...new Set(selected.map(key => CONTROL_GROUPS.find(node => nodeMembers(node).includes(key) && nodeMembers(node).every(member => selected.includes(member)))?.control ?? key))];
}
export function controlKey(selected: string[], active: string) {
  return CONTROL_GROUPS.find(node => nodeMembers(node).includes(active) && nodeMembers(node).every(member => selected.includes(member)))?.control ?? active;
}
export function visibleTreeHandles(expanded: string[], nodes = BODY_GROUPS): string[] {
  return nodes.flatMap(node => node.handle ? [node.handle] : !expanded.includes(node.id) && node.control ? [node.control] : visibleTreeHandles(expanded, node.children ?? []));
}
