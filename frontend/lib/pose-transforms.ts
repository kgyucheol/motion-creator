import { Quaternion, Vector3, Euler, MathUtils } from 'three';

export const counterpart = (key: string) => key.startsWith('left_') ? 'right_' + key.slice(5) : key.startsWith('right_') ? 'left_' + key.slice(6) : null;
export function canMirrorSelection(keys: string[]) {
  return keys.length >= 2 && keys.every(key => { const other = counterpart(key); return other !== null && keys.includes(other); });
}
export function translatedTargets(positions: Record<string, number[]>, delta: number[], mirror?: { active: string; rootQuaternion: number[] }) {
  const offset = new Vector3().fromArray(delta);
  const normal = mirror ? new Vector3(0, 1, 0).applyQuaternion(new Quaternion().fromArray(mirror.rootQuaternion)).normalize() : null;
  const reflected = normal ? offset.clone().addScaledVector(normal, -2*offset.dot(normal)) : offset;
  const side = mirror?.active.startsWith('left_') ? 'left_' : 'right_';
  return Object.fromEntries(Object.entries(positions).map(([key, position]) => [key,
    new Vector3().fromArray(position).add(mirror && !key.startsWith(side) ? reflected : offset).toArray()]));
}

export function eulerDegrees(quaternion: number[]) {
  const e = new Euler().setFromQuaternion(new Quaternion().fromArray(quaternion), 'XYZ');
  return [e.x, e.y, e.z].map(MathUtils.radToDeg);
}

export function quaternionFromDegrees(degrees: number[]) {
  return new Quaternion().setFromEuler(new Euler(...degrees.map(MathUtils.degToRad) as [number, number, number], 'XYZ')).toArray();
}

export function rotatedGroupTargets(
  positions: Record<string, number[]>, orientations: Record<string, number[]>, center: number[],
  pivotBefore: number[], pivotAfter: number[], pins: string[],
) {
  const delta = new Quaternion().fromArray(pivotAfter).multiply(new Quaternion().fromArray(pivotBefore).invert()).normalize();
  const origin = new Vector3().fromArray(center);
  const targets: Record<string, number[]> = {};
  const rotations: Record<string, number[]> = {};
  for (const key of Object.keys(positions)) {
    if (!pins.includes(key)) targets[key] = new Vector3().fromArray(positions[key]).sub(origin).applyQuaternion(delta).add(origin).toArray();
    rotations[key] = delta.clone().multiply(new Quaternion().fromArray(orientations[key])).normalize().toArray();
  }
  return { targets, orientations: rotations };
}

export function incrementRotation(quaternion: number[], axis: number, degrees: number, space: 'world' | 'local') {
  const direction = new Vector3().setComponent(axis, 1);
  const delta = new Quaternion().setFromAxisAngle(direction, MathUtils.degToRad(degrees));
  const current = new Quaternion().fromArray(quaternion);
  return (space === 'local' ? current.multiply(delta) : delta.multiply(current)).normalize().toArray();
}
