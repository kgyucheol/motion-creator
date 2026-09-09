import { Quaternion, Vector3, Euler, MathUtils } from 'three';

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
