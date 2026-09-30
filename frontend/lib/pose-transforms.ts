import { Quaternion, Vector3, Euler, MathUtils } from 'three';

export const counterpart = (key: string) => key.startsWith('left_') ? 'right_' + key.slice(5) : key.startsWith('right_') ? 'left_' + key.slice(6) : null;
export function canMirrorSelection(keys: string[]) {
  return keys.length >= 2 && keys.every(key => { const other = counterpart(key); return other !== null && keys.includes(other); });
}
export function isJointPairSelection(keys: string[]) {
  return keys.length === 2 && keys.every(key => key.endsWith('_joint')) && counterpart(keys[0]) === keys[1];
}
/** Arms mirror across the torso's left/right plane (the waist can be turned away from the pelvis); legs use the pelvis. */
export function mirrorFrameHandle(keys: string[]) {
  return keys.length > 0 && keys.every(key => /shoulder|elbow|wrist|hand/.test(key)) ? 'waist' : 'pelvis';
}
export function mirroredJointSign(axis: number[], otherAxis: number[], rootQuaternion: number[]) {
  const normal = new Vector3(0, 1, 0).applyQuaternion(new Quaternion().fromArray(rootQuaternion)).normalize();
  const reflectedAxial = new Vector3().fromArray(axis).addScaledVector(normal, -2 * new Vector3().fromArray(axis).dot(normal)).negate();
  return reflectedAxial.dot(new Vector3().fromArray(otherAxis)) < 0 ? -1 : 1;
}
type JointAnchor = { joint_name: string; angle: number; limits: number[]; axis_world: number[] };
export function pairedJointTargets(active: JointAnchor, other: JointAnchor, requestedAngle: number, mirror: boolean, rootQuaternion: number[]) {
  const clamp = (value: number, limits: number[]) => Math.max(limits[0], Math.min(limits[1], value));
  const angle = clamp(requestedAngle, active.limits);
  const sign = mirror ? mirroredJointSign(active.axis_world, other.axis_world, rootQuaternion) : 1;
  return { [active.joint_name]: angle, [other.joint_name]: clamp(other.angle + sign * (angle - active.angle), other.limits) };
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
