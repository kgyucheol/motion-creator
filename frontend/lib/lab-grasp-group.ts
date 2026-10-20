import type { Keyframe } from './keyframes';
import type { LabLateralOffsets, LabPathPreview } from './lab-path-apply';
import type { SceneObject } from './scene-objects';

// These signatures describe editor content, not security-sensitive data.
function signature(value: unknown) {
  const source = JSON.stringify(value);
  let hash = 2166136261;
  for (let i = 0; i < source.length; i++) hash = Math.imul(hash ^ source.charCodeAt(i), 16777619);
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export function authoredSignature(frame: Keyframe) {
  return signature({ name: frame.name, duration: frame.duration, qpos: frame.qpos, pins: frame.pins,
    angle_pins: frame.angle_pins, samples: frame.samples, grasp: frame.grasp, interaction: frame.interaction });
}

export function labSourceSignature(preview: Pick<LabPathPreview, 'sourcePaths'>, target: SceneObject, anchor: Keyframe,
  lateralOffsets: LabLateralOffsets = {}) {
  const offsets = Object.fromEntries((['left', 'right'] as const).filter(side => preview.sourcePaths[side] &&
    Math.abs(lateralOffsets[side] ?? 0) > 1e-9).map(side => [side, lateralOffsets[side]]));
  return signature({ paths: preview.sourcePaths, target: { id: target.id, position: target.position,
    quaternion_xyzw: target.quaternion_xyzw, size: target.size, collision_size: target.collision_size },
    anchor: { qpos: anchor.qpos, pins: anchor.pins, angle_pins: anchor.angle_pins },
    ...(Object.keys(offsets).length ? { lateralOffsets: offsets } : {}) });
}

export function labGroupRange(keyframes: Keyframe[], index: number) {
  const id = keyframes[index]?.lab_grasp?.group_id;
  if (!id) return null;
  let start = index, end = index;
  while (keyframes[start - 1]?.lab_grasp?.group_id === id) start--;
  while (keyframes[end + 1]?.lab_grasp?.group_id === id) end++;
  const frames = keyframes.slice(start, end + 1);
  const valid = start > 0 && frames.every((frame, offset) => frame.lab_grasp?.point_index === offset &&
    frame.lab_grasp.target_id === frames[0].lab_grasp?.target_id &&
    frame.lab_grasp.source_signature === frames[0].lab_grasp?.source_signature &&
    frame.lab_grasp.tools.join('|') === frames[0].lab_grasp?.tools.join('|')) &&
    keyframes.filter(frame => frame.lab_grasp?.group_id === id).length === frames.length;
  return { id, start, end, frames, valid,
    edited: frames.filter(frame => authoredSignature(frame) !== frame.lab_grasp?.authored_signature) };
}

export function legacyLabRange(keyframes: Keyframe[], index: number) {
  const match = /^그린 점 (\d+)(?: · 파지점)?$/.exec(keyframes[index]?.name ?? '');
  if (!match || keyframes[index]?.lab_grasp) return null;
  const number = Number(match[1]);
  const start = index - number + 1;
  if (start < 1) return null;
  let end = start;
  while (keyframes[end] && !keyframes[end].lab_grasp && keyframes[end].name ===
    (end === start ? '그린 점 1 · 파지점' : `그린 점 ${end - start + 1}`)) end++;
  return end > index ? { start, end: end - 1 } : null;
}
