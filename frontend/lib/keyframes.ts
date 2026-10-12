import type { SceneObjectPose } from './scene-objects';

export type TwoHandGrasp = {
  format: 'motioncreator.two-hand-grasp.v1';
  object_id: string;
  left_surface_uv: number[];
  right_surface_uv: number[];
  hand_gap_m?: number;
  /** Legacy: hand_gap_m = box width - inward_offset_m. */
  inward_offset_m?: number;
  closure_seconds: number;
  target_force_n: number;
  max_force_n: number;
  follow_object?: boolean;
  object_pose?: SceneObjectPose;
  contact_anchor?: 'lower_palm_wrist' | 'finger_wrist_pad';
  hand_twist_deg?: number;
  contact_points_world?: Record<'left' | 'right', number[]>;
  object_signature?: string;
  closure_qpos?: number[];
};

export type ToolInteraction = {
  format: 'motioncreator.ramen-interaction.v1';
  task: 'ramen_extract';
  phase: string;
  stage_id?: string;
  stage_label?: string;
  object_id: string;
  mode: 'pose' | 'insertion' | 'hold' | 'carry';
  tcp_targets: Record<'left' | 'right', { position: number[]; quaternion_xyzw: number[] }>;
  insertion_axes_world: Record<'left' | 'right', number[]>;
  object_reference?: {
    center_world: number[];
    bundle_axis_world: number[];
    box_up_world: number[];
    side_axis_world: number[];
    bodyward_world: number[];
    diameter_m: number;
    length_m: number;
  };
  control: {
    lateral_stiffness_n_per_m: number;
    insertion_stiffness_n_per_m: number;
    translation_damping_ns_per_m: number;
    orientation_stiffness_nm_per_rad: number;
    orientation_damping_nms_per_rad: number;
    maximum_feedback_torque_fraction: number;
    force_limit_n: number;
  };
  perception?: {
    pose_source: 'scene_ground_truth' | 'rgb_pose_estimator';
    frame_id: string;
    camera: {
      link: string;
      parent_link: string;
      mount_xyz_m: number[];
      mount_rpy_rad: number[];
      world_position_m: number[];
      world_quaternion_xyzw: number[];
      optical_frame_declared: boolean;
      intrinsics_declared: boolean;
    };
    object_pose_world: { position: number[]; quaternion_xyzw: number[] };
    insertion_sites_world: Record<'left' | 'right', number[]>;
    approach_sites_world: Record<'left' | 'right', number[]>;
    approach_strategy: 'staggered_vertical_corridor';
  };
};

export type Keyframe = {
  name: string;
  duration: number;
  qpos: number[];
  pins: string[];
  angle_pins?: string[];
  samples?: number[][];
  grasp?: TwoHandGrasp;
  interaction?: ToolInteraction;
  generated_transition?: boolean;
  lab_grasp?: { group_id: string; target_id: string; tools: [string, string]; source_signature: string;
    point_index: number; authored_signature: string };
};

export function duplicateKeyframeAfter(keyframes: Keyframe[], selectedIndex: number) {
  if (selectedIndex < 0 || selectedIndex >= keyframes.length) return { keyframes, index: selectedIndex };
  const index = selectedIndex + 1;
  const duplicate = structuredClone(keyframes[selectedIndex]);
  // The new pose will be edited independently; inherited TCP targets would pin playback.
  delete duplicate.interaction;
  delete duplicate.lab_grasp;
  return { keyframes: [...keyframes.slice(0, index), duplicate, ...keyframes.slice(index)], index };
}

/** Retiming uses the existing quintic playback curve (peak slope 1.875). */
export function smoothTransitionDuration(first: Keyframe, second: Keyframe) {
  const a = first.qpos, b = second.qpos;
  if (a.length !== b.length || a.length < 7 || !a.every(Number.isFinite) || !b.every(Number.isFinite))
    throw new Error('연결할 키프레임의 관절 데이터가 올바르지 않습니다.');
  const rootDistance = Math.hypot(...a.slice(0, 3).map((value, index) => b[index] - value));
  const rootDot = Math.min(1, Math.abs(a.slice(3, 7).reduce((sum, value, index) => sum + value * b[index + 3], 0)));
  const rootAngle = 2 * Math.acos(rootDot);
  const maxJointAngle = a.slice(7).reduce((maximum, value, index) => Math.max(maximum, Math.abs(b[index + 7] - value)), 0);
  // Conservative editing speeds, not hardware torque/velocity guarantees.
  const seconds = Math.max(.8, 1.875 * rootDistance / .35, 1.875 * rootAngle / .8,
    1.875 * maxJointAngle / 1.0);
  return Math.min(60, Math.ceil(seconds * 10) / 10);
}

export function retimeSmoothTransition(keyframes: Keyframe[], firstIndex: number, secondIndex: number) {
  const start = Math.min(firstIndex, secondIndex), end = Math.max(firstIndex, secondIndex);
  if (start < 0 || end >= keyframes.length || start === end) return keyframes;
  return keyframes.map((frame, index) => index > start && index <= end
    ? { ...frame, duration: smoothTransitionDuration(keyframes[index - 1], frame) } : frame);
}

export function insertSmoothTransitionFrames(keyframes: Keyframe[], firstIndex: number, secondIndex: number,
                                             samples: { qpos: number[] }[], fps = 30) {
  const start = Math.min(firstIndex, secondIndex), end = Math.max(firstIndex, secondIndex);
  if (start < 0 || end >= keyframes.length || start === end) return keyframes;
  if (keyframes.length + end - start > 100) throw new Error('중간 자세를 추가하면 키프레임 최대 100개를 넘습니다.');
  if (keyframes.slice(start, end + 1).some(frame => frame.samples || frame.grasp || frame.interaction))
    throw new Error('파지·도구 상호작용이 있는 구간은 자동 중간 자세를 지원하지 않습니다. 해당 구간은 직접 키프레임을 추가해 주세요.');
  const result: Keyframe[] = [];
  let cursor = 0;
  keyframes.forEach((frame, index) => {
    if (index === 0) { result.push(frame); return; }
    const count = Math.max(1, Math.round(frame.duration * fps));
    if (index > start && index <= end) {
      const halfway = Math.round(count / 2);
      const qpos = samples[cursor + halfway]?.qpos;
      if (!qpos || qpos.length !== frame.qpos.length) throw new Error('중간 자세 샘플을 생성하지 못했습니다.');
      const previous = keyframes[index - 1];
      result.push({ name: `자동 중간 자세 ${index}→${index + 1}`, duration: halfway / fps,
        qpos: [...qpos], pins: previous.pins.filter(pin => frame.pins.includes(pin)),
        angle_pins: (previous.angle_pins ?? []).filter(pin => (frame.angle_pins ?? []).includes(pin)),
        generated_transition: true });
      result.push({ ...frame, duration: (count - halfway) / fps });
    } else result.push(frame);
    cursor += count;
  });
  return result;
}

/** Index of the source sample under `time` (project seconds) inside clip `index`, kept splittable (1..n-2). */
export function clipSampleAtTime(keyframes: Keyframe[], index: number, time: number) {
  const clip = keyframes[index];
  const count = clip?.samples?.length ?? 0;
  if (count < 3) return null;
  const offset = keyframes.slice(0, index).reduce((sum, frame) => sum + frame.duration, 0);
  const progress = Math.max(0, Math.min(1, (time - offset) / clip.duration));
  return Math.max(1, Math.min(count - 2, Math.round(progress * (count - 1))));
}

/**
 * Split an imported motion clip into two clips at `sample`. Both clips keep the shared boundary sample so playback
 * is unchanged, and the clip's play time is divided in proportion to the source intervals each part covers.
 */
export function splitMotionClip(keyframes: Keyframe[], index: number, sample: number) {
  const clip = keyframes[index];
  const samples = clip?.samples;
  if (!samples) throw new Error('가져온 모션 클립만 분할할 수 있습니다.');
  if (!Number.isInteger(sample) || sample < 1 || sample > samples.length - 2)
    throw new Error(`분할 프레임은 1~${samples.length - 2} 사이여야 합니다.`);
  if (keyframes.length >= 100) throw new Error('키프레임은 최대 100개입니다.');
  const intervals = samples.length - 1;
  const parts = [samples.slice(0, sample + 1), samples.slice(sample)];
  const durations = [clip.duration * sample / intervals, clip.duration * (intervals - sample) / intervals];
  if (durations.some(value => value < 1 / 120)) throw new Error('분할한 구간의 재생 시간이 너무 짧습니다.');
  const halves: Keyframe[] = parts.map((part, side) => ({
    name: `${clip.name} ${side + 1}`, duration: durations[side], qpos: [...part[0]], pins: [], samples: part,
  }));
  return { keyframes: [...keyframes.slice(0, index), ...halves, ...keyframes.slice(index + 1)], index };
}

/** Merge clip `index` with the next clip. A shared boundary sample (as left by splitMotionClip) is kept only once. */
export function mergeMotionClips(keyframes: Keyframe[], index: number) {
  const first = keyframes[index], second = keyframes[index + 1];
  if (!first?.samples || !second?.samples) throw new Error('합칠 다음 클립이 없습니다.');
  const shared = first.samples.at(-1)!.length === second.samples[0].length
    && first.samples.at(-1)!.every((value, i) => value === second.samples![0][i]);
  const merged: Keyframe = { ...first, duration: first.duration + second.duration, pins: [],
    samples: [...first.samples, ...(shared ? second.samples.slice(1) : second.samples)] };
  return { keyframes: [...keyframes.slice(0, index), merged, ...keyframes.slice(index + 2)], index };
}
