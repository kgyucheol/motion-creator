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

export type DynamicWindow = { ramp_in_start: number; hold_start: number; hold_end: number; ramp_out_end: number };
export const DYNAMIC_WINDOW_KEYS = ['ramp_in_start', 'hold_start', 'hold_end', 'ramp_out_end'] as const;

/**
 * Clip only: the hands of this clip follow a scene object that has been moved away from `reference_pose`.
 * The window numbers are clip sample numbers (they may lie outside the clip after a split, which keeps the weights).
 */
export type DynamicGrasp = {
  /** The scene object the hands follow, or null to move the grip by `offset` alone. */
  object_id: string | null;
  /** Where the object was when the motion was recorded (or, without an object, the point the yaw offset turns about). */
  reference_pose: { position: number[]; quaternion_xyzw: number[] };
  hands: ('left' | 'right')[];
  window: DynamicWindow;
  /** Extra world-frame shift (m) and turn about the reference position (deg) of the grip, on top of the object's move. */
  offset?: { position: number[]; yaw_deg: number };
  freedom?: 'yaw_xyz' | 'full6d';
  waist?: 'free' | 'lock';
  max_shift_m?: number;
};

export type DynamicReport = {
  index: number; status: 'ok' | 'approx' | 'unreachable' | 'too_far' | 'missing_object' | 'unchanged';
  object_id: string | null; shift_mm?: number; max_position_error_mm?: number; max_angle_error_deg?: number; max_shift_mm?: number;
};

/** How a card moves relative to the card before it (see motioncreator/card_motion.py). */
export type CardMotion = {
  mode: 'recorded' | 'delta' | 'cartesian';
  /** 앞 카드 이동 연결(delta)이 지키는 변화량: 관절 변화량(기본) 또는 손(TCP) 위치 변화량(포즈 카드만). */
  basis?: 'joint' | 'position';
  target?: { kind: 'object'; object_id: string; reference_pose: { position: number[]; quaternion_xyzw: number[] };
    /** 기준 물체: 원본 자세를 만들 때 쓴 물체. 손은 이 물체의 끝에서 놓인 자리를 새 물체의 같은 쪽 끝에 유지한다. 없으면 기준 위치의 추적 물체. */
    ends?: { reference: { shape: 'box' | 'open_box' | 'cylinder' | 'sphere'; size: number[]; position: number[]; quaternion_xyzw: number[] };
      /** 양끝 파지: 지정하면 녹화된 거리 대신, 각 손의 TCP를 물체 끝에서 안쪽·옆(+는 로봇 쪽)·위로 이만큼(m) 떨어진 점에 둔다. */
      tcp?: { inward_m: number; lateral_m: number; up_m: number } } } | { kind: 'point'; point: number[] };
  hands?: ('left' | 'right')[];
  offset?: { position: number[]; yaw_deg: number };
  freedom?: 'yaw_xyz' | 'full6d';
  waist?: 'free' | 'lock';
  max_shift_m?: number;
};

export type MotionReport = {
  index: number; mode: CardMotion['mode']; status: 'ok' | 'approx' | 'unreachable' | 'too_far' | 'missing_object' | 'unchanged' | 'limit' | 'partial';
  /** 'partial': the cube is beyond reach, the hands went this far toward it (a fraction of the requested move). */
  reached_fraction?: number; reached_shift_mm?: number;
  /** 손(TCP)별 월드 좌표 오차 mm = 도달한 위치 − 목표 위치 (x 앞, y 왼쪽, z 위). */
  hand_errors_mm?: Partial<Record<'left_hand' | 'right_hand', number[]>>;
  /** 위치 변화량으로 이어받은 카드. */
  basis?: 'position';
  shift_mm?: number; max_position_error_mm?: number; max_shift_mm?: number; offset_deg?: number; clamped_deg?: number; after_unreachable?: number;
  /** Set when a joint limit or an unreachable card could be avoided by moving less in the same direction. */
  caused_by?: number; blocks?: number[]; safe_fraction?: number; safe_shift_mm?: number[]; requested_shift_mm?: number[];
};

export type Keyframe = {
  name: string;
  duration: number;
  qpos: number[];
  pins: string[];
  angle_pins?: string[];
  samples?: number[][];
  /** Clip only: seconds to travel from the previous pose into this clip's first sample (automatic when unset). */
  transition?: number;
  /** Clip only: follow a moved scene object (the older window-based form, see DynamicGrasp). */
  dynamic?: DynamicGrasp;
  motion?: CardMotion;
  /** Pose keyframes: how the hands travel from the previous card to this one (joint blend by default; straight line or arc). */
  path?: { kind: 'joint' | 'linear' | 'circular'; via_offset?: number[] };
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

/** Default cut: the middle sample of a clip (kept splittable, 1..n-2). */
export function middleSample(count: number) {
  return Math.max(1, Math.min(Math.max(1, count - 2), Math.floor((count - 1) / 2)));
}

/**
 * Source sample under `time` (project seconds) inside clip `index`, or null when the time lies outside that clip.
 * `start` is the clip's own start time when the compiled timeline knows it; otherwise it is summed from durations.
 */
export function clipSampleAtTime(keyframes: Keyframe[], index: number, time: number, start?: number) {
  const clip = keyframes[index];
  const count = clip?.samples?.length ?? 0;
  if (count < 3) return null;
  const offset = start ?? keyframes.slice(0, index).reduce((sum, frame) => sum + frame.duration, 0);
  const progress = (time - offset) / clip.duration;
  if (progress < 0 || progress > 1) return null;
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
  const halves: Keyframe[] = parts.map((part, side) => {
    const dynamic = shiftDynamic(clip.dynamic, side === 0 ? 0 : -sample, part.length);
    // 델타는 양쪽이 그대로 이어받고, 목표(카테시안)를 향하는 끝은 뒤쪽 조각만 갖는다. 앞쪽 조각은 앞 카드를 따라 이어서 움직인다.
    const motion = !clip.motion ? undefined : clip.motion.mode === 'cartesian' && side === 0 ? { mode: 'delta' as const } : structuredClone(clip.motion);
    return { name: `${clip.name} ${side + 1}`, duration: durations[side], qpos: [...part[0]], pins: [], samples: part,
      ...(dynamic ? { dynamic } : {}), ...(motion ? { motion } : {}) };
  });
  return { keyframes: [...keyframes.slice(0, index), ...halves, ...keyframes.slice(index + 1)], index };
}

/** The binding as seen by a part of a clip that starts `-offset` samples later (offset <= 0 for a later part). Undefined when the part is not touched. */
export function shiftDynamic(dynamic: DynamicGrasp | undefined, offset: number, count: number): DynamicGrasp | undefined {
  if (!dynamic) return undefined;
  const window = Object.fromEntries(DYNAMIC_WINDOW_KEYS.map(key => [key, dynamic.window[key] + offset])) as DynamicWindow;
  if (window.ramp_out_end < 0 || window.ramp_in_start > count - 1) return undefined;
  return { ...structuredClone(dynamic), window };
}

/** Binding of two merged clips; `offset` is where the second clip starts inside the merged one. */
function mergeDynamic(first: Keyframe, second: Keyframe, offset: number): DynamicGrasp | undefined {
  const a = first.dynamic, b = second.dynamic;
  if (!a && !b) return undefined;
  if (a && !b) {
    if (a.window.ramp_out_end > first.samples!.length - 1) throw new Error('다음 클립까지 이어지는 동적 파지 구간이라 다른 설정 없이는 합칠 수 없습니다.');
    return a;
  }
  if (!a && b) {
    if (b.window.ramp_in_start < 0) throw new Error('앞 클립까지 이어지는 동적 파지 구간이라 다른 설정 없이는 합칠 수 없습니다.');
    return shiftDynamic(b, offset, first.samples!.length + second.samples!.length);
  }
  const same = a!.object_id === b!.object_id && JSON.stringify(a!.reference_pose) === JSON.stringify(b!.reference_pose)
    && JSON.stringify(a!.hands) === JSON.stringify(b!.hands) && a!.freedom === b!.freedom && a!.waist === b!.waist
    && a!.max_shift_m === b!.max_shift_m && JSON.stringify(a!.offset) === JSON.stringify(b!.offset) && DYNAMIC_WINDOW_KEYS.every(key => b!.window[key] + offset === a!.window[key]);
  if (!same) throw new Error('두 클립의 동적 파지 설정이 달라 합칠 수 없습니다. 한쪽을 해제하거나 같은 설정으로 맞춰 주세요.');
  return a;
}

/** Sample of clip `index` under project time `time`, 0..n-1 (the whole clip, unlike clipSampleAtTime which keeps off the ends). */
export function clipSampleIndexAtTime(keyframes: Keyframe[], index: number, time: number, start?: number) {
  const clip = keyframes[index];
  const count = clip?.samples?.length ?? 0;
  if (count < 2) return null;
  const offset = start ?? keyframes.slice(0, index).reduce((sum, frame) => sum + frame.duration, 0);
  const progress = (time - offset) / clip.duration;
  if (progress < -1e-9 || progress > 1 + 1e-9) return null;
  return Math.max(0, Math.min(count - 1, Math.round(progress * (count - 1))));
}

/** A first window around sample `at`: holds for about a tenth of the clip, with a tenth either side to ramp. */
export function defaultDynamicWindow(count: number, at: number): DynamicWindow {
  const span = Math.max(2, Math.round(count * .1));
  const hold_start = Math.max(0, Math.min(count - 1, at));
  const hold_end = Math.min(count - 1, hold_start + span);
  return { ramp_in_start: Math.max(0, hold_start - span), hold_start, hold_end, ramp_out_end: Math.min(count - 1, hold_end + span) };
}

/** Change one window number and keep the four in order (earlier ones follow down, later ones follow up). */
export function setDynamicWindowValue(window: DynamicWindow, key: keyof DynamicWindow, value: number): DynamicWindow {
  const index = DYNAMIC_WINDOW_KEYS.indexOf(key);
  const next = { ...window, [key]: Math.round(value) };
  DYNAMIC_WINDOW_KEYS.forEach((other, position) => {
    if (position < index) next[other] = Math.min(next[other], next[key]);
    if (position > index) next[other] = Math.max(next[other], next[key]);
  });
  if (next.ramp_out_end <= next.ramp_in_start) next.ramp_out_end = next.ramp_in_start + 1;
  return next;
}

/** Merge clip `index` with the next clip. A shared boundary sample (as left by splitMotionClip) is kept only once. */
export function mergeMotionClips(keyframes: Keyframe[], index: number) {
  const first = keyframes[index], second = keyframes[index + 1];
  if (!first?.samples || !second?.samples) throw new Error('합칠 다음 클립이 없습니다.');
  const shared = first.samples.at(-1)!.length === second.samples[0].length
    && first.samples.at(-1)!.every((value, i) => value === second.samples![0][i]);
  const dynamic = mergeDynamic(first, second, shared ? first.samples.length - 1 : first.samples.length);
  const { dynamic: _dropped, ...rest } = first;
  // 합친 클립의 끝은 뒤 클립의 끝이므로 뒤가 목표(카테시안)면 그것을, 아니면 앞의 방식을 따른다.
  const motion = second.motion?.mode === 'cartesian' ? second.motion : first.motion ?? second.motion;
  const merged: Keyframe = { ...rest, duration: first.duration + second.duration, pins: [],
    samples: [...first.samples, ...(shared ? second.samples.slice(1) : second.samples)], ...(dynamic ? { dynamic } : {}), ...(motion ? { motion: structuredClone(motion) } : {}) };
  if (!motion) delete merged.motion;
  return { keyframes: [...keyframes.slice(0, index), merged, ...keyframes.slice(index + 2)], index };
}

/**
 * Replace clip `index` with `count` editable pose keyframes taken at evenly spaced samples (first and last included).
 * Travel times add up to the clip's play time, so an unedited result takes as long as the clip did. The first pose
 * continues from the previous item: a shared boundary needs only the minimum time, otherwise the usual smooth time.
 */
export function convertClipToPoses(keyframes: Keyframe[], index: number, count: number, pins: string[]) {
  const clip = keyframes[index];
  const samples = clip?.samples;
  if (!samples) throw new Error('가져온 모션 클립만 키프레임으로 변환할 수 있습니다.');
  if (!Number.isInteger(count) || count < 2 || count > samples.length)
    throw new Error(`키프레임 수는 2~${samples.length} 사이여야 합니다.`);
  if (keyframes.length - 1 + count > 100) throw new Error('키프레임은 최대 100개입니다.');
  const picks = Array.from({ length: count }, (_, i) => Math.round(i * (samples.length - 1) / (count - 1)));
  const previous = keyframes[index - 1];
  const previousEnd = previous ? (previous.samples ? previous.samples.at(-1)! : previous.qpos) : undefined;
  const joined = !!previousEnd && previousEnd.every((value, i) => value === samples[0][i]);
  const travel = Math.min(60, Math.max(.1, clip.duration / (count - 1)));
  const poses: Keyframe[] = picks.map((pick, i) => ({
    name: i === 0 ? `${clip.name} 시작` : i === count - 1 ? `${clip.name} 끝` : `${clip.name} ${i + 1}`,
    duration: i > 0 ? travel : !previousEnd || joined ? .1 : clip.transition ?? smoothTransitionDuration({ qpos: previousEnd } as Keyframe, { qpos: samples[0] } as Keyframe),
    qpos: [...samples[pick]], pins: [...pins], angle_pins: [],
  }));
  return { keyframes: [...keyframes.slice(0, index), ...poses, ...keyframes.slice(index + 1)], index };
}

/*
 * Timing model shown in the editor: a card's time is the travel time from THIS card to the NEXT one.
 * The project file keeps the older "arrival" form (a pose's `duration`, or a clip's `transition`, is the time to
 * reach that item from its predecessor), so saved projects need no conversion: card i simply shows and edits the
 * arrival time stored on item i + 1. A clip's own `duration` is its playback length and is not a travel time.
 */
const endPose = (frame: Keyframe) => frame.samples ? frame.samples[frame.samples.length - 1] : frame.qpos;
const startPose = (frame: Keyframe) => frame.samples ? frame.samples[0] : frame.qpos;
const samePose = (a: number[], b: number[]) => a.length === b.length && a.every((value, i) => Math.abs(value - b[i]) <= 1e-6);

export type Outgoing = { kind: 'none' | 'joined' | 'pose' | 'clip'; seconds?: number };

/** What card `index` shows as "time to the next card". */
export function outgoingSeconds(keyframes: Keyframe[], index: number): Outgoing {
  const frame = keyframes[index], next = keyframes[index + 1];
  if (!frame || !next) return { kind: 'none' };
  if (next.samples) return frame && samePose(endPose(frame), startPose(next)) ? { kind: 'joined' } : { kind: 'clip', seconds: next.transition };
  return { kind: 'pose', seconds: next.duration };
}

/** Set card `index`'s time to the next card; `undefined` restores the automatic time for a clip entry. */
export function setOutgoingSeconds(keyframes: Keyframe[], index: number, seconds?: number) {
  const next = keyframes[index + 1];
  if (!next) return keyframes;
  const value = seconds === undefined ? undefined : Math.min(60, Math.max(.1, seconds));
  return keyframes.map((frame, i) => {
    if (i !== index + 1) return frame;
    if (frame.samples) { const copy = { ...frame }; if (value === undefined) delete copy.transition; else copy.transition = value; return copy; }
    return value === undefined ? frame : { ...frame, duration: value };
  });
}

/** Each card's own time to its next card, as an array parallel to `keyframes` (the last card has none). */
function takeOutgoing(keyframes: Keyframe[]) {
  return keyframes.map((_, i) => { const out = outgoingSeconds(keyframes, i); return out.kind === 'pose' || out.kind === 'clip' ? out.seconds : undefined; });
}

/** Re-apply per-card outgoing times to a new card order; a missing value keeps the item's current arrival time. */
function applyOutgoing(keyframes: Keyframe[], outgoing: (number | undefined)[]) {
  return keyframes.map((frame, i) => {
    if (i === 0) return frame;
    const seconds = outgoing[i - 1];
    if (frame.samples) { const copy = { ...frame }; if (seconds === undefined) delete copy.transition; else copy.transition = seconds; return copy; }
    return seconds === undefined ? frame : { ...frame, duration: seconds };
  });
}

/** Swap card `index` with its neighbour (delta ±1). Each card keeps its own time to its next card. */
export function moveKeyframe(keyframes: Keyframe[], index: number, delta: -1 | 1) {
  const target = index + delta;
  if (index < 0 || target < 0 || target >= keyframes.length) return keyframes;
  const out = takeOutgoing(keyframes);
  const frames = [...keyframes], times = [...out];
  [frames[index], frames[target]] = [frames[target], frames[index]];
  [times[index], times[target]] = [times[target], times[index]];
  return applyOutgoing(frames, times);
}

/** Remove card `index`; the previous card keeps its own time to whatever now follows it. */
/**
 * Merge card `index` into the keyframe after it: the pose of `index` disappears, the later keyframe (with its name and motion mode)
 * stays, and the time from the card before to it becomes the sum of the two times, so the motion keeps its total length.
 * A clip at `index` counts with its playing time. The next card must be a pose keyframe.
 */
export function mergeIntoNextPose(keyframes: Keyframe[], index: number) {
  const frame = keyframes[index], next = keyframes[index + 1];
  if (!frame || !next) throw new Error('합칠 다음 카드가 없습니다.');
  if (next.samples) throw new Error('다음 카드가 클립이면 키프레임으로 합칠 수 없습니다. 클립 쪽에서 "다음과 합치기"를 쓰세요.');
  const out = takeOutgoing(keyframes);
  if (index > 0) {
    const through = (out[index] ?? next.duration) + (frame.samples ? frame.duration : 0);
    out[index - 1] = (out[index - 1] ?? 0) + through;
  }
  return applyOutgoing(keyframes.filter((_, i) => i !== index), out.filter((_, i) => i !== index));
}

export function removeKeyframe(keyframes: Keyframe[], index: number) {
  const out = takeOutgoing(keyframes);
  return applyOutgoing(keyframes.filter((_, i) => i !== index), out.filter((_, i) => i !== index));
}

/** Duplicate a pose after `index`; the copy repeats that card's time to its next card. */
export function duplicateKeyframeKeepingTimes(keyframes: Keyframe[], index: number) {
  const result = duplicateKeyframeAfter(keyframes, index);
  if (result.keyframes === keyframes) return result;
  const out = takeOutgoing(keyframes);
  const times = [...out.slice(0, index + 1), out[index], ...out.slice(index + 1)];
  return { ...result, keyframes: applyOutgoing(result.keyframes, times) };
}

const quantize = (seconds: number, fps: number) => Math.max(1, Math.round(seconds * fps)) / fps;

/** Start (arrival) second of every card, mirroring the server's compile so it can be shown before any preview runs. */
export function keyframeStartTimes(keyframes: Keyframe[], fps = 30) {
  const starts: number[] = [];
  let now = 0;
  let end: number[] | undefined;
  keyframes.forEach((frame, index) => {
    if (frame.samples) {
      if (end && !samePose(end, frame.samples[0]))
        now += quantize(frame.transition ?? smoothTransitionDuration({ qpos: end } as Keyframe, { qpos: frame.samples[0] } as Keyframe), fps);
      starts.push(now);
      if (frame.samples.length > 1) now += quantize(frame.duration, fps);
    } else {
      if (index > 0) now += quantize(frame.duration, fps);
      starts.push(now);
    }
    end = endPose(frame);
  });
  return starts;
}

/**
 * Insert an editable pose right after clip `index`, starting from the clip's last sample. It takes the minimum time to
 * leave the clip, and the clip's old time to whatever followed is now the pose's time to it.
 */
export function insertPoseAfterClip(keyframes: Keyframe[], index: number, pins: string[]) {
  const clip = keyframes[index];
  if (!clip?.samples) throw new Error('클립 뒤에만 이 방식으로 자세를 추가할 수 있습니다.');
  if (keyframes.length >= 100) throw new Error('키프레임은 최대 100개입니다.');
  const pose: Keyframe = { name: `${clip.name} 이후 자세`, duration: .1, qpos: [...clip.samples[clip.samples.length - 1]], pins: [...pins], angle_pins: [] };
  return { keyframes: [...keyframes.slice(0, index + 1), pose, ...keyframes.slice(index + 1)], index: index + 1 };
}
