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
  object_id: string;
  mode: 'pose' | 'insertion' | 'hold' | 'carry';
  tcp_targets: Record<'left' | 'right', { position: number[]; quaternion_xyzw: number[] }>;
  insertion_axes_world: Record<'left' | 'right', number[]>;
  control: {
    lateral_stiffness_n_per_m: number;
    insertion_stiffness_n_per_m: number;
    translation_damping_ns_per_m: number;
    orientation_stiffness_nm_per_rad: number;
    orientation_damping_nms_per_rad: number;
    maximum_feedback_torque_fraction: number;
    force_limit_n: number;
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
};

export function duplicateKeyframeAfter(keyframes: Keyframe[], selectedIndex: number) {
  if (selectedIndex < 0 || selectedIndex >= keyframes.length) return { keyframes, index: selectedIndex };
  const index = selectedIndex + 1;
  const duplicate = structuredClone(keyframes[selectedIndex]);
  return { keyframes: [...keyframes.slice(0, index), duplicate, ...keyframes.slice(index)], index };
}
