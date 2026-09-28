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
};

export function duplicateKeyframeAfter(keyframes: Keyframe[], selectedIndex: number) {
  if (selectedIndex < 0 || selectedIndex >= keyframes.length) return { keyframes, index: selectedIndex };
  const index = selectedIndex + 1;
  const duplicate = structuredClone(keyframes[selectedIndex]);
  return { keyframes: [...keyframes.slice(0, index), duplicate, ...keyframes.slice(index)], index };
}
