export type TwoHandGrasp = {
  format: 'motioncreator.two-hand-grasp.v1';
  object_id: string;
  left_surface_uv: number[];
  right_surface_uv: number[];
  inward_offset_m: number;
  closure_seconds: number;
  target_force_n: number;
  max_force_n: number;
  contact_anchor?: 'lower_palm_wrist';
  hand_twist_deg?: number;
  contact_points_world?: Record<'left' | 'right', number[]>;
  object_signature?: string;
  closure_qpos?: number[];
};

export type Keyframe = {
  name: string;
  duration: number;
  qpos: number[];
  pins: string[];
  samples?: number[][];
  grasp?: TwoHandGrasp;
};

export function duplicateKeyframeAfter(keyframes: Keyframe[], selectedIndex: number) {
  if (selectedIndex < 0 || selectedIndex >= keyframes.length) return { keyframes, index: selectedIndex };
  const index = selectedIndex + 1;
  const duplicate = structuredClone(keyframes[selectedIndex]);
  return { keyframes: [...keyframes.slice(0, index), duplicate, ...keyframes.slice(index)], index };
}
