// The editor's ramen grasp points (candidateRamenGraspPoints) projected onto the lab's XZ section.
import * as THREE from 'three';
import { candidateRamenGraspPoints, DEFAULT_RAMEN_GRASP_COEFFICIENTS, savedRamenGraspCoefficients,
  type RamenGraspCoefficients, type SceneObject } from './scene-objects.ts';

export const EDITOR_GRASP_KEY = 'motioncreator-editor-ramen-grasp-v1';

/** A grasp point in the section: TCP position, tool angle (+X toward +Z) and which blade face is up. */
export type SectionGrasp = { x: number; z: number; angle: number; flip: boolean; lateral: number; inPlane: number };

/** The shared coefficients the editor saves in this browser, or its defaults. */
export function editorGraspCoefficients(storage: Pick<Storage, 'getItem'> | null = globalThis.localStorage ?? null): RamenGraspCoefficients {
  try {
    return savedRamenGraspCoefficients(JSON.parse(storage?.getItem(EDITOR_GRASP_KEY) ?? 'null')) ?? DEFAULT_RAMEN_GRASP_COEFFICIENTS;
  } catch {
    return DEFAULT_RAMEN_GRASP_COEFFICIENTS;
  }
}

/**
 * Run the editor's grasp calculation for a cylinder lying across the robot (axis along world Y) and
 * project both hands onto the XZ plane. The axis offset (`lateral`) drops out of the section; the
 * tool's forward axis gives the angle and its top-face normal (+Z) decides the blade face side.
 */
export function sectionGrasps(target: { center: number[]; radius: number }, planeY: number, robotPosition: number[],
  robotQuaternion: number[], coefficients: RamenGraspCoefficients): Record<'left' | 'right', SectionGrasp> {
  const object = { id: 'section-target', name: 'section target', shape: 'cylinder', position: [target.center[0], planeY, target.center[1]],
    quaternion_xyzw: [0, 0, 0, 1], size: [2 * target.radius, 1, 2 * target.radius] } as unknown as SceneObject;
  const points = candidateRamenGraspPoints(object, robotPosition, robotQuaternion, coefficients);
  const project = (side: 'left' | 'right'): SectionGrasp => {
    const rotation = new THREE.Quaternion().fromArray(points.orientation[side]);
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(rotation);
    const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
    const position = points[side];
    return { x: position[0], z: position[2], angle: Math.atan2(forward.z, forward.x),
      // The blade's top face is on the left of travel when the normal points to +90 deg of the forward axis.
      flip: normal.x * -forward.z + normal.z * forward.x < 0,
      lateral: position[1] - planeY, inPlane: Math.hypot(forward.x, forward.z) };
  };
  return { left: project('left'), right: project('right') };
}
