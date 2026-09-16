export type SceneObjectShape = 'box' | 'sphere' | 'cylinder';
export type ObjectTransformMode = 'translate' | 'rotate' | 'scale';

export type SceneObject = {
  id: string;
  name: string;
  shape: SceneObjectShape;
  position: number[];
  quaternion_xyzw: number[];
  size: number[];
  mass_kg: number;
  friction: number;
  color: string;
  opacity: number;
  visible: boolean;
};

export type SceneObjectPose = { position: number[]; quaternion_xyzw: number[] };

export function normalizedObjectSize(shape: SceneObjectShape, size: number[], axis = '') {
  const safe = size.map(value => Math.max(.01, Number.isFinite(value) ? value : .01));
  if (shape === 'sphere') {
    const component = axis.includes('Y') ? safe[1] : axis.includes('Z') ? safe[2] : safe[0];
    return [component, component, component];
  }
  if (shape === 'cylinder') {
    const diameter = axis.includes('Y') ? safe[1] : safe[0];
    return [diameter, diameter, safe[2]];
  }
  return safe;
}

export function createSceneObject(index = 1, shape: SceneObjectShape = 'box'): SceneObject {
  const base = shape === 'sphere' ? [.28, .28, .28] : shape === 'cylinder' ? [.24, .24, .32] : [.3, .32, .24];
  return {
    id: `object-${Date.now().toString(36)}-${index}`,
    name: `${shape === 'box' ? '박스' : shape === 'sphere' ? '구' : '원통'} ${index}`,
    shape,
    position: [.4, 0, .3],
    quaternion_xyzw: [0, 0, 0, 1],
    size: base,
    mass_kg: 1,
    friction: .7,
    color: '#b98853',
    opacity: .62,
    visible: true,
  };
}

export function objectsFromProject(project: { scene_objects?: SceneObject[]; box?: { position: number[]; size: number[]; visible: boolean } }) {
  if (Array.isArray(project.scene_objects)) return structuredClone(project.scene_objects);
  if (project.box) {
    const object = createSceneObject(1);
    return [{ ...object, id: 'legacy-box', position: [...project.box.position], size: [...project.box.size], visible: project.box.visible }];
  }
  return [createSceneObject(1)];
}
