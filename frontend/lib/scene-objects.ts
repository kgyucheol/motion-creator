import * as THREE from 'three';

export type SceneObjectShape = 'box' | 'open_box' | 'sphere' | 'cylinder';
export type SceneCollisionShape = Exclude<SceneObjectShape, 'open_box'> | 'convex_hull';
export type ObjectTransformMode = 'translate' | 'rotate' | 'scale';
export type SceneObjectPlacement = { prevent_overlap: boolean; surface_snap: boolean; ground_lock: boolean };

export type SceneObject = {
  id: string;
  /** Editor hierarchy only; world transforms and physics bodies remain independent. */
  parent_id?: string | null;
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
  placement?: SceneObjectPlacement;
  /** Editor-only visualization; never persisted as a physical scene object. */
  ghost?: boolean;
  fixed?: boolean;
  asset_id?: string;
  asset_bounds_min?: number[];
  asset_bounds_max?: number[];
  asset_axis_transform_xyzw?: number[];
  asset_part_id?: string;
  asset_node_name?: string;
  collision_shape?: SceneCollisionShape;
  collision_size?: number[];
  collision_quaternion_xyzw?: number[];
  collision_hull_vertices?: number[][];
  collision_hull_faces?: number[][];
  wall_thickness_m?: number;
};

export type SceneObjectGroup = {
  id: string;
  name: string;
  member_ids: string[];
  position: number[];
  quaternion_xyzw: number[];
};

export type SceneAssetPart = {
  part_id: string;
  node_name: string;
  name: string;
  url: string;
  bounds_min: number[];
  bounds_max: number[];
  dimensions: number[];
  suggestion: SceneAssetSuggestion;
};

type SceneAssetSuggestion = {
  shape: SceneObjectShape;
  size: number[];
  mass_kg: number;
  friction: number;
  color: string;
  fixed: boolean;
  wall_thickness_m?: number;
  collision_shape?: SceneCollisionShape;
  collision_size?: number[];
  collision_quaternion_xyzw?: number[];
  collision_hull_vertices?: number[][];
  collision_hull_faces?: number[][];
};

export type SceneAssetImport = {
  asset_id: string;
  name: string;
  source_name: string;
  source_format: string;
  url: string;
  bounds_min: number[];
  bounds_max: number[];
  dimensions: number[];
  axis_transform_xyzw: number[];
  suggestion: SceneAssetSuggestion;
  parts?: SceneAssetPart[];
};

export type SceneObjectPose = { position: number[]; quaternion_xyzw: number[] };
export type CandidateGraspPoints = { left: number[]; right: number[] };
export type GraspPointCoefficients = { lateral: number; clearance: number; height: number };
export type RamenGraspCoefficients = { left: GraspPointCoefficients; right: GraspPointCoefficients };
export const DEFAULT_RAMEN_GRASP_COEFFICIENTS: RamenGraspCoefficients = {
  left: { lateral: .25, clearance: .10, height: 0 },
  right: { lateral: .25, clearance: .10, height: 0 },
};
export type ScenePlacementOptions = { preventOverlap: boolean; surfaceSnap: boolean; groundLock: boolean; snapDistance?: number };

const DEFAULT_PLACEMENT: SceneObjectPlacement = { prevent_overlap: true, surface_snap: false, ground_lock: true };

export function normalizedSceneAsset(template: THREE.Object3D, boundsMin: number[], boundsMax: number[], axisTransform = [0, 0, 0, 1]) {
  const minimum = new THREE.Vector3().fromArray(boundsMin);
  const maximum = new THREE.Vector3().fromArray(boundsMax);
  const dimensions = maximum.clone().sub(minimum);
  const center = minimum.add(maximum).multiplyScalar(.5);
  const normalized = new THREE.Group();
  normalized.scale.set(1 / dimensions.x, 1 / dimensions.y, 1 / dimensions.z);
  const centered = new THREE.Group();
  centered.position.copy(center).multiplyScalar(-1);
  const aligned = new THREE.Group();
  aligned.quaternion.fromArray(axisTransform).normalize();
  aligned.add(template);
  centered.add(aligned);
  normalized.add(centered);
  return normalized;
}

export function scenePlacementOptions(object?: SceneObject): ScenePlacementOptions {
  const placement = object?.placement ?? DEFAULT_PLACEMENT;
  return {
    preventOverlap: placement.prevent_overlap,
    surfaceSnap: placement.surface_snap,
    groundLock: placement.ground_lock,
  };
}

export function withScenePlacement<T extends SceneObject>(object: T, options: ScenePlacementOptions): T {
  return { ...object, placement: {
    prevent_overlap: options.preventOverlap,
    surface_snap: options.surfaceSnap,
    ground_lock: options.groundLock,
  } };
}

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

export function isRamenBundle(object: SceneObject) {
  return /mupama.stack|ramen.bundle|라면.*묶음/i.test(object.name) && object.shape !== 'open_box';
}

export function candidateRamenGraspPoints(object: SceneObject, robotPosition: number[], robotQuaternion: number[], coefficients: RamenGraspCoefficients = DEFAULT_RAMEN_GRASP_COEFFICIENTS): CandidateGraspPoints {
  const center = new THREE.Vector3().fromArray(object.position);
  const rotation = new THREE.Quaternion().fromArray(object.quaternion_xyzw).normalize();
  const axisIndex = object.size.indexOf(Math.max(...object.size));
  const axis = new THREE.Vector3().setComponent(axisIndex, 1).applyQuaternion(rotation).normalize();
  const robotLeft = new THREE.Vector3(0, 1, 0).applyQuaternion(new THREE.Quaternion().fromArray(robotQuaternion));
  if (axis.dot(robotLeft) < 0) axis.negate();
  const towardRobot = new THREE.Vector3().fromArray(robotPosition).sub(center);
  towardRobot.z = 0;
  towardRobot.addScaledVector(axis, -towardRobot.dot(axis));
  if (towardRobot.lengthSq() < 1e-8) {
    towardRobot.set(-1, 0, 0).applyQuaternion(new THREE.Quaternion().fromArray(robotQuaternion));
    towardRobot.z = 0;
    towardRobot.addScaledVector(axis, -towardRobot.dot(axis));
  }
  towardRobot.normalize();
  const radius = Math.max(...object.size.filter((_, index) => index !== axisIndex)) / 2;
  const point = (side: 'left' | 'right') => center.clone()
    .addScaledVector(axis, (side === 'left' ? 1 : -1) * coefficients[side].lateral)
    .addScaledVector(towardRobot, radius + coefficients[side].clearance)
    .addScaledVector(new THREE.Vector3(0, 0, 1), coefficients[side].height).toArray();
  return {
    left: point('left'),
    right: point('right'),
  };
}

export function objectHalfExtents(object: Pick<SceneObject, 'shape' | 'size' | 'quaternion_xyzw'>) {
  const size = normalizedObjectSize(object.shape, object.size);
  if (object.shape === 'sphere') return [size[0] / 2, size[0] / 2, size[0] / 2];
  const [x, y, z, w] = object.quaternion_xyzw;
  const rotation = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
  if (object.shape === 'cylinder') {
    const radius = size[0] / 2;
    const halfHeight = size[2] / 2;
    return rotation.map(row => radius * Math.hypot(row[0], row[1]) + halfHeight * Math.abs(row[2]));
  }
  const local = size.map(value => value / 2);
  return rotation.map(row => row.reduce((sum, value, index) => sum + Math.abs(value) * local[index], 0));
}

export function objectVerticalHalfExtent(object: Pick<SceneObject, 'shape' | 'size' | 'quaternion_xyzw'>) {
  return objectHalfExtents(object)[2];
}

export function groundedSceneObject<T extends SceneObject>(object: T): T {
  const minimumZ = objectVerticalHalfExtent(object);
  if (object.position[2] >= minimumZ) return object;
  return { ...object, position: [object.position[0], object.position[1], minimumZ] };
}

function objectBounds(object: SceneObject) {
  const half = objectHalfExtents(object);
  return { half, min: object.position.map((value, index) => value - half[index]), max: object.position.map((value, index) => value + half[index]) };
}

export function sceneObjectsOverlap(a: SceneObject, b: SceneObject, tolerance = 1e-9) {
  const first = objectBounds(a);
  const second = objectBounds(b);
  return [0, 1, 2].every(axis => Math.min(first.max[axis], second.max[axis]) - Math.max(first.min[axis], second.min[axis]) > tolerance);
}

function snapToSurface<T extends SceneObject>(object: T, others: SceneObject[], distance: number): T {
  const bounds = objectBounds(object);
  let nearest: { axis: number; shift: number; distance: number } | undefined;
  for (const other of others) {
    const target = objectBounds(other);
    if (Math.min(bounds.max[2], target.max[2]) <= Math.max(bounds.min[2], target.min[2])) continue;
    for (const axis of [0, 1]) {
      const cross = axis === 0 ? 1 : 0;
      if (Math.min(bounds.max[cross], target.max[cross]) <= Math.max(bounds.min[cross], target.min[cross])) continue;
      const gaps = [
        { distance: target.min[axis] - bounds.max[axis], shift: target.min[axis] - bounds.max[axis] },
        { distance: bounds.min[axis] - target.max[axis], shift: target.max[axis] - bounds.min[axis] },
      ];
      for (const gap of gaps) {
        if (gap.distance < 0 || gap.distance > distance || nearest && gap.distance >= nearest.distance) continue;
        nearest = { axis, shift: gap.shift, distance: gap.distance };
      }
    }
  }
  if (!nearest) return object;
  const position = [...object.position];
  position[nearest.axis] += nearest.shift;
  return { ...object, position };
}

export function placeSceneObject<T extends SceneObject>(candidate: T, others: SceneObject[], options: ScenePlacementOptions, previous?: T): T {
  let placed = options.groundLock
    ? { ...candidate, position: [candidate.position[0], candidate.position[1], objectVerticalHalfExtent(candidate)] }
    : groundedSceneObject(candidate);
  if (options.surfaceSnap) placed = snapToSurface(placed, others, options.snapDistance ?? .02);
  if (!options.preventOverlap) return placed;
  const movement = previous ? placed.position.map((value, index) => value - previous.position[index]) : [0, 0, 0];
  const axes = options.groundLock ? [0, 1] : [0, 1, 2];
  for (let iteration = 0; iteration < Math.max(1, others.length * 4); iteration++) {
    const other = others.find(value => sceneObjectsOverlap(placed, value));
    if (!other) return placed;
    const bounds = objectBounds(placed);
    const target = objectBounds(other);
    const overlaps = axes.map(axis => Math.min(bounds.max[axis], target.max[axis]) - Math.max(bounds.min[axis], target.min[axis]));
    const movingAxes = axes.filter(axis => Math.abs(movement[axis]) > 1e-9);
    const axis = movingAxes.length
      ? movingAxes.reduce((best, value) => Math.abs(movement[value]) > Math.abs(movement[best]) ? value : best)
      : axes[overlaps.indexOf(Math.min(...overlaps))];
    const direction = Math.abs(movement[axis]) > 1e-9 ? -Math.sign(movement[axis]) : placed.position[axis] < other.position[axis] ? -1 : 1;
    const position = [...placed.position];
    position[axis] += direction * overlaps[axes.indexOf(axis)];
    placed = { ...placed, position };
  }
  if (previous && !others.some(other => sceneObjectsOverlap(previous, other))) return previous;
  return placed;
}

export function createSceneObject(index = 1, shape: SceneObjectShape = 'box'): SceneObject {
  const base = shape === 'sphere' ? [.28, .28, .28] : shape === 'cylinder' ? [.24, .24, .32] : [.3, .32, .24];
  const position = [.4, 0, base[2] / 2];
  return {
    id: `object-${Date.now().toString(36)}-${index}`,
    name: `${shape === 'box' || shape === 'open_box' ? '박스' : shape === 'sphere' ? '구' : '원통'} ${index}`,
    shape,
    position,
    quaternion_xyzw: [0, 0, 0, 1],
    size: base,
    mass_kg: 1,
    friction: .7,
    color: '#b98853',
    opacity: .62,
    visible: true,
    placement: { ...DEFAULT_PLACEMENT },
  };
}

export function createImportedSceneObject(asset: SceneAssetImport, index = 1): SceneObject {
  const size = normalizedObjectSize(asset.suggestion.shape, asset.suggestion.size ?? asset.dimensions);
  return {
    id: `asset-${asset.asset_id}-${Date.now().toString(36)}-${index}`,
    name: asset.name || asset.source_name.replace(/\.[^.]+$/, ''),
    shape: asset.suggestion.shape,
    position: [.4, 0, size[2] / 2],
    quaternion_xyzw: [0, 0, 0, 1],
    size,
    mass_kg: asset.suggestion.mass_kg,
    friction: asset.suggestion.friction,
    color: asset.suggestion.color,
    opacity: 1,
    visible: true,
    fixed: asset.suggestion.fixed,
    placement: { prevent_overlap: true, surface_snap: false, ground_lock: true },
    asset_id: asset.asset_id,
    asset_bounds_min: [...asset.bounds_min],
    asset_bounds_max: [...asset.bounds_max],
    asset_axis_transform_xyzw: [...asset.axis_transform_xyzw],
    ...sceneAssetCollision(asset.suggestion),
    ...(asset.suggestion.wall_thickness_m ? { wall_thickness_m: asset.suggestion.wall_thickness_m } : {}),
  };
}

export function createImportedSceneObjects(asset: SceneAssetImport, startIndex = 1): SceneObject[] {
  if (!asset.parts?.length) return [createImportedSceneObject(asset, startIndex)];
  const origin = [.4, 0, -asset.bounds_min[2]];
  const stamp = Date.now().toString(36);
  return asset.parts.map((part, offset) => {
    const size = normalizedObjectSize(part.suggestion.shape, part.suggestion.size ?? part.dimensions);
    const center = part.bounds_min.map((value, axis) => (value + part.bounds_max[axis]) / 2);
    return {
      id: `asset-${asset.asset_id}-${stamp}-${startIndex + offset}`,
      name: part.name,
      shape: part.suggestion.shape,
      position: center.map((value, axis) => value + origin[axis]),
      quaternion_xyzw: [0, 0, 0, 1], size,
      mass_kg: part.suggestion.mass_kg, friction: part.suggestion.friction,
      color: part.suggestion.color, opacity: 1, visible: true, fixed: part.suggestion.fixed,
      placement: { prevent_overlap: false, surface_snap: false, ground_lock: false },
      asset_id: asset.asset_id, asset_part_id: part.part_id,
      asset_node_name: part.node_name,
      asset_bounds_min: [...part.bounds_min], asset_bounds_max: [...part.bounds_max],
      asset_axis_transform_xyzw: [...asset.axis_transform_xyzw],
      ...sceneAssetCollision(part.suggestion),
      ...(part.suggestion.wall_thickness_m ? { wall_thickness_m: part.suggestion.wall_thickness_m } : {}),
    };
  });
}

function sceneAssetCollision(suggestion: SceneAssetSuggestion): Partial<SceneObject> {
  if (!suggestion.collision_shape) return {};
  return {
    collision_shape: suggestion.collision_shape,
    ...(suggestion.collision_size ? { collision_size: [...suggestion.collision_size] } : {}),
    ...(suggestion.collision_quaternion_xyzw
      ? { collision_quaternion_xyzw: [...suggestion.collision_quaternion_xyzw] } : {}),
    ...(suggestion.collision_hull_vertices
      ? { collision_hull_vertices: suggestion.collision_hull_vertices.map(vertex => [...vertex]) } : {}),
    ...(suggestion.collision_hull_faces
      ? { collision_hull_faces: suggestion.collision_hull_faces.map(face => [...face]) } : {}),
  };
}

export function createSceneObjectGroup(objects: SceneObject[], memberIds: string[], index = 1): SceneObjectGroup {
  const members = objects.filter(object => memberIds.includes(object.id));
  const position = [0, 1, 2].map(axis => members.reduce((sum, object) => sum + object.position[axis], 0) / Math.max(1, members.length));
  return {
    id: `group-${Date.now().toString(36)}-${index}`,
    name: `오브젝트 그룹 ${index}`,
    member_ids: members.map(object => object.id),
    position,
    quaternion_xyzw: [0, 0, 0, 1],
  };
}

export function transformSceneObjectGroup(objects: SceneObject[], previous: SceneObjectGroup, next: SceneObjectGroup) {
  const beforePosition = new THREE.Vector3().fromArray(previous.position);
  const afterPosition = new THREE.Vector3().fromArray(next.position);
  const beforeRotation = new THREE.Quaternion().fromArray(previous.quaternion_xyzw).normalize();
  const afterRotation = new THREE.Quaternion().fromArray(next.quaternion_xyzw).normalize();
  const delta = afterRotation.clone().multiply(beforeRotation.clone().invert());
  return objects.map(object => {
    if (!previous.member_ids.includes(object.id)) return object;
    const position = new THREE.Vector3().fromArray(object.position).sub(beforePosition).applyQuaternion(delta).add(afterPosition);
    const rotation = delta.clone().multiply(new THREE.Quaternion().fromArray(object.quaternion_xyzw)).normalize();
    return { ...object, position: position.toArray(), quaternion_xyzw: rotation.toArray() };
  });
}

export function sceneObjectDescendantIds(objects: SceneObject[], parentId: string): Set<string> {
  const descendants = new Set<string>();
  let frontier = [parentId];
  while (frontier.length) {
    const children = objects.filter(object => object.parent_id && frontier.includes(object.parent_id) && !descendants.has(object.id));
    frontier = children.map(object => object.id);
    for (const child of children) descendants.add(child.id);
  }
  return descendants;
}

export function removeSceneObjectSubtree(objects: SceneObject[], id: string) {
  if (!objects.some(object => object.id === id)) return { objects, removedIds: new Set<string>() };
  const removedIds = sceneObjectDescendantIds(objects, id);
  removedIds.add(id);
  return { objects: objects.filter(object => !removedIds.has(object.id)), removedIds };
}

export function reparentSceneObject(objects: SceneObject[], childId: string, parentId: string | null) {
  const child = objects.find(object => object.id === childId);
  if (!child || (child.parent_id ?? null) === parentId) return objects;
  if (parentId && (!objects.some(object => object.id === parentId)
      || parentId === childId || sceneObjectDescendantIds(objects, childId).has(parentId))) return objects;
  return objects.map(object => object.id === childId ? { ...object, parent_id: parentId } : object);
}

export function reparentSceneObjects(objects: SceneObject[], childIds: string[], parentId: string | null) {
  const requested = new Set(childIds);
  if (!requested.size || parentId && !objects.some(object => object.id === parentId)) return objects;
  const byId = new Map(objects.map(object => [object.id, object]));
  if ([...requested].some(id => !byId.has(id) || id === parentId
      || parentId && sceneObjectDescendantIds(objects, id).has(parentId))) return objects;
  // Moving a selected parent already carries its descendants. Keep their
  // existing parent links even if they were also selected by a range click.
  const roots = [...requested].filter(id => {
    let ancestor = byId.get(id)?.parent_id;
    while (ancestor) {
      if (requested.has(ancestor)) return false;
      ancestor = byId.get(ancestor)?.parent_id;
    }
    return true;
  });
  if (roots.every(id => (byId.get(id)?.parent_id ?? null) === parentId)) return objects;
  const moving = new Set(roots);
  return objects.map(object => moving.has(object.id) ? { ...object, parent_id: parentId } : object);
}

export function selectSceneObjectRows(selectedIds: string[], anchorId: string | null, clickedId: string,
                                      orderedIds: string[], toggle: boolean, range: boolean) {
  if (range) {
    const anchorIndex = orderedIds.indexOf(anchorId ?? '');
    const clickedIndex = orderedIds.indexOf(clickedId);
    if (anchorIndex >= 0 && clickedIndex >= 0) {
      const between = orderedIds.slice(Math.min(anchorIndex, clickedIndex), Math.max(anchorIndex, clickedIndex) + 1);
      return { ids: toggle ? [...new Set([...selectedIds, ...between])] : between, anchorId };
    }
  }
  if (toggle) {
    const ids = selectedIds.includes(clickedId)
      ? selectedIds.filter(id => id !== clickedId) : [...selectedIds, clickedId];
    return { ids, anchorId: ids.includes(anchorId ?? '') ? anchorId : ids[0] ?? null };
  }
  return { ids: [clickedId], anchorId: clickedId };
}

export function transformSceneObjectChildren(objects: SceneObject[], previous: SceneObject, next: SceneObject) {
  const descendants = sceneObjectDescendantIds(objects, previous.id);
  if (!descendants.size) return objects;
  const beforePosition = new THREE.Vector3().fromArray(previous.position);
  const afterPosition = new THREE.Vector3().fromArray(next.position);
  const beforeRotation = new THREE.Quaternion().fromArray(previous.quaternion_xyzw).normalize();
  const afterRotation = new THREE.Quaternion().fromArray(next.quaternion_xyzw).normalize();
  const rotationDelta = afterRotation.clone().multiply(beforeRotation.clone().invert());
  return objects.map(object => {
    if (!descendants.has(object.id)) return object;
    const position = new THREE.Vector3().fromArray(object.position).sub(beforePosition).applyQuaternion(rotationDelta).add(afterPosition);
    const rotation = rotationDelta.clone().multiply(new THREE.Quaternion().fromArray(object.quaternion_xyzw)).normalize();
    return { ...object, position: position.toArray(), quaternion_xyzw: rotation.toArray() };
  });
}

export function objectsFromProject(project: { scene_objects?: SceneObject[]; box?: { position: number[]; size: number[]; visible: boolean } }) {
  if (Array.isArray(project.scene_objects)) return structuredClone(project.scene_objects)
    .map(object => object.parent_id ? withScenePlacement(object, scenePlacementOptions(object))
      : groundedSceneObject(withScenePlacement(object, scenePlacementOptions(object))));
  if (project.box) {
    const object = createSceneObject(1);
    return [groundedSceneObject({ ...object, id: 'legacy-box', position: [...project.box.position], size: [...project.box.size], visible: project.box.visible })];
  }
  return [createSceneObject(1)];
}
