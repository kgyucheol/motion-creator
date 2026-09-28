import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { RobotScene, canRotateSelection, jointControls, jointForRing, COMBINED_JOINTS } from '../lib/robot-scene.ts';
import { rotatedGroupTargets, incrementRotation, quaternionFromDegrees, eulerDegrees, canMirrorSelection, translatedTargets } from '../lib/pose-transforms.ts';
import { BODY_GROUPS, allNodes, nodeMembers, selectMembers, selectionState, controlSelection, controlKey, visibleTreeHandles, expandVirtualControls } from '../lib/body-groups.ts';
import { createImportedSceneObjects, createSceneObjectGroup, groundedSceneObject, normalizedObjectSize, normalizedSceneAsset, objectVerticalHalfExtent, placeSceneObject, sceneObjectsOverlap, transformSceneObjectGroup } from '../lib/scene-objects.ts';
import { duplicateKeyframeAfter } from '../lib/keyframes.ts';

test('a duplicated keyframe is inserted immediately after the selection as an independent copy', () => {
  const frames = [
    { name: 'Start', duration: 2, qpos: [0, 1], pins: ['left_foot'] },
    { name: 'Reach', duration: 1.25, qpos: [2, 3], pins: ['right_foot'] },
    { name: 'Finish', duration: .5, qpos: [4, 5], pins: [] },
  ];
  const result = duplicateKeyframeAfter(frames, 1);
  assert.equal(result.index, 2);
  assert.deepEqual(result.keyframes.map(frame => frame.name), ['Start', 'Reach', 'Reach', 'Finish']);
  assert.deepEqual(result.keyframes[2], frames[1]);
  assert.notEqual(result.keyframes[2], frames[1]);
  assert.notEqual(result.keyframes[2].qpos, frames[1].qpos);
  assert.notEqual(result.keyframes[2].pins, frames[1].pins);
});

test('primitive scale constraints preserve spheres and round cylinders', () => {
  assert.deepEqual(normalizedObjectSize('box', [.2, .3, .4]), [.2, .3, .4]);
  assert.deepEqual(normalizedObjectSize('sphere', [.2, .3, .4], 'Y'), [.3, .3, .3]);
  assert.deepEqual(normalizedObjectSize('cylinder', [.2, .3, .4], 'Y'), [.3, .3, .4]);
});

test('rotated primitives are raised until their lowest point rests on the floor', () => {
  const base = { id: 'shape', name: 'Shape', position: [0, 0, -.2], size: [.2, .4, .6], mass_kg: 1,
    friction: .7, color: '#ffffff', opacity: 1, visible: true };
  const upright = { ...base, shape: 'box', quaternion_xyzw: [0, 0, 0, 1] };
  assert.equal(objectVerticalHalfExtent(upright), .3);
  assert.equal(groundedSceneObject(upright).position[2], .3);
  const turned = { ...upright, quaternion_xyzw: [0, Math.SQRT1_2, 0, Math.SQRT1_2] };
  assert.ok(Math.abs(objectVerticalHalfExtent(turned) - .1) < 1e-12);
  const cylinder = { ...base, shape: 'cylinder', size: [.2, .2, .6], quaternion_xyzw: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] };
  assert.ok(Math.abs(objectVerticalHalfExtent(cylinder) - .1) < 1e-12);
});

test('ground-locked objects stop at the nearest non-overlapping surface', () => {
  const fixed = { id: 'fixed', name: 'Fixed', shape: 'box', position: [0, 0, .5], size: [1, 1, 1],
    quaternion_xyzw: [0, 0, 0, 1], mass_kg: 1, friction: .7, color: '#ffffff', opacity: 1, visible: true };
  const previous = { ...fixed, id: 'moving', position: [-2, 0, .5] };
  const candidate = { ...previous, position: [-.4, 0, 2] };
  const placed = placeSceneObject(candidate, [fixed], { preventOverlap: true, surfaceSnap: false, groundLock: true }, previous);
  assert.deepEqual(placed.position, [-1, 0, .5]);
  assert.equal(sceneObjectsOverlap(placed, fixed), false);
});

test('surface snapping closes a small gap without requiring an overlap', () => {
  const fixed = { id: 'fixed', name: 'Fixed', shape: 'box', position: [0, 0, .5], size: [1, 1, 1],
    quaternion_xyzw: [0, 0, 0, 1], mass_kg: 1, friction: .7, color: '#ffffff', opacity: 1, visible: true };
  const candidate = { ...fixed, id: 'moving', position: [-1.015, 0, .5] };
  const placed = placeSceneObject(candidate, [fixed], { preventOverlap: true, surfaceSnap: true, groundLock: true });
  assert.ok(Math.abs(placed.position[0] + 1) < 1e-12);
  assert.equal(sceneObjectsOverlap(placed, fixed), false);
});

test('scene group rotation keeps child spacing and rotates each child orientation', () => {
  const base = { name: 'item', shape: 'box', size: [.2, .2, .2], quaternion_xyzw: [0, 0, 0, 1],
    mass_kg: 1, friction: .7, color: '#ffffff', opacity: 1, visible: true };
  const objects = [{ ...base, id: 'a', position: [1, 0, .5] }, { ...base, id: 'b', position: [-1, 0, .5] }];
  const group = createSceneObjectGroup(objects, ['a', 'b']);
  const next = { ...group, position: [0, 1, .5], quaternion_xyzw: quaternionFromDegrees([0, 0, 90]) };
  const transformed = transformSceneObjectGroup(objects, group, next);
  assert.ok(new THREE.Vector3(...transformed[0].position).distanceTo(new THREE.Vector3(0, 2, .5)) < 1e-10);
  assert.ok(new THREE.Vector3(...transformed[1].position).distanceTo(new THREE.Vector3(0, 0, .5)) < 1e-10);
  assert.ok(Math.abs(eulerDegrees(transformed[0].quaternion_xyzw)[2] - 90) < 1e-10);
});

test('imported scene assets are centered even when their source origin is far away', () => {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 4, 6));
  mesh.position.set(101, -48, 23);
  const normalized = normalizedSceneAsset(mesh, [100, -50, 20], [102, -46, 26]);
  normalized.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(normalized);
  assert.ok(bounds.min.distanceTo(new THREE.Vector3(-.5, -.5, -.5)) < 1e-10);
  assert.ok(bounds.max.distanceTo(new THREE.Vector3(.5, .5, .5)) < 1e-10);
});

test('standard Y-up glTF assets are converted to the editor Z-up frame before scaling', () => {
  const source = new THREE.Group();
  source.add(new THREE.Mesh(new THREE.BoxGeometry(2, 6, 4)));
  const sourceTop = new THREE.Object3D(); sourceTop.position.set(0, 3, 0); source.add(sourceTop);
  const aligned = normalizedSceneAsset(source, [-1, -2, -3], [1, 2, 3], [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
  aligned.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(aligned);
  assert.ok(bounds.min.distanceTo(new THREE.Vector3(-.5, -.5, -.5)) < 1e-10);
  assert.ok(bounds.max.distanceTo(new THREE.Vector3(.5, .5, .5)) < 1e-10);
  assert.ok(sourceTop.getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(0, 0, .5)) < 1e-10);
});

test('a multi-part GLB becomes independently positioned scene objects', () => {
  const suggestion = { shape: 'box', size: [1, 1, .1], mass_kg: .4, friction: .7, color: '#ffffff', fixed: true };
  const objects = createImportedSceneObjects({ asset_id: '0123456789abcdef01234567', name: 'Carton', source_name: 'carton.glb', source_format: 'glb', url: '',
    bounds_min: [-.5, -.5, 0], bounds_max: [.5, .5, 1], dimensions: [1, 1, 1], axis_transform_xyzw: [Math.SQRT1_2, 0, 0, Math.SQRT1_2], suggestion,
    parts: [
      { part_id: 'part-01-floor', node_name: 'Floor', name: 'Floor', url: '', bounds_min: [-.5, -.5, 0], bounds_max: [.5, .5, .1], dimensions: [1, 1, .1], suggestion },
      { part_id: 'part-02-wall', node_name: 'Wall', name: 'Wall', url: '', bounds_min: [-.5, -.5, .1], bounds_max: [-.4, .5, 1], dimensions: [.1, 1, .9], suggestion: { ...suggestion, size: [.1, 1, .9] } },
    ] }, 3);
  assert.equal(objects.length, 2);
  assert.equal(objects[0].asset_part_id, 'part-01-floor');
  assert.equal(objects[0].asset_node_name, 'Floor');
  assert.ok(new THREE.Vector3(...objects[0].position).distanceTo(new THREE.Vector3(.4, 0, .05)) < 1e-12);
  assert.ok(new THREE.Vector3(...objects[1].position).distanceTo(new THREE.Vector3(-.05, 0, .55)) < 1e-12);
  assert.ok(objects.every(object => object.placement?.ground_lock === false));
});

test('Delete removes a selected scene object but does not target robot controls', () => {
  const viewer = fixture();
  let deleted = null; let prevented = false;
  viewer.keyboardEnabled = true; viewer.selectedSceneObject = 'carton-wall';
  viewer.callbacks = { deleteObject: id => { deleted = id; } };
  viewer.handleKeyDown({ code: 'Delete', repeat: false, isComposing: false, altKey: false, ctrlKey: false, metaKey: false,
    target: null, preventDefault: () => { prevented = true; } });
  assert.equal(deleted, 'carton-wall'); assert.equal(prevented, true);
  deleted = null; viewer.selectedSceneObject = null;
  viewer.handleKeyDown({ code: 'Delete', repeat: false, isComposing: false, altKey: false, ctrlKey: false, metaKey: false,
    target: null, preventDefault() {} });
  assert.equal(deleted, null);
});

test('collision-only mode shows a five-panel open box and hides visual assets', () => {
  const viewer = Object.create(RobotScene.prototype);
  const object = { id: 'carton', name: 'Carton', shape: 'open_box', position: [0, 0, .2],
    quaternion_xyzw: [0, 0, 0, 1], size: [.72, .55, .4], wall_thickness_m: .02,
    mass_kg: 2, friction: .7, color: '#9b6b3c', opacity: 1, visible: true };
  const visual = new THREE.Group(); visual.userData.objectVisible = true;
  const proxy = viewer.primitiveSceneObject(object); proxy.userData.objectVisible = true;
  viewer.styleCollisionProxy(proxy, object);
  viewer.sceneObjects = { carton: visual }; viewer.collisionProxies = { carton: proxy }; viewer.dirty = false;

  viewer.setCollisionProxiesVisible(true);

  assert.equal(visual.visible, false); assert.equal(proxy.visible, true);
  assert.equal(proxy.children.length, 5);
  assert.ok(proxy.children.every(mesh => mesh.material.wireframe));
  assert.equal(viewer.dirty, true);
});

test('a ramen visual box can use a separate inscribed cylinder collision proxy', () => {
  const viewer = Object.create(RobotScene.prototype);
  const object = { id: 'ramen', name: 'Ramen', shape: 'box', position: [0, 0, 0],
    quaternion_xyzw: [0, 0, 0, 1], size: [.586, .147, .133], mass_kg: .5,
    friction: .7, color: '#ffffff', opacity: 1, visible: true,
    collision_shape: 'cylinder', collision_size: [.133, .133, .586],
    collision_quaternion_xyzw: [0, Math.SQRT1_2, 0, Math.SQRT1_2] };

  const proxy = viewer.collisionProxy(object);
  const geometryRoot = proxy.children[0]; const mesh = geometryRoot.children[0];
  assert.equal(mesh.geometry.type, 'CylinderGeometry');
  assert.ok(geometryRoot.scale.distanceTo(new THREE.Vector3(.133, .133, .586)) < 1e-12);
  assert.ok(new THREE.Vector3(0, 0, 1).applyQuaternion(geometryRoot.quaternion)
    .distanceTo(new THREE.Vector3(1, 0, 0)) < 1e-12);
});

function fixture() {
  const viewer = Object.create(RobotScene.prototype);
  viewer.state = { handles: { left_hand: { position: [.3, .2, .8] }, right_hand: { position: [.3, -.2, .8] } } };
  viewer.members = ['left_hand', 'right_hand'];
  viewer.selected = 'left_hand';
  viewer.editable = true;
  viewer.selectionLocked = false;
  viewer.gizmo = { dragging: false, axis: null };
  viewer.camera = new THREE.PerspectiveCamera(38, 1.5, .01, 100);
  viewer.camera.position.set(2, -3, 1.8);
  viewer.orbit = { target: new THREE.Vector3(0, 0, .7), update() {} };
  return viewer;
}

test('group focus centers on both hands and preserves viewing direction', () => {
  const viewer = fixture();
  const before = viewer.camera.position.clone().sub(viewer.orbit.target).normalize();
  viewer.focusSelection();
  assert.deepEqual(viewer.orbit.target.toArray(), [.3, 0, .8]);
  const after = viewer.camera.position.clone().sub(viewer.orbit.target).normalize();
  assert.ok(before.distanceTo(after) < 1e-10);
  assert.ok(viewer.camera.position.distanceTo(viewer.orbit.target) >= .55);
});

test('single focus centers on the selected hand', () => {
  const viewer = fixture();
  viewer.members = ['right_hand'];
  viewer.focusSelection();
  assert.deepEqual(viewer.orbit.target.toArray(), [.3, -.2, .8]);
});

test('a pose from another robot model is rejected before geom transforms are applied', () => {
  const viewer = Object.create(RobotScene.prototype);
  let error = '';
  viewer.modelId = 'g1';
  viewer.modelMismatchReported = false;
  viewer.callbacks = { error: value => { error = value; } };
  viewer.state = { marker: 'unchanged' };
  viewer.update({ model_id: 'g1-tools', geoms: {} });
  assert.deepEqual(viewer.state, { marker: 'unchanged' });
  assert.match(error, /새로고침/);
});

test('group rotation preserves separation and rotates around the selection center', () => {
  const initial = quaternionFromDegrees([25, 15, 10]);
  const after = incrementRotation(initial, 2, 90, 'world');
  const result = rotatedGroupTargets({ left_hand: [1, 2, 1], right_hand: [1, 0, 1] },
    { left_hand: initial, right_hand: initial }, [1, 1, 1], initial, after, []);
  assert.ok(new THREE.Vector3(...result.targets.left_hand).distanceTo(new THREE.Vector3(0, 1, 1)) < 1e-10);
  assert.ok(new THREE.Vector3(...result.targets.right_hand).distanceTo(new THREE.Vector3(2, 1, 1)) < 1e-10);
  for (const q of Object.values(result.orientations)) {
    assert.ok(Math.abs(new THREE.Quaternion(...q).dot(new THREE.Quaternion(...after))) > 1-1e-10);
  }
});

test('a position-pinned hand has a rotation target without a conflicting position target', () => {
  const result = rotatedGroupTargets({ left_hand: [1, 2, 1] }, { left_hand: [0, 0, 0, 1] },
    [1, 2, 1], [0, 0, 0, 1], quaternionFromDegrees([15, 0, 0]), ['left_hand']);
  assert.deepEqual(result.targets, {});
  assert.ok(Math.abs(eulerDegrees(result.orientations.left_hand)[0]-15) < 1e-10);
});

test('local and world rotation increments use different axes for a rotated hand', () => {
  const initial = quaternionFromDegrees([0, 0, 90]);
  const local = incrementRotation(initial, 0, 20, 'local');
  const world = incrementRotation(initial, 1, 20, 'world');
  const wrong = incrementRotation(initial, 0, 20, 'world');
  assert.ok(Math.abs(new THREE.Quaternion(...local).dot(new THREE.Quaternion(...world))) > 1-1e-10);
  assert.ok(Math.abs(new THREE.Quaternion(...local).dot(new THREE.Quaternion(...wrong))) < .99);
});

test('a new wrist axis uses a single ring at its physical anchor', () => {
  const viewer = fixture();
  const name = 'left_wrist_pitch_joint';
  viewer.state.handles[name] = { position: [.2, .3, .8], quaternion: [0, 0, 0, 1] };
  viewer.state.hinges = { [name]: { position: [.2, .3, .8], axis_world: [0, 1, 0] } };
  viewer.markers = {}; viewer.labels = {};
  viewer.pivot = new THREE.Object3D();
  viewer.markerVisible = true;
  viewer.transformMode = 'rotate';
  viewer.gizmo.setSpace = (space) => { viewer.gizmo.space = space; };
  viewer.gizmo.attach = (object) => { viewer.gizmo.object = object; };
  viewer.gizmo.detach = () => { viewer.gizmo.object = null; };
  viewer.select(name, ['left_foot', 'right_foot']);
  assert.equal(canRotateSelection([name]), true);
  assert.equal(canRotateSelection([name, 'right_wrist_pitch_joint']), false);
  assert.equal(viewer.gizmo.object, viewer.pivot);
  assert.equal(viewer.gizmo.space, 'local');
  assert.equal(viewer.gizmo.showX, false);
  assert.equal(viewer.gizmo.showY, false);
  assert.equal(viewer.gizmo.showZ, true);
  assert.deepEqual(viewer.pivot.position.toArray(), [.2, .3, .8]);
  assert.ok(new THREE.Vector3(0, 0, 1).applyQuaternion(viewer.pivot.quaternion).distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-10);
});

test('an angle-only pin blocks rotation while keeping the handle selectable for translation', () => {
  const viewer = fixture();
  viewer.state.handles.left_hand.quaternion = [0, 0, 0, 1];
  viewer.markers = { left_hand: new THREE.Mesh(new THREE.SphereGeometry(.02), new THREE.MeshBasicMaterial()) };
  viewer.labels = {}; viewer.pivot = new THREE.Object3D(); viewer.markerVisible = true;
  viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = () => {};
  viewer.gizmo.attach = object => { viewer.gizmo.object = object; };
  viewer.gizmo.detach = () => { viewer.gizmo.object = null; };
  viewer.select('left_hand', [], ['left_hand'], ['left_hand']);
  assert.equal(viewer.gizmo.object, null);
  viewer.transformMode = 'translate';
  viewer.select('left_hand', [], ['left_hand'], ['left_hand']);
  assert.equal(viewer.gizmo.object, viewer.pivot);
  viewer.markers.left_hand.geometry.dispose(); viewer.markers.left_hand.material.dispose();
});

test('Alt picking cycles through coincident joint markers and ignores hidden handles', () => {
  const viewer = fixture();
  viewer.renderer = { domElement: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }) } };
  viewer.ray = new THREE.Raycaster(); viewer.pointer = new THREE.Vector2();
  viewer.camera.position.set(0, 0, 2); viewer.camera.lookAt(0, 0, 0); viewer.camera.updateMatrixWorld();
  viewer.markers = Object.fromEntries(['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_hand'].map(name => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(.03), new THREE.MeshBasicMaterial());
    mesh.name = name; mesh.visible = name !== 'left_hand'; mesh.updateMatrixWorld();
    return [name, mesh];
  }));
  const event = { clientX: 50, clientY: 50, altKey: true };
  const visited = [];
  for (let i = 0; i < 3; i++) { viewer.selected = viewer.pick(event); visited.push(viewer.selected); }
  assert.equal(new Set(visited).size, 3);
  assert.ok(!visited.includes('left_hand'));
  assert.equal(viewer.pick(event), visited[0]);
  Object.values(viewer.markers).forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
});

test('left and right hip axes collapse separately into three-axis controls', () => {
  const names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint'];
  assert.deepEqual(jointControls(names), ['left_hip', 'left_knee_joint', 'right_hip', 'right_knee_joint']);
  assert.ok(canRotateSelection(['left_hip', 'right_hip']));
  const viewer = fixture();
  viewer.state.handles.left_hip = { position: [0, .12, .65], quaternion: [0, 0, 0, 1] };
  viewer.state.hinges = {};
  viewer.markers = Object.fromEntries([...names, 'left_hip', 'right_hip'].map(name => [name, new THREE.Mesh(new THREE.SphereGeometry(.02), new THREE.MeshBasicMaterial())]));
  viewer.labels = {};
  viewer.pivot = new THREE.Object3D(); viewer.markerVisible = true; viewer.handleLayer = 'joints';
  viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = () => {};
  viewer.gizmo.attach = () => {};
  viewer.gizmo.detach = () => {};
  viewer.select('left_hip', []);
  assert.deepEqual(viewer.pivot.position.toArray(), [0, .12, .65]);
  assert.ok(viewer.gizmo.showX && viewer.gizmo.showY && viewer.gizmo.showZ);
  assert.ok(viewer.markers.left_hip.visible && viewer.markers.right_hip.visible);
  assert.equal(viewer.markers.left_hip_yaw_joint.visible, false);
  assert.equal(viewer.markers.right_hip_roll_joint.visible, false);
  Object.values(viewer.markers).forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
});

test('29 motors remain addressable through 21 joint controls', () => {
  const names = [...Object.values(COMBINED_JOINTS).flat(), 'left_knee_joint', 'right_knee_joint',
    ...['left', 'right'].flatMap(side => ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'].map(part => `${side}_${part}_joint`))];
  assert.equal(new Set(names).size, 29);
  const controls = jointControls(names);
  assert.equal(controls.length, 21);
  for (const key of Object.keys(COMBINED_JOINTS)) assert.ok(controls.includes(key));
  assert.ok(canRotateSelection(['waist']));
  assert.ok(canRotateSelection(['left_ankle']));
  assert.equal(canRotateSelection(['left_ankle', 'right_ankle']), false);
});

test('ankle rings follow real axes and send only the corresponding motor angle', () => {
  const viewer = fixture();
  const initialFrame = new THREE.Quaternion().setFromEuler(new THREE.Euler(.2, -.35, .6));
  const x = new THREE.Vector3(1, 0, 0).applyQuaternion(initialFrame);
  const y = new THREE.Vector3(0, 1, 0).applyQuaternion(initialFrame);
  viewer.state.handles.left_ankle = { position: [.1, .2, .05], quaternion: [0, 0, 0, 1] };
  viewer.state.hinges = {
    left_ankle_roll_joint: { axis_world: x.toArray(), angle: .1, limits: [-.25, .25] },
    left_ankle_pitch_joint: { axis_world: y.toArray(), angle: -.1, limits: [-.8, .5] },
  };
  viewer.markers = {}; viewer.labels = {}; viewer.pivot = new THREE.Object3D();
  viewer.markerVisible = true; viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = space => { viewer.gizmo.space = space; };
  viewer.gizmo.attach = object => { viewer.gizmo.object = object; };
  viewer.gizmo.detach = () => {};
  viewer.select('left_ankle', []);
  assert.equal(viewer.gizmo.space, 'local');
  assert.ok(viewer.gizmo.showX && viewer.gizmo.showY);
  assert.equal(viewer.gizmo.showZ, false);
  assert.deepEqual(viewer.pivot.position.toArray(), [.1, .2, .05]);
  assert.ok(new THREE.Vector3(1, 0, 0).applyQuaternion(viewer.pivot.quaternion).distanceTo(x) < 1e-10);
  assert.ok(new THREE.Vector3(0, 1, 0).applyQuaternion(viewer.pivot.quaternion).distanceTo(y) < 1e-10);
  for (const axis of ['Z', 'E', 'XYZE', null]) assert.equal(jointForRing('left_ankle', axis), null);
  for (const [axis, part, component] of [['X', 'roll', 'x'], ['Y', 'pitch', 'y']]) {
    const ring = jointForRing('left_ankle', axis);
    assert.deepEqual(ring, { key: `left_ankle_${part}_joint`, component });
    const joint = viewer.state.hinges[ring.key];
    viewer.hingeDrag = { quaternion: initialFrame.clone(), angle: joint.angle, limits: joint.limits, delta: 0, lastTwist: 0, ...ring };
    const calls = [];
    viewer.callbacks = { jointAngle: (name, value) => calls.push([name, value]) };
    const direction = axis === 'X' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    viewer.pivot.quaternion.copy(initialFrame).multiply(new THREE.Quaternion().setFromAxisAngle(direction, .1));
    viewer.applyHingeDrag(); viewer.applyHingeDrag();
    assert.equal(calls[0][0], ring.key);
    assert.ok(Math.abs(calls[0][1] - joint.angle - .1) < 1e-10);
    assert.ok(Math.abs(calls[1][1] - calls[0][1]) < 1e-10);
    viewer.pivot.quaternion.copy(initialFrame).multiply(new THREE.Quaternion().setFromAxisAngle(direction, 1));
    viewer.applyHingeDrag();
    assert.equal(calls[2][1], joint.limits[1]);
  }
});

test('anatomical tree covers all 29 motors exactly once and includes hands and support points', () => {
  const leaves = BODY_GROUPS.flatMap(nodeMembers);
  assert.equal(leaves.length, new Set(leaves).size);
  const motorNames = [...Object.values(COMBINED_JOINTS).flat(), 'left_knee_joint', 'right_knee_joint',
    ...['left', 'right'].flatMap(side => ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'].map(part => `${side}_${part}_joint`))];
  assert.deepEqual(leaves.filter(key => key.endsWith('_joint')).sort(), motorNames.sort());
  assert.deepEqual(leaves.filter(key => !key.endsWith('_joint')).sort(), ['pelvis','left_hand','right_hand','left_foot','right_foot'].sort());
  const nodes = allNodes();
  const hands = nodes.find(node => node.id === 'hands');
  const left = nodes.find(node => node.id === 'left_hands');
  assert.equal(nodeMembers(hands).length, 8);
  assert.deepEqual(nodeMembers(left), ['left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint','left_hand']);
});

test('group selection, single-child replacement, partial state and Shift toggle', () => {
  const hands = allNodes().find(node => node.id === 'hands');
  const group = nodeMembers(hands);
  let selected = selectMembers(['pelvis'], group, false);
  assert.deepEqual(selected, group);
  assert.equal(selectionState(hands, selected).all, true);
  selected = selectMembers(selected, ['left_wrist_pitch_joint'], false);
  assert.deepEqual(selected, ['left_wrist_pitch_joint']);
  assert.deepEqual(selectionState(hands, selected), { count: 1, total: 8, all: false, partial: true });
  selected = selectMembers(selected, ['pelvis'], true);
  selected = selectMembers(selected, group, true);
  assert.equal(new Set(selected).size, 9);
  selected = selectMembers(selected, group, true);
  assert.deepEqual(selected, ['pelvis']);
});

test('full groups preserve integrated controls while an individual child keeps its own axis', () => {
  const nodes = allNodes();
  const hands = nodeMembers(nodes.find(node => node.id === 'hands'));
  assert.deepEqual(controlSelection(hands), ['left_hand', 'right_hand']);
  assert.equal(controlKey(hands, 'left_wrist_pitch_joint'), 'left_hand');
  const hips = nodeMembers(nodes.find(node => node.id === 'left_hips'));
  assert.deepEqual(controlSelection(hips), ['left_hip']);
  assert.deepEqual(controlSelection(['left_hip_roll_joint']), ['left_hip_roll_joint']);
  const foot = nodeMembers(nodes.find(node => node.id === 'left_feet'));
  assert.deepEqual(controlSelection(foot), ['left_ankle']);
  assert.equal(jointForRing(controlKey(foot, foot[0]), 'Y').key, 'left_ankle_pitch_joint');
  const legacy = expandVirtualControls(['waist','left_hip','left_hand']);
  assert.equal(legacy.length, 7);
  assert.deepEqual(controlSelection(legacy), ['waist','left_hip','left_hand']);
});

test('accordion visibility changes do not alter selection', () => {
  const selected = ['left_wrist_pitch_joint'];
  const closed = visibleTreeHandles([]);
  const expanded = visibleTreeHandles(['hands', 'left_hands']);
  assert.ok(closed.includes('left_hand'));
  assert.equal(closed.includes('left_wrist_pitch_joint'), false);
  assert.ok(expanded.includes('left_hand') && expanded.includes('left_wrist_pitch_joint'));
  assert.deepEqual(selected, ['left_wrist_pitch_joint']);
});

test('selecting a hand group highlights every leaf and selecting a child restores a single-axis gizmo', () => {
  const viewer = fixture();
  const group = allNodes().find(node => node.id === 'left_hands');
  const leaves = nodeMembers(group);
  viewer.state.hinges = {};
  viewer.markers = {}; viewer.labels = {};
  leaves.forEach((key, i) => {
    viewer.state.handles[key] = { position: [.2+i*.01, .2, .8], quaternion: [0, 0, 0, 1] };
    viewer.markers[key] = new THREE.Mesh(new THREE.SphereGeometry(.02), new THREE.MeshBasicMaterial());
    if (key.endsWith('_joint')) viewer.state.hinges[key] = { position: viewer.state.handles[key].position, axis_world: [1,0,0] };
  });
  viewer.pivot = new THREE.Object3D(); viewer.markerVisible = true; viewer.visibleHandles = ['left_hand'];
  viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = () => {}; viewer.gizmo.attach = () => {}; viewer.gizmo.detach = () => {};
  viewer.select(leaves[0], [], leaves);
  assert.ok(viewer.gizmo.showX && viewer.gizmo.showY && viewer.gizmo.showZ);
  assert.deepEqual(viewer.pivot.position.toArray(), viewer.state.handles.left_hand.position);
  for (const key of leaves) {
    assert.equal(viewer.markers[key].visible, true);
    assert.equal(viewer.markers[key].material.color.getHexString(), '80f2c7');
  }
  viewer.select('left_wrist_pitch_joint', [], ['left_wrist_pitch_joint']);
  assert.equal(viewer.gizmo.showX, false); assert.equal(viewer.gizmo.showY, false); assert.equal(viewer.gizmo.showZ, true);
  assert.deepEqual(viewer.pivot.position.toArray(), viewer.state.handles.left_wrist_pitch_joint.position);
  Object.values(viewer.markers).forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
});

test('mirror translation opens and closes pairs from either active side without changing forward/up displacement', () => {
  const positions = { left_hand: [.2,.25,.8], right_hand: [.2,-.25,.8] };
  assert.ok(canMirrorSelection(Object.keys(positions)));
  for (const keys of [['left_hand'], ['left_hand','right_foot'], ['left_hand','right_hand','pelvis']]) assert.equal(canMirrorSelection(keys), false);
  const open = translatedTargets(positions, [.1,.05,.03], { active: 'left_hand', rootQuaternion: [0,0,0,1] });
  assert.ok(Math.abs(open.left_hand[1] - open.right_hand[1] - .6) < 1e-10);
  for (const p of Object.values(open)) { assert.ok(Math.abs(p[0]-.3)<1e-10); assert.ok(Math.abs(p[2]-.83)<1e-10); }
  const rightOpen = translatedTargets(positions, [0,-.05,0], { active: 'right_hand', rootQuaternion: [0,0,0,1] });
  assert.ok(Math.abs(rightOpen.left_hand[1] - rightOpen.right_hand[1] - .6) < 1e-10);
  const closed = translatedTargets(positions, [0,-.05,0], { active: 'left_hand', rootQuaternion: [0,0,0,1] });
  assert.ok(Math.abs(closed.left_hand[1] - closed.right_hand[1] - .4) < 1e-10);
  const normal = translatedTargets(positions, [0,.05,0]);
  assert.ok(Math.abs(normal.left_hand[1] - normal.right_hand[1] - .5) < 1e-10);
  assert.deepEqual(positions.left_hand, [.2,.25,.8]);
});

test('mirror follows the robot sagittal plane when the robot is turned', () => {
  const root = quaternionFromDegrees([0,0,90]);
  const positions = { left_hand: [-.25,.2,.8], right_hand: [.25,.2,.8] };
  const result = translatedTargets(positions, [-.05,.1,0], { active: 'left_hand', rootQuaternion: root });
  assert.ok(Math.abs(result.left_hand[0]+.3) < 1e-10);
  assert.ok(Math.abs(result.right_hand[0]-.3) < 1e-10);
  assert.ok(Math.abs(result.left_hand[1]-.3) < 1e-10);
  assert.ok(Math.abs(result.right_hand[1]-.3) < 1e-10);
});

test('Ctrl+Z invokes pose history while text input, dragging and task mode retain their behavior', () => {
  const viewer = fixture();
  const calls = [];
  viewer.callbacks = { history: redo => calls.push(redo) };
  let prevented = 0;
  const event = { code:'KeyZ', ctrlKey:true, metaKey:false, altKey:false, shiftKey:false, target:null,
    preventDefault() { prevented++; } };
  viewer.handleKeyDown(event);
  viewer.handleKeyDown({ ...event, shiftKey:true });
  assert.deepEqual(calls, [false,true]);
  assert.equal(prevented, 2);
  viewer.handleKeyDown({ ...event, target:{ closest: () => ({}) } });
  viewer.gizmo.dragging = true; viewer.handleKeyDown(event); viewer.gizmo.dragging = false;
  viewer.keyboardEnabled = false; viewer.handleKeyDown(event); viewer.keyboardEnabled = true;
  viewer.editable = false; viewer.handleKeyDown(event);
  assert.deepEqual(calls, [false,true]);
  assert.equal(prevented, 2);
});

test('mirror gizmo anchors at the active member while F still frames both sides', () => {
  const viewer = fixture();
  for (const h of Object.values(viewer.state.handles)) h.quaternion = [0,0,0,1];
  viewer.markers = {}; viewer.labels = {}; viewer.pivot = new THREE.Object3D();
  viewer.markerVisible = true; viewer.transformMode = 'translate'; viewer.mirrorTranslation = true;
  viewer.gizmo.setSpace = () => {}; viewer.gizmo.attach = () => {}; viewer.gizmo.detach = () => {};
  viewer.select('right_hand', [], ['left_hand','right_hand']);
  assert.deepEqual(viewer.pivot.position.toArray(), [.3,-.2,.8]);
  viewer.focusSelection();
  assert.deepEqual(viewer.orbit.target.toArray(), [.3,0,.8]);
  viewer.setMirrorTranslation(false);
  assert.deepEqual(viewer.pivot.position.toArray(), [.3,0,.8]);
});
