import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { RobotScene, canRotateSelection, jointControls, jointForRing, COMBINED_JOINTS } from '../lib/robot-scene.ts';
import { rotatedGroupTargets, incrementRotation, quaternionFromDegrees, eulerDegrees } from '../lib/pose-transforms.ts';

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
