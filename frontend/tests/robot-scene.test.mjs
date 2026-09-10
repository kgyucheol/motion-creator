import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { RobotScene, canRotateSelection } from '../lib/robot-scene.ts';
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
