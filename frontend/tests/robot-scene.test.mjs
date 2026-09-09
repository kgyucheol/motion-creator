import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { RobotScene } from '../lib/robot-scene.ts';
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
