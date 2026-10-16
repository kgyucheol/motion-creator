import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { DEFAULT_CONTAINER_RADIUS, DetectedObjectsLayer, axisEndpoints, cylinderFromEndpoints, parseDetections, segmentPlacement } from '../lib/detected-objects.ts';

const row = (overrides = {}) => ({
  label: 'ramen_container', confidence: .9, start: [-.3, 0, .49], end: [.3, .05, .43], normal: [0, 0, -1], axis: [1, .08, -.1],
  length: .6, width: .13, depth_extent: .06, points: 7000, clipped: [false, true], ...overrides,
});
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const message = objects => ({ type: 'objects', frame: 'camera_optical', stamp: 1, inference_ms: 740, objects });

test('the default radius is the 6.5 cm cup radius', () => {
  assert.equal(DEFAULT_CONTAINER_RADIUS, .065);
});

test('the axis is the visible ridge moved one radius away from the camera along the surface normal', () => {
  const { start, end } = axisEndpoints(row(), .065);
  assert.deepEqual(start.map(value => Number(value.toFixed(6))), [-.3, 0, .555]);
  assert.deepEqual(end.map(value => Number(value.toFixed(6))), [.3, .05, .495]);
  const tilted = axisEndpoints(row({ normal: [0, -.6, -.8] }), .1);
  assert.ok(Math.abs(tilted.start[1] - .06) < 1e-12 && Math.abs(tilted.start[2] - .57) < 1e-12);
});

test('a segment placement points the Y axis from the start to the end and centers between them', () => {
  const placement = segmentPlacement([0, 0, 1], [0, 0, 3]);
  assert.deepEqual(placement.position.toArray(), [0, 0, 2]);
  assert.equal(placement.length, 2);
  const direction = new THREE.Vector3(0, 1, 0).applyQuaternion(placement.quaternion);
  assert.ok(direction.distanceTo(new THREE.Vector3(0, 0, 1)) < 1e-9);
  assert.equal(segmentPlacement([1, 1, 1], [1, 1, 1]).length, 0);
});

test('detection messages decode and malformed ones are rejected', () => {
  const parsed = parseDetections(encode(message([row()])));
  assert.equal(parsed.objects.length, 1);
  assert.deepEqual(parsed.objects[0].clipped, [false, true]);
  assert.equal(parseDetections(encode({ type: 'cloud' })), null);
  assert.equal(parseDetections(new TextEncoder().encode('not json')), null);
  assert.equal(parseDetections(encode({ ...message([]), frame: 'base' })), null);
  const filtered = parseDetections(encode(message([row(), row({ start: [1, 2] }), row({ width: null })])));
  assert.equal(filtered.objects.length, 1);
});

test('a scene cylinder from two world points has its Z axis along start to end, its center between them and size [diameter, diameter, length]', () => {
  const cylinder = cylinderFromEndpoints([1, 0, .5], [1, .6, .5], .065);
  assert.ok(Math.abs(cylinder.size[0] - .13) < 1e-12 && Math.abs(cylinder.size[2] - .6) < 1e-12);
  assert.deepEqual(cylinder.position.map(value => Number(value.toFixed(9))), [1, .3, .5]);
  const axis = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion().fromArray(cylinder.quaternion_xyzw));
  assert.ok(axis.distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-9);
});

test('the layer reports its rows as world cylinders through the same transform that draws the overlay', () => {
  const layer = new DetectedObjectsLayer(() => {});
  assert.deepEqual(layer.worldRows(), []);
  layer.root.position.set(.1, 0, 1);
  layer.root.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);   // camera-frame x maps to world y
  layer.apply(parseDetections(encode(message([row({ start: [-.3, 0, .5], end: [.3, 0, .5], normal: [0, 0, -1], clipped: [false, false] })]))));
  const [world] = layer.worldRows();
  assert.equal(world.clipped, false);
  assert.ok(Math.abs(world.size[2] - .6) < 1e-9);
  assert.deepEqual(world.position.map(value => Number(value.toFixed(9))), [.1, 0, 1.565]);   // axis is 6.5 cm farther than the ridge
  const axis = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion().fromArray(world.quaternion_xyzw));
  assert.ok(axis.distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-9);
});
