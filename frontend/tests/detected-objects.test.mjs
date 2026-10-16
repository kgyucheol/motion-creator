import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { DEFAULT_CONTAINER_RADIUS, axisEndpoints, parseDetections, segmentPlacement } from '../lib/detected-objects.ts';

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
