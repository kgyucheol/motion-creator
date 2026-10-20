import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { LengthPrefixedFramer, cloudPlacement, parsePointCloud } from '../lib/point-cloud.ts';
import { headCameraViewQuaternion } from '../lib/robot-scene.ts';

function message(points) {
  const out = new Uint8Array(12 + points.length * 9);
  const view = new DataView(out.buffer);
  out.set([0x50, 0x43, 0x30, 0x31]);
  view.setUint32(4, points.length, true);
  view.setUint16(8, 4, true);
  points.forEach(({ xyz, rgb }, index) => {
    xyz.forEach((value, axis) => view.setInt16(12 + index * 6 + axis * 2, value, true));
    out.set(rgb, 12 + points.length * 6 + index * 3);
  });
  return out;
}

test('a cloud message decodes millimeter positions and colors', () => {
  const cloud = parsePointCloud(message([{ xyz: [-373, 194, 760], rgb: [255, 0, 10] }, { xyz: [1, -2, 3], rgb: [1, 2, 3] }]));
  assert.equal(cloud.count, 2);
  assert.deepEqual(Array.from(cloud.positions), [-0.373, 0.194, 0.76, 0.001, -0.002, 0.003].map(Math.fround));
  assert.deepEqual(Array.from(cloud.colors), [255, 0, 10, 1, 2, 3]);
});

test('malformed cloud messages are rejected', () => {
  const good = message([{ xyz: [1, 2, 3], rgb: [4, 5, 6] }]);
  assert.equal(parsePointCloud(good.slice(0, 11)), null);
  assert.equal(parsePointCloud(good.slice(0, good.length - 1)), null);
  const wrongMagic = good.slice();
  wrongMagic[0] = 0x58;
  assert.equal(parsePointCloud(wrongMagic), null);
});

test('the framer reassembles length-prefixed messages split across chunks', () => {
  const first = message([{ xyz: [1, 2, 3], rgb: [4, 5, 6] }]);
  const second = message([{ xyz: [7, 8, 9], rgb: [1, 1, 1] }, { xyz: [0, 0, 0], rgb: [2, 2, 2] }]);
  const stream = new Uint8Array(4 + first.length + 4 + second.length);
  new DataView(stream.buffer).setUint32(0, first.length, true);
  stream.set(first, 4);
  new DataView(stream.buffer).setUint32(4 + first.length, second.length, true);
  stream.set(second, 8 + first.length);
  const framer = new LengthPrefixedFramer();
  const seen = [];
  for (let offset = 0; offset < stream.length; offset += 7) seen.push(...framer.push(stream.slice(offset, offset + 7)));
  assert.equal(seen.length, 2);
  assert.deepEqual(Array.from(seen[0]), Array.from(first));
  assert.deepEqual(Array.from(seen[1]), Array.from(second));
});

test('optical axes land on the d435 link forward/right/down directions', () => {
  const placement = cloudPlacement([1, 2, 3], headCameraViewQuaternion([0, 0, 0, 1]));
  const world = optical => new THREE.Vector3(...optical).applyQuaternion(placement.quaternion).toArray().map(value => Math.round(value * 1e6) / 1e6 + 0);
  assert.deepEqual(world([0, 0, 1]), [1, 0, 0]);
  assert.deepEqual(world([1, 0, 0]), [0, -1, 0]);
  assert.deepEqual(world([0, 1, 0]), [0, 0, -1]);
  assert.deepEqual(placement.position.toArray(), [1, 2, 3]);
});

test('the lens offset is expressed in link axes and moves the whole cloud', () => {
  const view = headCameraViewQuaternion(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2).toArray());
  const placement = cloudPlacement([0, 0, 0], view, [0.01, 0.03, 0.02]);
  // link x/y/z after a +90° yaw point to world +y/-x/+z
  const expected = [-0.03, 0.01, 0.02];
  placement.position.toArray().forEach((value, axis) => assert.ok(Math.abs(value - expected[axis]) < 1e-9, `${axis}: ${value}`));
});
