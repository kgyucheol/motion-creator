import test from 'node:test';
import assert from 'node:assert/strict';
import { sectionGrasps } from '../lib/lab-grasp.ts';
import { DEFAULT_RAMEN_GRASP_COEFFICIENTS } from '../lib/scene-objects.ts';

const target = { center: [.5, .9], radius: .0675 };
const grasp = side => ({ ...DEFAULT_RAMEN_GRASP_COEFFICIENTS[side] });
const run = (left, right = grasp('right')) => sectionGrasps(target, 0, [0, 0, .79], [0, 0, 0, 1], { left, right });
const deg = radians => radians * 180 / Math.PI;

test('the editor grasp point sits clearance beyond the surface toward the robot, tool pointing at the cylinder', () => {
  const { left, right } = run(grasp('left'));
  assert.ok(Math.abs(left.x - (.5 - .0675 - .1)) < 1e-9 && Math.abs(left.z - .9) < 1e-9);
  assert.ok(Math.abs(deg(left.angle)) < 1e-6 && !left.flip && Math.abs(left.inPlane - 1) < 1e-9);
  assert.ok(Math.abs(left.lateral - .25) < 1e-9 && Math.abs(right.lateral + .25) < 1e-9); // along the axis, out of the section
});

test('height, pitch, roll and yaw carry over from the editor coefficients', () => {
  assert.ok(Math.abs(run({ ...grasp('left'), height: .03 }).left.z - .93) < 1e-9);
  assert.ok(Math.abs(deg(run({ ...grasp('left'), pitch: 20 }).left.angle) - 20) < 1e-6);
  assert.equal(run({ ...grasp('left'), roll: 180 }).left.flip, true);
  assert.ok(run({ ...grasp('left'), yaw: 90 }).left.inPlane < 1e-6);
});

test('in a tilted box the grasp angle follows the box: along the opening, plus the pitch; the position stays', () => {
  const tilt = 50 * Math.PI / 180, up = [-Math.sin(tilt), Math.cos(tilt)]; // box pitch -50 deg
  const boxed = (left) => sectionGrasps(target, 0, [0, 0, .79], [0, 0, 0, 1], { left, right: grasp('right') }, up).left;
  const flat = run(grasp('left')).left;
  const entry = boxed(grasp('left'));
  assert.ok(Math.abs(deg(entry.angle) + 40) < 1e-6, `enters along -up (${deg(entry.angle)})`);
  assert.ok(Math.abs(entry.x - flat.x) < 1e-12 && Math.abs(entry.z - flat.z) < 1e-12);
  assert.ok(Math.abs(deg(boxed({ ...grasp('left'), pitch: 20 }).angle) + 20) < 1e-6);
  assert.equal(entry.flip, false);
  assert.equal(boxed({ ...grasp('left'), roll: 180 }).flip, true);
});
