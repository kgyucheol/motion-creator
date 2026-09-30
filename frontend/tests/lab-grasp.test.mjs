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
