import test from 'node:test';
import assert from 'node:assert/strict';
import { autoSteps, drawnWaypoints, pointAngles, snapToCircle } from '../lib/drawn-path.ts';

test('drawn points follow the travel direction unless the user sets an angle', () => {
  const points = [{ x: 0, z: 0, angle: null, steps: null }, { x: .1, z: 0, angle: null, steps: null }, { x: .1, z: .1, angle: 1, steps: null }];
  const angles = pointAngles(points);
  assert.equal(angles[0], 0); assert.equal(angles[1], 0); assert.equal(angles[2], 1);
});

test('segments split by distance or turn, or by the user count, and keep the drawn points', () => {
  const stepping = { stepM: .02, stepDeg: 10, speedMS: .1, turnDegS: 90, minStepS: .1 };
  const points = [{ x: 0, z: 0, angle: 0, steps: null }, { x: .1, z: 0, angle: 0, steps: null }, { x: .1, z: 0, angle: Math.PI / 2, steps: null },
    { x: .2, z: 0, angle: Math.PI / 2, steps: 3 }];
  assert.equal(autoSteps(points, 1, stepping), 5); // 100 mm / 20 mm
  assert.equal(autoSteps(points, 2, stepping), 9); // 90 deg / 10 deg
  const waypoints = drawnWaypoints(points, stepping);
  assert.equal(waypoints.length, 1 + 5 + 9 + 3);
  assert.deepEqual(waypoints[5].tcp, [.1, 0]);
  assert.ok(Math.abs(waypoints[14].angle - Math.PI / 2) < 1e-12);
  assert.ok(waypoints.slice(1).every(w => w.duration >= .1 && w.stage === 'draw'));
});

test('snapping pulls a nearby point onto the circle only within the tolerance', () => {
  assert.deepEqual(snapToCircle(.105, 0, [0, 0], .1, .01), { x: .1, z: 0 });
  assert.equal(snapToCircle(.13, 0, [0, 0], .1, .01), null);
});

test('planned waypoints become editable points that replay the same path', async () => {
  const { waypointsToPoints } = await import('../lib/drawn-path.ts');
  const w = (stage, x, angle = 0) => ({ stage, label: stage, tcp: [x, 0], angle, duration: .2 });
  const planned = [w('start', 0), w('approach', .02), w('approach', .04), w('pry', .05, .1), w('pry', .06, .2), w('lift', .08, .2), w('lift', .1, .2)];
  const points = waypointsToPoints(planned);
  assert.deepEqual(points.map(p => [p.tcp?.[0] ?? p.x, p.steps]), [[0, null], [.04, 2], [.05, 1], [.06, 1], [.1, 2]]);
  const replay = drawnWaypoints(points, { stepM: 1, stepDeg: 90, speedMS: 1, turnDegS: 90, minStepS: .1 });
  assert.deepEqual(replay.map(r => r.tcp[0]).map(v => +v.toFixed(3)), planned.map(p => p.tcp[0]));
  assert.deepEqual(replay.map(r => +r.angle.toFixed(3)), planned.map(p => p.angle));
});

test('a drawn point can be moved to another place in the order', async () => {
  const { movePoint } = await import('../lib/drawn-path.ts');
  const p = ['a', 'b', 'c', 'd'].map((id, i) => ({ x: i, z: 0, angle: null, steps: null, id }));
  const ids = list => list.map(q => q.id).join('');
  assert.equal(ids(movePoint(p, 0, 2)), 'bcad');
  assert.equal(ids(movePoint(p, 3, 0)), 'dabc');
  assert.equal(ids(movePoint(p, 1, 1)), 'abcd');
  assert.equal(ids(movePoint(p, 1, 9)), 'acdb');
  assert.equal(movePoint(p, 7, 0), p);
});
