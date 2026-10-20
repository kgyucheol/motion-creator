import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { numberCylinders, retarget, runBatch, translatePaths } from '../lib/lab-batch.ts';
import { restingScene } from '../lib/cylinder-sim.ts';

const demos = JSON.parse(readFileSync(new URL('../lib/cylinder-demos.json', import.meta.url), 'utf8'));
const clamp = demos.find(d => d.id === 'clamp-rigid');
const short = name => name.replace('Mupama_Stack_', '');

test('cylinders are numbered layer by layer from the box opening, upper wall first', () => {
  const order = numberCylinders(clamp.base_scene).map(n => short(n.name));
  assert.deepEqual(order, ['L03_R04', 'L03_R03', 'L03_R02', 'L03_R01', 'L02_R04', 'L02_R03', 'L02_R02', 'L02_R01',
    'L01_R04', 'L01_R03', 'L01_R02', 'L01_R01']);
});

test('a path carries over to another cylinder by the offset between their centres', () => {
  const scene = clamp.base_scene;
  const other = scene.circles.find(c => c.id !== scene.target_id);
  const moved = translatePaths(clamp.blades, scene.target.center, other.center);
  const [a, b] = [clamp.blades[0].waypoints[3].tcp, moved[0].waypoints[3].tcp];
  assert.ok(Math.abs(b[0] - a[0] - (other.center[0] - scene.target.center[0])) < 1e-12);
  assert.equal(moved[0].waypoints[3].angle, clamp.blades[0].waypoints[3].angle);
  const aimed = retarget(scene, other.id);
  assert.deepEqual(aimed.frame.origin, other.center);
});

test('the batch goes in order and stops at the first cylinder it cannot take out', () => {
  const scene = restingScene(clamp.base_scene);
  const report = runBatch(scene, clamp.blades, scene.target_id, clamp.options);
  const plan = numberCylinders(scene);
  assert.deepEqual(report.attempts.map(a => a.id), plan.slice(0, report.attempts.length).map(n => n.id));
  assert.ok(report.attempts[0].result.success, 'the demo cylinder comes out first');
  if (report.stoppedAt) {
    assert.equal(report.extracted, report.attempts.length - 1);
    assert.equal(report.attempts.at(-1).result.success, false);
    assert.ok(report.attempts.slice(0, -1).every(a => a.result.success));
  }
  // each later attempt starts without the cylinders already taken out
  report.attempts.forEach((attempt, index) => assert.equal(attempt.scene.circles.length, scene.circles.length - index));
});

test('a carried-over drawn path lifts further for a deeper cylinder without touching the way down', async () => {
  const { stretchLift } = await import('../lib/lab-batch.ts');
  const w = (x, z) => ({ stage: 'draw', label: '', tcp: [x, z], angle: 0, duration: .2 });
  const path = [{ flip: false, waypoints: [w(0, .1), w(0, 0), w(0, -.05), w(0, .05), w(0, .2), w(.1, .2)] }];
  const [stretched] = stretchLift(path, [0, 1], [0, 0], .1);
  const z = stretched.waypoints.map(p => +p.tcp[1].toFixed(6));
  assert.deepEqual(z.slice(0, 3), [.1, 0, -.05]); // down to the deepest point: unchanged
  assert.ok(Math.abs(z[3] - (.05 + .1 * (.1 / .25))) < 1e-9); // part way up: part of the extra lift
  assert.deepEqual(z.slice(4), [.3, .3]); // from the peak on: the full extra lift
  assert.equal(stretchLift(path, [0, 1], [0, 0], -.1), path); // a shallower cylinder keeps the path
});

test('the cylinder mass option changes how much force lifting takes', async () => {
  const { simulateScoop, restingScene } = await import('../lib/cylinder-sim.ts');
  const scene = restingScene(clamp.base_scene);
  const light = simulateScoop(scene, clamp.blades, undefined, { ...clamp.options, cylinderMassKg: .2 });
  const heavy = simulateScoop(scene, clamp.blades, undefined, { ...clamp.options, cylinderMassKg: 6 });
  assert.ok(light.success, light.reason);
  assert.ok(!heavy.success || heavy.liftM < light.liftM - .01, `a 6 kg bundle is harder to lift (${heavy.liftM} vs ${light.liftM})`);
});
