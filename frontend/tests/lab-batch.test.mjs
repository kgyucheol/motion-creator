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
