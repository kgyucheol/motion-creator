import test from 'node:test';
import assert from 'node:assert/strict';
import { simulateScoop, softLayout } from '../lib/cylinder-sim.ts';

// One cylinder resting on a 30 deg ramp; a parked blade far away keeps the simulator busy for 2 s.
const ramp = Math.PI / 6;
const scene = {
  target_id: 'c', target: { id: 'c', center: [0, .0675 / Math.cos(ramp) + .01], radius: .0675 },
  circles: [{ id: 'c', name: 'c', center: [0, .0675 / Math.cos(ramp) + .01], radius: .0675, mass_kg: 1, friction: .7 }],
  rectangles: [], walls: [{ id: 'ramp', center: [0, -.01 / Math.cos(ramp)], half: [1, .01], angle: -ramp, friction: .7 }],
  frame: { origin: [0, .1], far: [1, 0], up: [0, 1], rim_up_m: 0 },
};
const parked = [{ stage: 'lift', label: 'a', tcp: [2, 2], angle: 0, duration: .1 }, { stage: 'lift', label: 'b', tcp: [2, 2], angle: 0, duration: 1 }];
const travel = options => {
  const result = simulateScoop(scene, parked, undefined, { settleSeconds: .5, holdSeconds: .5, ...options });
  const [first, last] = [result.frames[0].bodies.c, result.frames.at(-1).bodies.c];
  return Math.hypot(last.x - first.x, last.z - first.z);
};

test('rolling resistance and friction overrides change how a cylinder leaves a ramp', () => {
  const free = travel({});
  const damped = travel({ friction: { rolling: 50 } });
  assert.ok(free > .5, `a free cylinder rolls down (${free})`);
  assert.ok(damped < free / 3, `rolling resistance holds it back (${damped} vs ${free})`);
  const slippery = travel({ friction: { rolling: 50, cylinder: .05, wall: .05 } });
  assert.ok(slippery > damped * 3, `low friction lets the damped cylinder slide (${slippery} vs ${damped})`);
});

test('a smooth soft cylinder keeps a round outline at its radius', () => {
  const layout = softLayout(.0675, 'smooth');
  assert.equal(layout.nodes, 48);
  assert.ok(Math.abs(layout.ring + layout.nodeRadius - .0675) < 1e-12);
  const result = simulateScoop({ ...scene, walls: [{ ...scene.walls[0], angle: 0, center: [0, -.01] }], target: { ...scene.target, center: [0, .0675] },
    circles: [{ ...scene.circles[0], center: [0, .0675] }] }, parked, undefined,
  { settleSeconds: .3, holdSeconds: .3, soft: { mmPer10N: 5, scope: 'all', shape: 'smooth' } });
  const outline = result.frames.at(-1).shapes.c;
  assert.equal(outline.length, 48);
  const [hx, hz] = [result.frames.at(-1).bodies.c.x, result.frames.at(-1).bodies.c.z];
  const radii = outline.map(([x, z]) => Math.hypot(x - hx, z - hz));
  assert.ok(Math.max(...radii) < .0675 + .003 && Math.min(...radii) > .0675 - .012, `outline radii ${Math.min(...radii)}..${Math.max(...radii)}`);
});

test('every lab demo still lifts the cylinder out (re-run scripts/cylinder_lab_demos.py after planner changes)', async () => {
  const { readFileSync } = await import('node:fs');
  const { restingScene } = await import('../lib/cylinder-sim.ts');
  const demos = JSON.parse(readFileSync(new URL('../lib/cylinder-demos.json', import.meta.url), 'utf8'));
  assert.ok(demos.length >= 3);
  for (const demo of demos) {
    const result = simulateScoop(restingScene(demo.base_scene), demo.blades, undefined, demo.options);
    assert.ok(result.success, `${demo.id}: ${result.reason}`);
  }
});
