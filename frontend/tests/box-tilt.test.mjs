import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { boxPitchDeg, tiltScene } from '../lib/box-tilt.ts';
import { restingScene, isOutsideBox } from '../lib/cylinder-sim.ts';

const demo = JSON.parse(readFileSync(new URL('../lib/cylinder-demos.json', import.meta.url), 'utf8'))[0];

test('the project box reads as pitch -50 deg', () => {
  assert.ok(Math.abs(boxPitchDeg(demo.base_scene) + 50) < .1);
});

test('tilting turns the box and its cylinders rigidly and keeps the target frame on the box', () => {
  const base = demo.base_scene;
  const { scene, delta } = tiltScene(base, -30);
  assert.ok(Math.abs(delta + 20 * Math.PI / 180) < 1e-9);
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  for (let i = 1; i < base.circles.length; i++) {
    assert.ok(Math.abs(d(base.circles[0].center, base.circles[i].center) - d(scene.circles[0].center, scene.circles[i].center)) < 1e-9);
  }
  assert.ok(Math.abs(d(base.box.center, base.target.center) - d(scene.box.center, scene.target.center)) < 1e-9);
  assert.equal(scene.frame.rim_up_m, base.frame.rim_up_m);
  assert.ok(Math.abs(scene.frame.up[1] - Math.cos(30 * Math.PI / 180)) < 1e-9);
  assert.ok(scene.frame.far[0] * scene.toward_robot[0] + scene.frame.far[1] * scene.toward_robot[1] < 0);
  assert.ok(Math.abs(boxPitchDeg(scene) + 30) < 1e-9);
});

test('a steeper or flatter box still holds its cylinders after settling', () => {
  for (const pitch of [-20, -70]) {
    const rest = restingScene(tiltScene(demo.base_scene, pitch).scene);
    const escaped = rest.circles.filter(c => isOutsideBox(rest, c.center, c.radius, 0, 1));
    assert.equal(escaped.length, 0, `pitch ${pitch}: ${escaped.map(c => c.name)}`);
  }
});
