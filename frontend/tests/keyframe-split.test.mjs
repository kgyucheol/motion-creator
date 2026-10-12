import test from 'node:test';
import assert from 'node:assert/strict';
import { clipSampleAtTime, splitMotionClip } from '../lib/keyframes.ts';

const clip = (count, duration) => ({ name: 'clip', duration, pins: [], qpos: [0],
  samples: Array.from({ length: count }, (_, i) => [i]) });

test('splitting keeps the boundary sample and the total duration', () => {
  const { keyframes } = splitMotionClip([clip(11, 10)], 0, 4);
  assert.equal(keyframes.length, 2);
  assert.deepEqual(keyframes[0].samples.at(-1), keyframes[1].samples[0]);
  assert.equal(keyframes[0].samples.length + keyframes[1].samples.length, 12);
  assert.ok(Math.abs(keyframes[0].duration + keyframes[1].duration - 10) < 1e-9);
  assert.ok(Math.abs(keyframes[0].duration - 4) < 1e-9);
  assert.deepEqual(keyframes[1].qpos, [4]);
});

test('split rejects the first and last samples', () => {
  assert.throws(() => splitMotionClip([clip(11, 10)], 0, 0));
  assert.throws(() => splitMotionClip([clip(11, 10)], 0, 10));
});

test('time maps to a sample inside the selected clip', () => {
  const frames = [clip(11, 10), clip(21, 4)];
  assert.equal(clipSampleAtTime(frames, 0, 5), 5);
  assert.equal(clipSampleAtTime(frames, 1, 12), 10);
  assert.equal(clipSampleAtTime(frames, 0, 0), 1);
});

test('merging undoes a split', async () => {
  const { mergeMotionClips } = await import('../lib/keyframes.ts');
  const original = clip(11, 10);
  const { keyframes } = splitMotionClip([original], 0, 4);
  const merged = mergeMotionClips(keyframes, 0).keyframes;
  assert.equal(merged.length, 1);
  assert.deepEqual(merged[0].samples, original.samples);
  assert.ok(Math.abs(merged[0].duration - 10) < 1e-9);
});
