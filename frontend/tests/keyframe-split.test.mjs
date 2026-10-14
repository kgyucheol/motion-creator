import test from 'node:test';
import assert from 'node:assert/strict';
import { clipSampleAtTime, middleSample, splitMotionClip } from '../lib/keyframes.ts';

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
  assert.equal(clipSampleAtTime(frames, 0, 12), null);
  assert.equal(clipSampleAtTime(frames, 1, 11, 11), 1);
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

test('converting a clip yields poses whose travel time matches the clip', async () => {
  const { convertClipToPoses } = await import('../lib/keyframes.ts');
  const { keyframes } = convertClipToPoses([clip(11, 10)], 0, 3, ['left_foot']);
  assert.equal(keyframes.length, 3);
  assert.deepEqual(keyframes.map(frame => frame.qpos[0]), [0, 5, 10]);
  assert.ok(keyframes.every(frame => !frame.samples && frame.pins[0] === 'left_foot'));
  assert.ok(Math.abs(keyframes[1].duration + keyframes[2].duration - 10) < 1e-9);
});

test('a converted clip after a split neighbour keeps the minimum entry time', async () => {
  const { convertClipToPoses } = await import('../lib/keyframes.ts');
  const { keyframes } = splitMotionClip([clip(11, 10)], 0, 4);
  const converted = convertClipToPoses(keyframes, 1, 2, []).keyframes;
  assert.equal(converted.length, 3);
  assert.equal(converted[1].duration, .1);
  assert.deepEqual(converted[1].qpos, [4]);
});

test('the default cut is the middle of the clip', () => {
  assert.equal(middleSample(11), 5);
  assert.equal(middleSample(564), 281);
  assert.equal(middleSample(3), 1);
  const { keyframes } = splitMotionClip([clip(11, 10)], 0, middleSample(11));
  assert.equal(keyframes[0].duration, 5);
});

const pose = (q, duration) => ({ name: `p${q}`, duration, pins: [], qpos: [q] });

test('a card shows the time to the next card', async () => {
  const { outgoingSeconds, setOutgoingSeconds } = await import('../lib/keyframes.ts');
  const frames = [pose(0, 9), pose(1, 2), pose(2, 3)];
  assert.deepEqual(outgoingSeconds(frames, 0), { kind: 'pose', seconds: 2 });
  assert.deepEqual(outgoingSeconds(frames, 2), { kind: 'none' });
  assert.equal(setOutgoingSeconds(frames, 1, 4)[2].duration, 4);
  assert.equal(setOutgoingSeconds(frames, 2, 4), frames);
});

test('reordering and deleting keep each card\'s own time to its next card', async () => {
  const { moveKeyframe, removeKeyframe, outgoingSeconds } = await import('../lib/keyframes.ts');
  const frames = [pose(0, 9), pose(1, 2), pose(2, 3), pose(3, 5)];   // outgoing: A 2, B 3, C 5, D none
  const moved = moveKeyframe(frames, 1, 1);                          // A C B D
  assert.deepEqual(moved.map(f => f.qpos[0]), [0, 2, 1, 3]);
  assert.deepEqual([0, 1, 2].map(i => outgoingSeconds(moved, i).seconds), [2, 5, 3]);
  const removed = removeKeyframe(frames, 1);                         // A C D
  assert.deepEqual([0, 1].map(i => outgoingSeconds(removed, i).seconds), [2, 5]);
});

test('clip entry times belong to the previous card and joined clips need none', async () => {
  const { outgoingSeconds, setOutgoingSeconds } = await import('../lib/keyframes.ts');
  const { keyframes } = splitMotionClip([clip(11, 10)], 0, 4);
  assert.deepEqual(outgoingSeconds(keyframes, 0), { kind: 'joined' });
  const frames = [pose(99, 1), keyframes[0]];
  assert.deepEqual(outgoingSeconds(frames, 0), { kind: 'clip', seconds: undefined });
  assert.equal(setOutgoingSeconds(frames, 0, 2.5)[1].transition, 2.5);
});

test('start times follow arrival semantics', async () => {
  const { keyframeStartTimes } = await import('../lib/keyframes.ts');
  const starts = keyframeStartTimes([pose(0, 1), pose(1, 2), pose(2, 3)]);
  assert.deepEqual(starts.map(t => Math.round(t * 100) / 100), [0, 2, 5]);
  const { keyframes } = splitMotionClip([clip(11, 10)], 0, 4);
  assert.deepEqual(keyframeStartTimes(keyframes), [0, 4]);
});

test('a pose added after a clip starts from the clip end and keeps the following time', async () => {
  const { insertPoseAfterClip, outgoingSeconds } = await import('../lib/keyframes.ts');
  const frames = [...splitMotionClip([clip(11, 10)], 0, 4).keyframes, pose(7, 3)];
  const result = insertPoseAfterClip(frames, 1, ['left_foot']);
  assert.equal(result.keyframes.length, 4);
  assert.deepEqual(result.keyframes[2].qpos, [10]);
  assert.equal(outgoingSeconds(result.keyframes, 1).seconds, .1);
  assert.equal(outgoingSeconds(result.keyframes, 2).seconds, 3);
});

const grasp = (window) => ({ object_id: 'cube', reference_pose: { position: [.4, 0, .8], quaternion_xyzw: [0, 0, 0, 1] },
  hands: ['left', 'right'], window });
const WINDOW = { ramp_in_start: 10, hold_start: 20, hold_end: 40, ramp_out_end: 50 };

test('splitting a dynamic clip shifts the window and keeps it on both parts when it spans the cut', async () => {
  const { splitMotionClip } = await import('../lib/keyframes.ts');
  const { keyframes } = splitMotionClip([{ ...clip(61, 6), dynamic: grasp(WINDOW) }], 0, 30);
  assert.deepEqual(keyframes[0].dynamic.window, WINDOW);
  assert.deepEqual(keyframes[1].dynamic.window, { ramp_in_start: -20, hold_start: -10, hold_end: 10, ramp_out_end: 20 });
});

test('a part the window does not touch loses the binding', async () => {
  const { splitMotionClip } = await import('../lib/keyframes.ts');
  const early = splitMotionClip([{ ...clip(61, 6), dynamic: grasp(WINDOW) }], 0, 55).keyframes;
  assert.equal(early[1].dynamic, undefined);
  assert.ok(early[0].dynamic);
  const late = splitMotionClip([{ ...clip(61, 6), dynamic: grasp(WINDOW) }], 0, 5).keyframes;
  assert.equal(late[0].dynamic, undefined);
  assert.deepEqual(late[1].dynamic.window, { ramp_in_start: 5, hold_start: 15, hold_end: 35, ramp_out_end: 45 });
});

test('merging parts of one dynamic clip restores the original binding', async () => {
  const { splitMotionClip, mergeMotionClips } = await import('../lib/keyframes.ts');
  const original = { ...clip(61, 6), dynamic: grasp(WINDOW) };
  const merged = mergeMotionClips(splitMotionClip([original], 0, 30).keyframes, 0).keyframes[0];
  assert.deepEqual(merged.dynamic, original.dynamic);
  assert.equal(merged.samples.length, 61);
});

test('merging clips with different bindings is refused', async () => {
  const { splitMotionClip, mergeMotionClips } = await import('../lib/keyframes.ts');
  const { keyframes } = splitMotionClip([{ ...clip(61, 6), dynamic: grasp(WINDOW) }], 0, 30);
  keyframes[1].dynamic.object_id = 'other';
  assert.throws(() => mergeMotionClips(keyframes, 0), /동적 파지/);
  const spanning = splitMotionClip([{ ...clip(61, 6), dynamic: grasp(WINDOW) }], 0, 30).keyframes;
  delete spanning[1].dynamic;
  assert.throws(() => mergeMotionClips(spanning, 0), /동적 파지/);
});

test('window helpers keep the four numbers ordered and inside the clip', async () => {
  const { defaultDynamicWindow, setDynamicWindowValue, clipSampleIndexAtTime } = await import('../lib/keyframes.ts');
  const window = defaultDynamicWindow(100, 50);
  assert.deepEqual(window, { ramp_in_start: 40, hold_start: 50, hold_end: 60, ramp_out_end: 70 });
  const edge = defaultDynamicWindow(100, 98);
  assert.ok(edge.ramp_in_start <= edge.hold_start && edge.hold_start <= edge.hold_end && edge.ramp_out_end <= 99);
  const moved = setDynamicWindowValue(WINDOW, 'hold_start', 45);
  assert.deepEqual(moved, { ramp_in_start: 10, hold_start: 45, hold_end: 45, ramp_out_end: 50 });
  assert.equal(clipSampleIndexAtTime([clip(11, 10)], 0, 0), 0);
  assert.equal(clipSampleIndexAtTime([clip(11, 10)], 0, 10), 10);
  assert.equal(clipSampleIndexAtTime([clip(11, 10)], 0, 11), null);
});
