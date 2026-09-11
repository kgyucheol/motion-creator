import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { RobotScene, canRotateSelection, jointControls, jointForRing, COMBINED_JOINTS } from '../lib/robot-scene.ts';
import { rotatedGroupTargets, incrementRotation, quaternionFromDegrees, eulerDegrees, canMirrorSelection, translatedTargets } from '../lib/pose-transforms.ts';
import { BODY_GROUPS, allNodes, nodeMembers, selectMembers, selectionState, controlSelection, controlKey, visibleTreeHandles, expandVirtualControls } from '../lib/body-groups.ts';

function fixture() {
  const viewer = Object.create(RobotScene.prototype);
  viewer.state = { handles: { left_hand: { position: [.3, .2, .8] }, right_hand: { position: [.3, -.2, .8] } } };
  viewer.members = ['left_hand', 'right_hand'];
  viewer.selected = 'left_hand';
  viewer.editable = true;
  viewer.selectionLocked = false;
  viewer.gizmo = { dragging: false, axis: null };
  viewer.camera = new THREE.PerspectiveCamera(38, 1.5, .01, 100);
  viewer.camera.position.set(2, -3, 1.8);
  viewer.orbit = { target: new THREE.Vector3(0, 0, .7), update() {} };
  return viewer;
}

test('group focus centers on both hands and preserves viewing direction', () => {
  const viewer = fixture();
  const before = viewer.camera.position.clone().sub(viewer.orbit.target).normalize();
  viewer.focusSelection();
  assert.deepEqual(viewer.orbit.target.toArray(), [.3, 0, .8]);
  const after = viewer.camera.position.clone().sub(viewer.orbit.target).normalize();
  assert.ok(before.distanceTo(after) < 1e-10);
  assert.ok(viewer.camera.position.distanceTo(viewer.orbit.target) >= .55);
});

test('single focus centers on the selected hand', () => {
  const viewer = fixture();
  viewer.members = ['right_hand'];
  viewer.focusSelection();
  assert.deepEqual(viewer.orbit.target.toArray(), [.3, -.2, .8]);
});

test('group rotation preserves separation and rotates around the selection center', () => {
  const initial = quaternionFromDegrees([25, 15, 10]);
  const after = incrementRotation(initial, 2, 90, 'world');
  const result = rotatedGroupTargets({ left_hand: [1, 2, 1], right_hand: [1, 0, 1] },
    { left_hand: initial, right_hand: initial }, [1, 1, 1], initial, after, []);
  assert.ok(new THREE.Vector3(...result.targets.left_hand).distanceTo(new THREE.Vector3(0, 1, 1)) < 1e-10);
  assert.ok(new THREE.Vector3(...result.targets.right_hand).distanceTo(new THREE.Vector3(2, 1, 1)) < 1e-10);
  for (const q of Object.values(result.orientations)) {
    assert.ok(Math.abs(new THREE.Quaternion(...q).dot(new THREE.Quaternion(...after))) > 1-1e-10);
  }
});

test('a position-pinned hand has a rotation target without a conflicting position target', () => {
  const result = rotatedGroupTargets({ left_hand: [1, 2, 1] }, { left_hand: [0, 0, 0, 1] },
    [1, 2, 1], [0, 0, 0, 1], quaternionFromDegrees([15, 0, 0]), ['left_hand']);
  assert.deepEqual(result.targets, {});
  assert.ok(Math.abs(eulerDegrees(result.orientations.left_hand)[0]-15) < 1e-10);
});

test('local and world rotation increments use different axes for a rotated hand', () => {
  const initial = quaternionFromDegrees([0, 0, 90]);
  const local = incrementRotation(initial, 0, 20, 'local');
  const world = incrementRotation(initial, 1, 20, 'world');
  const wrong = incrementRotation(initial, 0, 20, 'world');
  assert.ok(Math.abs(new THREE.Quaternion(...local).dot(new THREE.Quaternion(...world))) > 1-1e-10);
  assert.ok(Math.abs(new THREE.Quaternion(...local).dot(new THREE.Quaternion(...wrong))) < .99);
});

test('a new wrist axis uses a single ring at its physical anchor', () => {
  const viewer = fixture();
  const name = 'left_wrist_pitch_joint';
  viewer.state.handles[name] = { position: [.2, .3, .8], quaternion: [0, 0, 0, 1] };
  viewer.state.hinges = { [name]: { position: [.2, .3, .8], axis_world: [0, 1, 0] } };
  viewer.markers = {}; viewer.labels = {};
  viewer.pivot = new THREE.Object3D();
  viewer.markerVisible = true;
  viewer.transformMode = 'rotate';
  viewer.gizmo.setSpace = (space) => { viewer.gizmo.space = space; };
  viewer.gizmo.attach = (object) => { viewer.gizmo.object = object; };
  viewer.gizmo.detach = () => { viewer.gizmo.object = null; };
  viewer.select(name, ['left_foot', 'right_foot']);
  assert.equal(canRotateSelection([name]), true);
  assert.equal(canRotateSelection([name, 'right_wrist_pitch_joint']), false);
  assert.equal(viewer.gizmo.object, viewer.pivot);
  assert.equal(viewer.gizmo.space, 'local');
  assert.equal(viewer.gizmo.showX, false);
  assert.equal(viewer.gizmo.showY, false);
  assert.equal(viewer.gizmo.showZ, true);
  assert.deepEqual(viewer.pivot.position.toArray(), [.2, .3, .8]);
  assert.ok(new THREE.Vector3(0, 0, 1).applyQuaternion(viewer.pivot.quaternion).distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-10);
});

test('Alt picking cycles through coincident joint markers and ignores hidden handles', () => {
  const viewer = fixture();
  viewer.renderer = { domElement: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }) } };
  viewer.ray = new THREE.Raycaster(); viewer.pointer = new THREE.Vector2();
  viewer.camera.position.set(0, 0, 2); viewer.camera.lookAt(0, 0, 0); viewer.camera.updateMatrixWorld();
  viewer.markers = Object.fromEntries(['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_hand'].map(name => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(.03), new THREE.MeshBasicMaterial());
    mesh.name = name; mesh.visible = name !== 'left_hand'; mesh.updateMatrixWorld();
    return [name, mesh];
  }));
  const event = { clientX: 50, clientY: 50, altKey: true };
  const visited = [];
  for (let i = 0; i < 3; i++) { viewer.selected = viewer.pick(event); visited.push(viewer.selected); }
  assert.equal(new Set(visited).size, 3);
  assert.ok(!visited.includes('left_hand'));
  assert.equal(viewer.pick(event), visited[0]);
  Object.values(viewer.markers).forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
});

test('left and right hip axes collapse separately into three-axis controls', () => {
  const names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint'];
  assert.deepEqual(jointControls(names), ['left_hip', 'left_knee_joint', 'right_hip', 'right_knee_joint']);
  assert.ok(canRotateSelection(['left_hip', 'right_hip']));
  const viewer = fixture();
  viewer.state.handles.left_hip = { position: [0, .12, .65], quaternion: [0, 0, 0, 1] };
  viewer.state.hinges = {};
  viewer.markers = Object.fromEntries([...names, 'left_hip', 'right_hip'].map(name => [name, new THREE.Mesh(new THREE.SphereGeometry(.02), new THREE.MeshBasicMaterial())]));
  viewer.labels = {};
  viewer.pivot = new THREE.Object3D(); viewer.markerVisible = true; viewer.handleLayer = 'joints';
  viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = () => {};
  viewer.gizmo.attach = () => {};
  viewer.gizmo.detach = () => {};
  viewer.select('left_hip', []);
  assert.deepEqual(viewer.pivot.position.toArray(), [0, .12, .65]);
  assert.ok(viewer.gizmo.showX && viewer.gizmo.showY && viewer.gizmo.showZ);
  assert.ok(viewer.markers.left_hip.visible && viewer.markers.right_hip.visible);
  assert.equal(viewer.markers.left_hip_yaw_joint.visible, false);
  assert.equal(viewer.markers.right_hip_roll_joint.visible, false);
  Object.values(viewer.markers).forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
});

test('29 motors remain addressable through 21 joint controls', () => {
  const names = [...Object.values(COMBINED_JOINTS).flat(), 'left_knee_joint', 'right_knee_joint',
    ...['left', 'right'].flatMap(side => ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'].map(part => `${side}_${part}_joint`))];
  assert.equal(new Set(names).size, 29);
  const controls = jointControls(names);
  assert.equal(controls.length, 21);
  for (const key of Object.keys(COMBINED_JOINTS)) assert.ok(controls.includes(key));
  assert.ok(canRotateSelection(['waist']));
  assert.ok(canRotateSelection(['left_ankle']));
  assert.equal(canRotateSelection(['left_ankle', 'right_ankle']), false);
});

test('ankle rings follow real axes and send only the corresponding motor angle', () => {
  const viewer = fixture();
  const initialFrame = new THREE.Quaternion().setFromEuler(new THREE.Euler(.2, -.35, .6));
  const x = new THREE.Vector3(1, 0, 0).applyQuaternion(initialFrame);
  const y = new THREE.Vector3(0, 1, 0).applyQuaternion(initialFrame);
  viewer.state.handles.left_ankle = { position: [.1, .2, .05], quaternion: [0, 0, 0, 1] };
  viewer.state.hinges = {
    left_ankle_roll_joint: { axis_world: x.toArray(), angle: .1, limits: [-.25, .25] },
    left_ankle_pitch_joint: { axis_world: y.toArray(), angle: -.1, limits: [-.8, .5] },
  };
  viewer.markers = {}; viewer.labels = {}; viewer.pivot = new THREE.Object3D();
  viewer.markerVisible = true; viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = space => { viewer.gizmo.space = space; };
  viewer.gizmo.attach = object => { viewer.gizmo.object = object; };
  viewer.gizmo.detach = () => {};
  viewer.select('left_ankle', []);
  assert.equal(viewer.gizmo.space, 'local');
  assert.ok(viewer.gizmo.showX && viewer.gizmo.showY);
  assert.equal(viewer.gizmo.showZ, false);
  assert.deepEqual(viewer.pivot.position.toArray(), [.1, .2, .05]);
  assert.ok(new THREE.Vector3(1, 0, 0).applyQuaternion(viewer.pivot.quaternion).distanceTo(x) < 1e-10);
  assert.ok(new THREE.Vector3(0, 1, 0).applyQuaternion(viewer.pivot.quaternion).distanceTo(y) < 1e-10);
  for (const axis of ['Z', 'E', 'XYZE', null]) assert.equal(jointForRing('left_ankle', axis), null);
  for (const [axis, part, component] of [['X', 'roll', 'x'], ['Y', 'pitch', 'y']]) {
    const ring = jointForRing('left_ankle', axis);
    assert.deepEqual(ring, { key: `left_ankle_${part}_joint`, component });
    const joint = viewer.state.hinges[ring.key];
    viewer.hingeDrag = { quaternion: initialFrame.clone(), angle: joint.angle, limits: joint.limits, delta: 0, lastTwist: 0, ...ring };
    const calls = [];
    viewer.callbacks = { jointAngle: (name, value) => calls.push([name, value]) };
    const direction = axis === 'X' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    viewer.pivot.quaternion.copy(initialFrame).multiply(new THREE.Quaternion().setFromAxisAngle(direction, .1));
    viewer.applyHingeDrag(); viewer.applyHingeDrag();
    assert.equal(calls[0][0], ring.key);
    assert.ok(Math.abs(calls[0][1] - joint.angle - .1) < 1e-10);
    assert.ok(Math.abs(calls[1][1] - calls[0][1]) < 1e-10);
    viewer.pivot.quaternion.copy(initialFrame).multiply(new THREE.Quaternion().setFromAxisAngle(direction, 1));
    viewer.applyHingeDrag();
    assert.equal(calls[2][1], joint.limits[1]);
  }
});

test('anatomical tree covers all 29 motors exactly once and includes hands and support points', () => {
  const leaves = BODY_GROUPS.flatMap(nodeMembers);
  assert.equal(leaves.length, new Set(leaves).size);
  const motorNames = [...Object.values(COMBINED_JOINTS).flat(), 'left_knee_joint', 'right_knee_joint',
    ...['left', 'right'].flatMap(side => ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'].map(part => `${side}_${part}_joint`))];
  assert.deepEqual(leaves.filter(key => key.endsWith('_joint')).sort(), motorNames.sort());
  assert.deepEqual(leaves.filter(key => !key.endsWith('_joint')).sort(), ['pelvis','left_hand','right_hand','left_foot','right_foot'].sort());
  const nodes = allNodes();
  const hands = nodes.find(node => node.id === 'hands');
  const left = nodes.find(node => node.id === 'left_hands');
  assert.equal(nodeMembers(hands).length, 8);
  assert.deepEqual(nodeMembers(left), ['left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint','left_hand']);
});

test('group selection, single-child replacement, partial state and Shift toggle', () => {
  const hands = allNodes().find(node => node.id === 'hands');
  const group = nodeMembers(hands);
  let selected = selectMembers(['pelvis'], group, false);
  assert.deepEqual(selected, group);
  assert.equal(selectionState(hands, selected).all, true);
  selected = selectMembers(selected, ['left_wrist_pitch_joint'], false);
  assert.deepEqual(selected, ['left_wrist_pitch_joint']);
  assert.deepEqual(selectionState(hands, selected), { count: 1, total: 8, all: false, partial: true });
  selected = selectMembers(selected, ['pelvis'], true);
  selected = selectMembers(selected, group, true);
  assert.equal(new Set(selected).size, 9);
  selected = selectMembers(selected, group, true);
  assert.deepEqual(selected, ['pelvis']);
});

test('full groups preserve integrated controls while an individual child keeps its own axis', () => {
  const nodes = allNodes();
  const hands = nodeMembers(nodes.find(node => node.id === 'hands'));
  assert.deepEqual(controlSelection(hands), ['left_hand', 'right_hand']);
  assert.equal(controlKey(hands, 'left_wrist_pitch_joint'), 'left_hand');
  const hips = nodeMembers(nodes.find(node => node.id === 'left_hips'));
  assert.deepEqual(controlSelection(hips), ['left_hip']);
  assert.deepEqual(controlSelection(['left_hip_roll_joint']), ['left_hip_roll_joint']);
  const foot = nodeMembers(nodes.find(node => node.id === 'left_feet'));
  assert.deepEqual(controlSelection(foot), ['left_ankle']);
  assert.equal(jointForRing(controlKey(foot, foot[0]), 'Y').key, 'left_ankle_pitch_joint');
  const legacy = expandVirtualControls(['waist','left_hip','left_hand']);
  assert.equal(legacy.length, 7);
  assert.deepEqual(controlSelection(legacy), ['waist','left_hip','left_hand']);
});

test('accordion visibility changes do not alter selection', () => {
  const selected = ['left_wrist_pitch_joint'];
  const closed = visibleTreeHandles([]);
  const expanded = visibleTreeHandles(['hands', 'left_hands']);
  assert.ok(closed.includes('left_hand'));
  assert.equal(closed.includes('left_wrist_pitch_joint'), false);
  assert.ok(expanded.includes('left_hand') && expanded.includes('left_wrist_pitch_joint'));
  assert.deepEqual(selected, ['left_wrist_pitch_joint']);
});

test('selecting a hand group highlights every leaf and selecting a child restores a single-axis gizmo', () => {
  const viewer = fixture();
  const group = allNodes().find(node => node.id === 'left_hands');
  const leaves = nodeMembers(group);
  viewer.state.hinges = {};
  viewer.markers = {}; viewer.labels = {};
  leaves.forEach((key, i) => {
    viewer.state.handles[key] = { position: [.2+i*.01, .2, .8], quaternion: [0, 0, 0, 1] };
    viewer.markers[key] = new THREE.Mesh(new THREE.SphereGeometry(.02), new THREE.MeshBasicMaterial());
    if (key.endsWith('_joint')) viewer.state.hinges[key] = { position: viewer.state.handles[key].position, axis_world: [1,0,0] };
  });
  viewer.pivot = new THREE.Object3D(); viewer.markerVisible = true; viewer.visibleHandles = ['left_hand'];
  viewer.transformMode = 'rotate'; viewer.space = 'world';
  viewer.gizmo.setSpace = () => {}; viewer.gizmo.attach = () => {}; viewer.gizmo.detach = () => {};
  viewer.select(leaves[0], [], leaves);
  assert.ok(viewer.gizmo.showX && viewer.gizmo.showY && viewer.gizmo.showZ);
  assert.deepEqual(viewer.pivot.position.toArray(), viewer.state.handles.left_hand.position);
  for (const key of leaves) {
    assert.equal(viewer.markers[key].visible, true);
    assert.equal(viewer.markers[key].material.color.getHexString(), '80f2c7');
  }
  viewer.select('left_wrist_pitch_joint', [], ['left_wrist_pitch_joint']);
  assert.equal(viewer.gizmo.showX, false); assert.equal(viewer.gizmo.showY, false); assert.equal(viewer.gizmo.showZ, true);
  assert.deepEqual(viewer.pivot.position.toArray(), viewer.state.handles.left_wrist_pitch_joint.position);
  Object.values(viewer.markers).forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
});

test('mirror translation opens and closes pairs from either active side without changing forward/up displacement', () => {
  const positions = { left_hand: [.2,.25,.8], right_hand: [.2,-.25,.8] };
  assert.ok(canMirrorSelection(Object.keys(positions)));
  for (const keys of [['left_hand'], ['left_hand','right_foot'], ['left_hand','right_hand','pelvis']]) assert.equal(canMirrorSelection(keys), false);
  const open = translatedTargets(positions, [.1,.05,.03], { active: 'left_hand', rootQuaternion: [0,0,0,1] });
  assert.ok(Math.abs(open.left_hand[1] - open.right_hand[1] - .6) < 1e-10);
  for (const p of Object.values(open)) { assert.ok(Math.abs(p[0]-.3)<1e-10); assert.ok(Math.abs(p[2]-.83)<1e-10); }
  const rightOpen = translatedTargets(positions, [0,-.05,0], { active: 'right_hand', rootQuaternion: [0,0,0,1] });
  assert.ok(Math.abs(rightOpen.left_hand[1] - rightOpen.right_hand[1] - .6) < 1e-10);
  const closed = translatedTargets(positions, [0,-.05,0], { active: 'left_hand', rootQuaternion: [0,0,0,1] });
  assert.ok(Math.abs(closed.left_hand[1] - closed.right_hand[1] - .4) < 1e-10);
  const normal = translatedTargets(positions, [0,.05,0]);
  assert.ok(Math.abs(normal.left_hand[1] - normal.right_hand[1] - .5) < 1e-10);
  assert.deepEqual(positions.left_hand, [.2,.25,.8]);
});

test('mirror follows the robot sagittal plane when the robot is turned', () => {
  const root = quaternionFromDegrees([0,0,90]);
  const positions = { left_hand: [-.25,.2,.8], right_hand: [.25,.2,.8] };
  const result = translatedTargets(positions, [-.05,.1,0], { active: 'left_hand', rootQuaternion: root });
  assert.ok(Math.abs(result.left_hand[0]+.3) < 1e-10);
  assert.ok(Math.abs(result.right_hand[0]-.3) < 1e-10);
  assert.ok(Math.abs(result.left_hand[1]-.3) < 1e-10);
  assert.ok(Math.abs(result.right_hand[1]-.3) < 1e-10);
});

test('Ctrl+Z invokes pose history while text input, dragging and task mode retain their behavior', () => {
  const viewer = fixture();
  const calls = [];
  viewer.callbacks = { history: redo => calls.push(redo) };
  let prevented = 0;
  const event = { code:'KeyZ', ctrlKey:true, metaKey:false, altKey:false, shiftKey:false, target:null,
    preventDefault() { prevented++; } };
  viewer.handleKeyDown(event);
  viewer.handleKeyDown({ ...event, shiftKey:true });
  assert.deepEqual(calls, [false,true]);
  assert.equal(prevented, 2);
  viewer.handleKeyDown({ ...event, target:{ closest: () => ({}) } });
  viewer.gizmo.dragging = true; viewer.handleKeyDown(event); viewer.gizmo.dragging = false;
  viewer.keyboardEnabled = false; viewer.handleKeyDown(event); viewer.keyboardEnabled = true;
  viewer.editable = false; viewer.handleKeyDown(event);
  assert.deepEqual(calls, [false,true]);
  assert.equal(prevented, 2);
});

test('mirror gizmo anchors at the active member while F still frames both sides', () => {
  const viewer = fixture();
  for (const h of Object.values(viewer.state.handles)) h.quaternion = [0,0,0,1];
  viewer.markers = {}; viewer.labels = {}; viewer.pivot = new THREE.Object3D();
  viewer.markerVisible = true; viewer.transformMode = 'translate'; viewer.mirrorTranslation = true;
  viewer.gizmo.setSpace = () => {}; viewer.gizmo.attach = () => {}; viewer.gizmo.detach = () => {};
  viewer.select('right_hand', [], ['left_hand','right_hand']);
  assert.deepEqual(viewer.pivot.position.toArray(), [.3,-.2,.8]);
  viewer.focusSelection();
  assert.deepEqual(viewer.orbit.target.toArray(), [.3,0,.8]);
  viewer.setMirrorTranslation(false);
  assert.deepEqual(viewer.pivot.position.toArray(), [.3,0,.8]);
});
