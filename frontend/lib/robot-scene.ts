import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { controlKey, controlSelection } from './body-groups.ts';
import { canMirrorSelection } from './pose-transforms.ts';
import { normalizedObjectSize, normalizedSceneAsset, type ObjectTransformMode, type SceneObject, type SceneObjectPose } from './scene-objects.ts';

export type PoseState = {
  model_id?: string;
  qpos: number[];
  handles: Record<string, { position: number[]; quaternion: number[]; label: string }>;
  geoms: Record<string, { position: number[]; quaternion: number[] }>;
  grip_pads: Record<'left' | 'right', { position: number[]; quaternion: number[]; size: number[] }>;
  com: number[];
  floor_min_mm: number;
  hinges: Record<string, { joint_name: string; angle: number; limits: number[]; axis_world: number[]; position: number[] }>;
};
export type TransformMode = 'translate' | 'rotate';
export const HIP_HANDLES = ['left_hip', 'right_hip'];
export const ANKLE_HANDLES = ['left_ankle', 'right_ankle'];
export const COMBINED_JOINTS: Record<string, string[]> = {
  left_hip: ['left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint'],
  right_hip: ['right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint'],
  waist: ['waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint'],
  left_ankle: ['left_ankle_roll_joint', 'left_ankle_pitch_joint'],
  right_ankle: ['right_ankle_roll_joint', 'right_ankle_pitch_joint'],
};
export const ROTATABLE = ['pelvis', 'left_hand', 'right_hand', 'left_foot', 'right_foot', ...HIP_HANDLES, 'waist'];
export const HINGE_HANDLES = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee'];
export const isJointHandle = (key: string) => key.endsWith('_joint') || Object.hasOwn(COMBINED_JOINTS, key);
export const isCombinedAxis = (key: string) => Object.values(COMBINED_JOINTS).some(names => names.includes(key));
export const jointControls = (names: string[]) => [...new Set(names.map(key => Object.entries(COMBINED_JOINTS).find(([, axes]) => axes.includes(key))?.[0] ?? key))];
export function jointForRing(key: string, axis: string | null): { key: string; component: 'x' | 'y' | 'z' } | null {
  if (ANKLE_HANDLES.includes(key)) return axis === 'X' ? { key: key + '_roll_joint', component: 'x' }
    : axis === 'Y' ? { key: key + '_pitch_joint', component: 'y' } : null;
  return { key, component: 'z' };
}
export function ankleFrame(state: PoseState, key: string) {
  const x = new THREE.Vector3().fromArray(state.hinges[key + '_roll_joint'].axis_world).normalize();
  const y = new THREE.Vector3().fromArray(state.hinges[key + '_pitch_joint'].axis_world).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, x.clone().cross(y).normalize()));
}
export const canRotateSelection = (members: string[]) => members.length > 0 && (members.every(k => ROTATABLE.includes(k)) || members.length === 1 && (HINGE_HANDLES.includes(members[0]) || isJointHandle(members[0])));
type Callbacks = {
  select: (key: string, additive: boolean, hover: boolean) => void;
  begin: () => void;
  move: (key: string, target: number[]) => void;
  rotate: (key: string, quaternion: number[]) => void;
  jointAngle: (key: string, angle: number) => void;
  transformMode: (mode: TransformMode) => void;
  end: () => void;
  error: (message: string) => void;
  history?: (redo: boolean) => void;
  selectObject?: (id: string | null) => void;
  deleteObject?: (id: string) => void;
  transformObject?: (id: string, patch: Partial<SceneObject>) => void;
  objectTransformBegin?: () => void;
  objectTransformEnd?: () => void;
  objectTransformMode?: (mode: ObjectTransformMode) => void;
  pickObjectSurface?: (id: string, localPoint: number[], localNormal: number[]) => void;
};

export class RobotScene {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, 1, .01, 100);
  renderer: THREE.WebGLRenderer;
  orbit: OrbitControls;
  gizmo: TransformControls;
  pivot = new THREE.Object3D();
  markers: Record<string, THREE.Mesh> = {};
  labels: Record<string, HTMLDivElement> = {};
  geoms: Record<string, THREE.Object3D> = {};
  referenceGeoms: Record<string, THREE.Object3D> = {};
  referenceRoot: THREE.Object3D | null = null;
  referenceState?: PoseState;
  referenceVisible = false;
  selected = 'pelvis';
  members = ['pelvis'];
  selectionLocked = false;
  pins: string[] = [];
  anglePins: string[] = [];
  state?: PoseState;
  editable = true;
  keyboardEnabled = true;
  destroyed = false;
  frame = 0;
  dirty = true;
  observer: ResizeObserver;
  ray = new THREE.Raycaster();
  pointer = new THREE.Vector2();
  box: THREE.Mesh;
  sceneObjects: Record<string, THREE.Object3D> = {};
  collisionProxies: Record<string, THREE.Object3D> = {};
  collisionProxiesVisible = false;
  sceneAssetLoads = new Map<string, Promise<THREE.Object3D>>();
  selectedSceneObject: string | null = null;
  objectTransformMode: ObjectTransformMode = 'translate';
  com: THREE.Mesh;
  markerVisible = true;
  handleLayer: 'body' | 'joints' = 'body';
  visibleHandles: string[] | null = null;
  mirrorTranslation = false;
  surfacePickMode = false;
  gripMarkerObjectId: string | null = null;
  gripMarkerUV: { left: number[]; right: number[] } | null = null;
  gripMarkers: Record<'left' | 'right', THREE.Mesh>;
  gripPads: Record<'left' | 'right', THREE.Mesh>;
  transformMode: TransformMode = 'translate';
  space: 'world' | 'local' = 'world';
  hingeDrag: { quaternion: THREE.Quaternion; angle: number; lastTwist: number; delta: number; limits: number[]; key: string; component: 'x' | 'y' | 'z' } | null = null;
  modelMismatchReported = false;

  constructor(public host: HTMLDivElement, private callbacks: Callbacks, public modelId = 'g1', visualRevision = '') {
    this.scene.background = new THREE.Color('#10171f');
    this.scene.fog = new THREE.Fog('#10171f', 5, 14);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.4, -2.8, 1.85);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.target.set(0, 0, .7);
    this.orbit.enableDamping = false;
    this.orbit.minDistance = .5;
    this.orbit.maxDistance = 10;
    this.orbit.maxPolarAngle = Math.PI * .49;
    this.orbit.update();
    this.orbit.addEventListener('change', () => { this.dirty = true; });
    this.scene.add(new THREE.HemisphereLight(0xcce9ff, 0x37434e, 2.6));
    const key = new THREE.DirectionalLight(0xffffff, 3.8);
    key.position.set(2, -3, 5);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x91cfff, 2);
    rim.position.set(-2, 2, 3);
    this.scene.add(rim);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: '#16212d', roughness: 1 }));
    floor.position.z = -.006;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(10, 100, 0x375066, 0x253647);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -.003;
    this.scene.add(grid);
    const origin = new THREE.AxesHelper(.25);
    origin.position.set(.55, -.5, .003);
    this.scene.add(origin);
    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSpace('world');
    this.gizmo.setSize(.85);
    this.gizmo.showX = this.gizmo.showY = this.gizmo.showZ = true;
    this.scene.add(this.pivot, this.gizmo.getHelper());
    this.gizmo.addEventListener('dragging-changed', event => {
      this.orbit.enabled = !event.value;
      if (this.selectedSceneObject) {
        if (event.value) this.callbacks.objectTransformBegin?.();
        else this.callbacks.objectTransformEnd?.();
        this.dirty = true;
        return;
      }
      if (event.value) {
        const ring = jointForRing(controlKey(this.members, this.selected), this.gizmo.axis);
        const hinge = this.transformMode === 'rotate' && controlSelection(this.members).length === 1 && ring ? this.state?.hinges?.[ring.key] : undefined;
        this.hingeDrag = hinge && ring ? { quaternion: this.pivot.quaternion.clone(), angle: hinge.angle, lastTwist: 0, delta: 0, limits: hinge.limits, ...ring } : null;
        this.callbacks.begin();
      } else {
        this.hingeDrag = null;
        this.callbacks.end();
      }
      this.dirty = true;
    });
    this.gizmo.addEventListener('objectChange', () => {
      this.dirty = true;
      if (this.gizmo.dragging && this.editable) {
        const object = this.selectedSceneObject ? this.sceneObjects[this.selectedSceneObject] : undefined;
        if (object && this.selectedSceneObject) {
          const shape = object.userData.shape as SceneObject['shape'];
          const size = normalizedObjectSize(shape, object.scale.toArray(), this.gizmo.axis ?? '');
          this.callbacks.transformObject?.(this.selectedSceneObject, {
            position: object.position.toArray(), quaternion_xyzw: object.quaternion.toArray(), size,
          });
        } else if (this.hingeDrag) {
          this.applyHingeDrag();
        } else if (this.transformMode === 'rotate') {
          if (!ANKLE_HANDLES.includes(controlKey(this.members, this.selected))) this.callbacks.rotate(this.selected, this.pivot.quaternion.toArray());
        }
        else this.callbacks.move(this.selected, this.pivot.position.toArray());
      }
    });
    this.gizmo.addEventListener('change', () => { this.dirty = true; });
    this.box = new THREE.Mesh(new THREE.BoxGeometry(.3, .32, .24), new THREE.MeshStandardMaterial({ color: '#b98853', roughness: .9, transparent: true, opacity: .62 }));
    this.box.position.set(.40, 0, .30);
    this.scene.add(this.box);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(this.box.geometry), new THREE.LineBasicMaterial({ color: '#e7bf8b' }));
    this.box.add(edges);
    this.com = new THREE.Mesh(new THREE.RingGeometry(.022, .03, 24), new THREE.MeshBasicMaterial({ color: '#f1c267', side: THREE.DoubleSide }));
    this.scene.add(this.com);
    this.gripMarkers = Object.fromEntries((['left', 'right'] as const).map(side => {
      const marker = new THREE.Mesh(new THREE.SphereGeometry(.018, 16, 10), new THREE.MeshBasicMaterial({ color: side === 'left' ? '#55e7c1' : '#ffbd70', depthTest: false }));
      marker.visible = false; marker.renderOrder = 5; this.scene.add(marker); return [side, marker];
    })) as unknown as Record<'left' | 'right', THREE.Mesh>;
    this.gripPads = Object.fromEntries((['left', 'right'] as const).map(side => {
      const color = side === 'left' ? '#42e5bc' : '#ff9e45';
      const pad = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({
        color, emissive: color, emissiveIntensity: .18, transparent: true, opacity: .5,
        roughness: .55, metalness: .05, side: THREE.DoubleSide, depthWrite: false,
      }));
      pad.renderOrder = 1;
      const outline = new THREE.LineSegments(new THREE.EdgesGeometry(pad.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: .9 }));
      pad.add(outline); this.scene.add(pad); return [side, pad];
    })) as unknown as Record<'left' | 'right', THREE.Mesh>;
    this.renderer.domElement.addEventListener('pointermove', this.hover);
    this.renderer.domElement.addEventListener('pointerdown', this.click, true);
    window.addEventListener('keydown', this.keydown);
    this.observer = new ResizeObserver(() => {
      const { width, height } = host.getBoundingClientRect();
      this.renderer.setSize(width, height);
      this.camera.aspect = width / Math.max(height, 1);
      this.camera.updateProjectionMatrix();
      this.dirty = true;
    });
    this.observer.observe(host);
    const render = () => {
      if (this.destroyed) return;
      if (this.dirty) {
        this.renderer.render(this.scene, this.camera);
        this.positionLabels();
        this.dirty = false;
      }
      this.frame = requestAnimationFrame(render);
    };
    render();
    const revision = visualRevision ? `?revision=${encodeURIComponent(visualRevision)}` : '';
    new GLTFLoader().load(`/api/robot/${encodeURIComponent(modelId)}.glb${revision}`, gltf => {
      if (this.destroyed) return;
      gltf.scene.traverse(node => {
        if (node.name.startsWith('geom_')) this.geoms[node.name.slice(5)] = node;
        if (node instanceof THREE.Mesh) {
          const mats = Array.isArray(node.material) ? node.material : [node.material];
          mats.forEach(mat => { if (mat instanceof THREE.MeshStandardMaterial) { mat.roughness = .65; mat.metalness = .25; } });
        }
      });
      this.scene.add(gltf.scene);
      const reference = gltf.scene.clone(true);
      reference.traverse(node => {
        if (node.name.startsWith('geom_')) this.referenceGeoms[node.name.slice(5)] = node;
        if (node instanceof THREE.Mesh) {
          const ghostMaterial = (material: THREE.Material) => {
            const clone = material.clone();
            clone.transparent = true;
            clone.opacity = .23;
            clone.depthWrite = false;
            clone.polygonOffset = true;
            clone.polygonOffsetFactor = -1;
            clone.polygonOffsetUnits = -1;
            if (clone instanceof THREE.MeshStandardMaterial) {
              clone.color.set('#77cfff');
              clone.emissive.set('#255b76');
              clone.emissiveIntensity = .35;
              clone.roughness = .75;
              clone.metalness = 0;
            }
            return clone;
          };
          node.material = Array.isArray(node.material)
            ? node.material.map(ghostMaterial)
            : ghostMaterial(node.material);
          node.renderOrder = 3;
        }
      });
      reference.visible = this.referenceVisible;
      this.referenceRoot = reference;
      this.scene.add(reference);
      if (this.referenceState) this.updateReference(this.referenceState);
      if (this.state) this.update(this.state);
    }, undefined, () => callbacks.error('G1 모델을 불러오지 못했습니다. 서버 연결을 확인하세요.'));
  }

  applyHingeDrag() {
    const drag = this.hingeDrag;
    if (!drag) return;
    const relative = drag.quaternion.clone().invert().multiply(this.pivot.quaternion);
    const twist = 2 * Math.atan2(relative[drag.component], relative.w);
    drag.delta += Math.atan2(Math.sin(twist - drag.lastTwist), Math.cos(twist - drag.lastTwist));
    drag.lastTwist = twist;
    this.callbacks.jointAngle(drag.key, THREE.MathUtils.clamp(drag.angle + drag.delta, drag.limits[0], drag.limits[1]));
  }

  private pick(event: PointerEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.ray.setFromCamera(this.pointer, this.camera);
    const keys = [...new Set(this.ray.intersectObjects(Object.values(this.markers).filter(m => m.visible), false).map(hit => hit.object.name))];
    return event.altKey ? keys[(keys.indexOf(this.selected) + 1) % keys.length] : keys[0];
  }
  private pickSceneObject(event: PointerEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.ray.setFromCamera(this.pointer, this.camera);
    const hit = this.ray.intersectObjects(Object.values(this.sceneObjects).filter(object => object.visible), true)[0]?.object;
    let node: THREE.Object3D | null = hit ?? null;
    while (node && !node.userData.sceneObjectId) node = node.parent;
    return node?.userData.sceneObjectId as string | undefined;
  }
  private pickSceneObjectSurface(event: PointerEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.ray.setFromCamera(this.pointer, this.camera);
    return this.ray.intersectObjects(Object.values(this.sceneObjects)
      .filter(object => object.visible && !object.userData.graspGhost && object.userData.shape === 'box' && !object.userData.assetId), true)[0];
  }
  hover = (event: PointerEvent) => {
    if (!this.editable || event.buttons || event.shiftKey || event.altKey || this.selectionLocked || this.members.length > 1 || this.gizmo.dragging || this.gizmo.axis) return;
    const key = this.pick(event);
    if (key && key !== this.selected) this.callbacks.select(key, false, true);
  };
  click = (event: PointerEvent) => {
    if (!this.editable || this.gizmo.dragging || event.button !== 0) return;
    if (this.surfacePickMode) {
      const hit = this.pickSceneObjectSurface(event);
      if (hit?.face) {
        event.preventDefault(); event.stopImmediatePropagation();
        const mesh = hit.object as THREE.Mesh;
        let root: THREE.Object3D | null = mesh;
        while (root && !root.userData.sceneObjectId) root = root.parent;
        const localPoint = (root ?? mesh).worldToLocal(hit.point.clone()).toArray();
        this.callbacks.pickObjectSurface?.(root?.userData.sceneObjectId as string, localPoint, hit.face.normal.toArray());
      }
      return;
    }
    const key = this.pick(event);
    if (event.altKey && key) {
      event.preventDefault();
      event.stopImmediatePropagation();
      this.callbacks.select(key, event.shiftKey, false);
      return;
    }
    if (event.shiftKey && key) {
      event.preventDefault();
      event.stopImmediatePropagation();
      this.callbacks.select(key, true, false);
      return;
    }
    if (this.gizmo.axis) return;
    if (key) this.callbacks.select(key, event.shiftKey, false);
    else {
      const object = this.pickSceneObject(event);
      if (object) this.callbacks.selectObject?.(object);
      else if (!event.shiftKey) { this.selectionLocked = false; this.callbacks.selectObject?.(null); }
    }
  };

  keydown = (event: KeyboardEvent) => this.handleKeyDown(event);

  handleKeyDown(event: KeyboardEvent) {
    const target = event.target as HTMLElement | null;
    if (this.keyboardEnabled === false || event.repeat || event.isComposing || event.altKey || this.gizmo.dragging
        || target?.isContentEditable || target?.closest('input, textarea, select')) return;
    const key = event.code || event.key.toLowerCase();
    if (event.ctrlKey || event.metaKey) {
      if ((key === 'KeyZ' || key === 'z') && this.editable && this.state && this.callbacks.history) {
        event.preventDefault();
        this.callbacks.history(event.shiftKey);
      }
      return;
    }
    if (key === 'KeyF' || key === 'f') {
      event.preventDefault();
      this.focusSelection();
    } else if (this.editable && this.selectedSceneObject && !this.sceneObjects?.[this.selectedSceneObject]?.userData.graspGhost
        && ['Delete', 'Backspace'].includes(key)) {
      event.preventDefault();
      this.callbacks.deleteObject?.(this.selectedSceneObject);
    } else if (this.editable && this.selectedSceneObject && ['KeyW', 'KeyE', 'KeyR', 'w', 'e', 'r'].includes(key)) {
      event.preventDefault();
      this.callbacks.objectTransformMode?.(key === 'KeyW' || key === 'w' ? 'translate' : key === 'KeyE' || key === 'e' ? 'rotate' : 'scale');
    } else if (this.editable && this.state && ['KeyW', 'KeyE', 'w', 'e'].includes(key)) {
      event.preventDefault();
      this.callbacks.transformMode(key === 'KeyW' || key === 'w' ? 'translate' : 'rotate');
    }
  }

  center() {
    const center = new THREE.Vector3();
    const controls = controlSelection(this.members);
    if (this.state) controls.forEach(k => center.add(new THREE.Vector3().fromArray(this.state!.handles[k].position)));
    return center.divideScalar(controls.length);
  }

  focusSelection() {
    if (!this.state) return;
    const object = this.selectedSceneObject ? this.sceneObjects[this.selectedSceneObject] : undefined;
    const center = object ? object.position.clone() : this.center();
    const radius = object ? Math.max(.12, object.scale.length() / 2) : Math.max(.12, ...this.members.map(k => new THREE.Vector3().fromArray(this.state!.handles[k].position).distanceTo(center) + .10));
    const halfFov = Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * Math.min(1, this.camera.aspect));
    const distance = Math.max(.55, radius / Math.sin(halfFov));
    const direction = this.camera.position.clone().sub(this.orbit.target).normalize();
    this.orbit.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.orbit.update();
    this.dirty = true;
  }

  select(key: string, pins: string[], members = [key], anglePins = this.anglePins ?? []) {
    this.selected = key;
    this.members = members;
    this.pins = pins;
    this.anglePins = anglePins;
    const controls = controlSelection(members);
    const activeControl = controlKey(members, key);
    Object.entries(this.markers).forEach(([k, mesh]) => {
      mesh.visible = this.markerVisible && ((this.visibleHandles ? this.visibleHandles.includes(k) : isJointHandle(k) === (this.handleLayer === 'joints') && !isCombinedAxis(k)) || members.includes(k) || controls.includes(k));
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.color.set(members.includes(k) || controls.includes(k) ? '#80f2c7' : pins.includes(k) ? '#f1bc65' : anglePins.includes(k) ? '#ba9cff' : '#56bdec');
      mat.opacity = members.includes(k) || controls.includes(k) ? .64 : .28;
      this.labels[k]?.classList.toggle('chosen', members.includes(k));
      this.labels[k]?.classList.toggle('pinned', pins.includes(k));
      this.labels[k]?.classList.toggle('angle-pinned', anglePins.includes(k));
    });
    const hinge = this.transformMode === 'rotate' && controls.length === 1 ? this.state?.hinges?.[activeControl] : undefined;
    const ankle = this.transformMode === 'rotate' && controls.length === 1 && ANKLE_HANDLES.includes(activeControl);
    this.gizmo.setSpace(hinge || ankle ? 'local' : this.space);
    this.gizmo.showX = this.gizmo.showY = !hinge;
    this.gizmo.showZ = !ankle;
    if (this.state && !this.gizmo.dragging) {
      const mirror = this.mirrorTranslation && this.transformMode === 'translate' && canMirrorSelection(controls);
      this.pivot.position.copy(hinge ? new THREE.Vector3().fromArray(hinge.position) : mirror ? new THREE.Vector3().fromArray(this.state.handles[activeControl].position) : this.center());
      if (ankle) this.pivot.quaternion.copy(ankleFrame(this.state, activeControl));
      else if (hinge) this.pivot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3().fromArray(hinge.axis_world).normalize());
      else this.pivot.quaternion.fromArray(this.state.handles[activeControl].quaternion);
    }
    const blocked = this.transformMode === 'rotate'
      ? !canRotateSelection(controls) || controls.some(k => anglePins.includes(k) || pins.includes(k) && (controls.length > 1 || k.endsWith('_foot')))
      : members.some(k => pins.includes(k));
    if (this.editable && !blocked && this.markerVisible && !this.selectedSceneObject) this.gizmo.attach(this.pivot); else if (!this.selectedSceneObject) this.gizmo.detach();
    this.dirty = true;
  }

  update(state: PoseState) {
    if (state.model_id && state.model_id !== this.modelId) {
      if (!this.modelMismatchReported) this.callbacks.error(`로봇 모델이 ${this.modelId}에서 ${state.model_id}(으)로 변경되었습니다. 화면을 새로고침하세요.`);
      this.modelMismatchReported = true;
      return;
    }
    this.modelMismatchReported = false;
    this.state = state;
    for (const [id, pose] of Object.entries(state.geoms)) {
      const obj = this.geoms[id];
      if (obj) { obj.position.fromArray(pose.position); obj.quaternion.fromArray(pose.quaternion); }
    }
    for (const side of ['left', 'right'] as const) this.gripPads[side].visible = !!state.grip_pads?.[side];
    for (const [side, pose] of Object.entries(state.grip_pads ?? {})) {
      const pad = this.gripPads[side as 'left' | 'right'];
      pad.position.fromArray(pose.position); pad.quaternion.fromArray(pose.quaternion); pad.scale.fromArray(pose.size);
    }
    for (const [key, h] of Object.entries(state.handles)) {
      if (!this.markers[key]) {
        const geometry = new THREE.SphereGeometry(key === 'pelvis' ? .06 : isJointHandle(key) ? .028 : .042, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2);
        geometry.rotateX(Math.PI / 2);
        const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: '#56bdec', transparent: true, opacity: .3, side: THREE.DoubleSide, depthWrite: false, depthTest: false }));
        mesh.name = key;
        mesh.renderOrder = 2;
        this.markers[key] = mesh;
        this.scene.add(mesh);
        const label = document.createElement('div');
        label.className = 'joint-label';
        label.textContent = h.label;
        this.host.appendChild(label);
        this.labels[key] = label;
      }
      this.markers[key].position.fromArray(h.position);
    }
    this.com.position.set(state.com[0], state.com[1], .002);
    this.select(this.selected, this.pins, this.members);
    this.dirty = true;
  }

  updateReference(state: PoseState) {
    if (state.model_id && state.model_id !== this.modelId) return;
    this.referenceState = state;
    for (const [id, pose] of Object.entries(state.geoms)) {
      const object = this.referenceGeoms[id];
      if (object) { object.position.fromArray(pose.position); object.quaternion.fromArray(pose.quaternion); }
    }
    this.dirty = true;
  }

  setReferenceVisible(visible: boolean) {
    this.referenceVisible = visible;
    if (this.referenceRoot) this.referenceRoot.visible = visible;
    this.dirty = true;
  }

  positionLabels() {
    const rect = this.host.getBoundingClientRect();
    Object.entries(this.markers).forEach(([k, mesh]) => {
      const p = mesh.position.clone().project(this.camera);
      const label = this.labels[k];
      label.style.display = mesh.visible && p.z < 1 && (this.members.includes(k) || this.pins.includes(k) || this.anglePins.includes(k)) ? 'block' : 'none';
      label.style.transform = `translate(${(p.x + 1) * rect.width / 2 + 15}px,${(1 - p.y) * rect.height / 2 - 12}px)`;
    });
  }

  setView(view: 'perspective' | 'front' | 'side') {
    const positions = { perspective: [2.4, -2.8, 1.85], front: [3.2, 0, .85], side: [0, -3.2, .85] };
    this.camera.position.fromArray(positions[view]);
    this.orbit.target.set(0, 0, .7);
    this.orbit.update();
    this.dirty = true;
  }
  showHandles(show: boolean) {
    this.markerVisible = show;
    Object.values(this.markers).forEach(m => { m.visible = show; });
    this.select(this.selected, this.pins, this.members);
  }
  setHandleLayer(layer: 'body' | 'joints') {
    this.handleLayer = layer;
    this.select(this.selected, this.pins, this.members);
  }
  setVisibleHandles(handles: string[]) {
    this.visibleHandles = handles;
    this.select(this.selected, this.pins, this.members);
  }
  setMirrorTranslation(enabled: boolean) {
    this.mirrorTranslation = enabled;
    this.select(this.selected, this.pins, this.members);
  }
  setEditable(editable: boolean) {
    this.editable = editable;
    if (this.selectedSceneObject) this.selectSceneObject(this.selectedSceneObject, this.objectTransformMode);
    else this.select(this.selected, this.pins, this.members);
  }
  setTransformMode(mode: TransformMode, space: 'world' | 'local') {
    this.transformMode = mode;
    this.space = space;
    if (!this.selectedSceneObject) {
      this.gizmo.setMode(mode);
      this.gizmo.setSpace(space);
      this.select(this.selected, this.pins, this.members);
    }
  }
  setBox(position: number[], size: number[], visible: boolean) {
    this.box.visible = visible;
    this.box.position.fromArray(position);
    this.box.scale.set(size[0] / .3, size[1] / .32, size[2] / .24);
    this.dirty = true;
  }
  private disposeSceneObject(object: THREE.Object3D) {
    object.traverse(node => {
      if (!(node instanceof THREE.Mesh || node instanceof THREE.LineSegments)) return;
      node.geometry?.dispose();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach(material => material?.dispose());
    });
  }
  private styleSceneObject(root: THREE.Object3D, object: SceneObject) {
    root.traverse(node => {
      node.userData.sceneObjectId = object.id;
      if (!(node instanceof THREE.Mesh)) return;
      const materials = (Array.isArray(node.material) ? node.material : [node.material]) as THREE.MeshStandardMaterial[];
      materials.forEach(material => {
        material.transparent = object.opacity < 1 || !!object.ghost;
        material.opacity = object.opacity;
        material.depthWrite = object.opacity >= .98 && !object.ghost;
        material.wireframe = !!object.ghost;
        if (!object.asset_id) material.color?.set(object.color);
        if ('emissive' in material) material.emissive.set(this.selectedSceneObject === object.id ? '#254e43' : object.ghost ? '#174c55' : '#000000');
      });
    });
  }
  private styleCollisionProxy(root: THREE.Object3D, object: SceneObject) {
    root.traverse(node => {
      if (!(node instanceof THREE.Mesh)) return;
      const materials = (Array.isArray(node.material) ? node.material : [node.material]) as THREE.MeshStandardMaterial[];
      materials.forEach(material => {
        material.color.set(object.shape === 'open_box' ? '#ffb45f' : '#55d9ff');
        material.emissive.set(object.shape === 'open_box' ? '#4f2508' : '#07394d');
        material.emissiveIntensity = .45;
        material.transparent = true;
        material.opacity = .72;
        material.depthWrite = false;
        material.wireframe = true;
      });
    });
  }
  private collisionProxy(object: SceneObject) {
    if (object.collision_shape === 'convex_hull' && object.collision_hull_vertices && object.collision_hull_faces) {
      const positions = object.collision_hull_vertices.flat();
      const indices = object.collision_hull_faces.flat();
      const buffer = new THREE.BufferGeometry();
      buffer.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      buffer.setIndex(indices); buffer.computeVertexNormals();
      const geometry = new THREE.Group();
      geometry.add(new THREE.Mesh(buffer, new THREE.MeshStandardMaterial()));
      geometry.scale.fromArray(object.size);
      const root = new THREE.Group(); root.add(geometry);
      return root;
    }
    const descriptor = {
      ...object,
      shape: object.collision_shape ?? object.shape,
      size: object.collision_size ?? object.size,
    } as SceneObject;
    const root = new THREE.Group();
    const geometry = this.primitiveSceneObject(descriptor);
    geometry.quaternion.fromArray(object.collision_quaternion_xyzw ?? [0, 0, 0, 1]);
    geometry.scale.fromArray(normalizedObjectSize(descriptor.shape, descriptor.size));
    root.add(geometry);
    return root;
  }
  private primitiveSceneObject(object: SceneObject) {
    const root = new THREE.Group();
    const material = () => new THREE.MeshStandardMaterial({ roughness: .82, metalness: .04 });
    if (object.shape === 'open_box') {
      const thickness = Math.max(.001, Math.min(object.wall_thickness_m ?? .02, Math.min(...object.size) / 3));
      const [x, y, z] = object.size;
      const parts: [number[], number[]][] = [
        [[1, 1, thickness / z], [0, 0, -.5 + thickness / z / 2]],
        [[thickness / x, Math.max(.01, 1 - 2 * thickness / y), Math.max(.01, 1 - thickness / z)], [-.5 + thickness / x / 2, 0, thickness / z / 2]],
        [[thickness / x, Math.max(.01, 1 - 2 * thickness / y), Math.max(.01, 1 - thickness / z)], [.5 - thickness / x / 2, 0, thickness / z / 2]],
        [[1, thickness / y, Math.max(.01, 1 - thickness / z)], [0, -.5 + thickness / y / 2, thickness / z / 2]],
        [[1, thickness / y, Math.max(.01, 1 - thickness / z)], [0, .5 - thickness / y / 2, thickness / z / 2]],
      ];
      parts.forEach(([scale, position]) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material());
        mesh.scale.fromArray(scale); mesh.position.fromArray(position); root.add(mesh);
      });
    } else {
      const geometry = object.shape === 'box' ? new THREE.BoxGeometry(1, 1, 1)
        : object.shape === 'sphere' ? new THREE.SphereGeometry(.5, 32, 20)
        : new THREE.CylinderGeometry(.5, .5, 1, 32);
      if (object.shape === 'cylinder') geometry.rotateX(Math.PI / 2);
      root.add(new THREE.Mesh(geometry, material()));
    }
    return root;
  }
  private loadSceneAsset(assetId: string) {
    let pending = this.sceneAssetLoads.get(assetId);
    if (!pending) {
      const url = `/api/scene-assets/${encodeURIComponent(assetId)}.glb`;
      pending = new Promise((resolve, reject) => new GLTFLoader().load(
        url,
        gltf => resolve(gltf.scene), undefined, reject,
      ));
      this.sceneAssetLoads.set(assetId, pending);
    }
    return pending;
  }
  private installSceneAsset(object: SceneObject, root: THREE.Object3D) {
    if (!object.asset_id || !object.asset_bounds_min || !object.asset_bounds_max) return;
    const expectedId = object.asset_id;
    void this.loadSceneAsset(expectedId).then(template => {
      if (this.destroyed || this.sceneObjects[object.id] !== root || root.userData.assetId !== expectedId) return;
      let clone: THREE.Object3D;
      if (object.asset_node_name) {
        template.updateMatrixWorld(true);
        const source = template.getObjectByName(object.asset_node_name);
        if (!source) throw new Error(`GLB node not found: ${object.asset_node_name}`);
        const selected = source.clone(true);
        source.matrixWorld.decompose(selected.position, selected.quaternion, selected.scale);
        clone = new THREE.Group(); clone.add(selected);
      } else clone = template.clone(true);
      clone.traverse(node => {
        if (!(node instanceof THREE.Mesh)) return;
        node.geometry = node.geometry.clone();
        node.material = Array.isArray(node.material) ? node.material.map(material => material.clone()) : node.material.clone();
      });
      const normalized = normalizedSceneAsset(clone, object.asset_bounds_min!, object.asset_bounds_max!, object.asset_axis_transform_xyzw);
      this.disposeSceneObject(root);
      root.clear(); root.add(normalized);
      this.styleSceneObject(root, object);
      this.dirty = true;
    }).catch(() => this.callbacks.error(`3D 모델(${object.name})을 불러오지 못했습니다.`));
  }
  setSceneObjects(objects: SceneObject[]) {
    this.box.visible = false;
    const incoming = new Set(objects.map(object => object.id));
    for (const [id, mesh] of Object.entries(this.sceneObjects)) {
      if (incoming.has(id)) continue;
      this.scene.remove(mesh);
      this.disposeSceneObject(mesh);
      delete this.sceneObjects[id];
      const proxy = this.collisionProxies[id];
      if (proxy) {
        this.scene.remove(proxy); this.disposeSceneObject(proxy);
        delete this.collisionProxies[id];
      }
    }
    for (const object of objects) {
      let mesh = this.sceneObjects[object.id];
      if (!mesh || mesh.userData.shape !== object.shape || mesh.userData.assetId !== object.asset_id
          || mesh.userData.assetNodeName !== object.asset_node_name) {
        if (mesh) {
          this.scene.remove(mesh); this.disposeSceneObject(mesh);
        }
        mesh = this.primitiveSceneObject(object);
        mesh.userData.sceneObjectId = object.id;
        mesh.userData.shape = object.shape;
        mesh.userData.assetId = object.asset_id;
        mesh.userData.assetNodeName = object.asset_node_name;
        this.sceneObjects[object.id] = mesh;
        this.scene.add(mesh);
        this.installSceneAsset(object, mesh);
      }
      mesh.userData.graspGhost = !!object.ghost;
      mesh.userData.objectVisible = object.visible;
      mesh.position.fromArray(object.position);
      mesh.quaternion.fromArray(object.quaternion_xyzw);
      mesh.scale.fromArray(normalizedObjectSize(object.shape, object.size));
      mesh.visible = object.visible && !this.collisionProxiesVisible;
      this.styleSceneObject(mesh, object);
      let proxy = this.collisionProxies[object.id];
      if (object.ghost) {
        if (proxy) {
          this.scene.remove(proxy); this.disposeSceneObject(proxy);
          delete this.collisionProxies[object.id];
        }
        continue;
      }
      const proxyShape = object.collision_shape ?? object.shape;
      const proxySize = object.collision_size ?? object.size;
      const proxySignature = JSON.stringify([proxyShape, proxySize, object.collision_quaternion_xyzw,
        object.collision_hull_vertices, object.collision_hull_faces]);
      if (!proxy || proxy.userData.signature !== proxySignature) {
        if (proxy) { this.scene.remove(proxy); this.disposeSceneObject(proxy); }
        proxy = this.collisionProxy(object);
        proxy.userData.signature = proxySignature;
        this.collisionProxies[object.id] = proxy;
        this.scene.add(proxy);
        this.styleCollisionProxy(proxy, object);
      }
      proxy.userData.objectVisible = object.visible;
      proxy.position.copy(mesh.position); proxy.quaternion.copy(mesh.quaternion); proxy.scale.set(1, 1, 1);
      proxy.visible = object.visible && this.collisionProxiesVisible;
    }
    if (this.selectedSceneObject && !incoming.has(this.selectedSceneObject)) this.selectedSceneObject = null;
    if (this.selectedSceneObject) this.selectSceneObject(this.selectedSceneObject, this.objectTransformMode);
    this.refreshGripMarkers();
    this.dirty = true;
  }
  setCollisionProxiesVisible(visible: boolean) {
    this.collisionProxiesVisible = visible;
    Object.values(this.sceneObjects).forEach(object => {
      object.visible = !!object.userData.objectVisible && !visible;
    });
    Object.values(this.collisionProxies).forEach(proxy => {
      proxy.visible = !!proxy.userData.objectVisible && visible;
    });
    this.dirty = true;
  }
  setSurfacePickMode(enabled: boolean) {
    this.surfacePickMode = enabled;
    this.renderer.domElement.style.cursor = enabled ? 'crosshair' : '';
  }
  setGripMarkers(id: string | null, leftUV?: number[], rightUV?: number[]) {
    this.gripMarkerObjectId = id;
    this.gripMarkerUV = id && leftUV && rightUV ? { left: [...leftUV], right: [...rightUV] } : null;
    this.refreshGripMarkers();
  }
  private refreshGripMarkers() {
    const object = this.gripMarkerObjectId ? this.sceneObjects[this.gripMarkerObjectId] : undefined;
    if (!object || object.userData.shape !== 'box' || !this.gripMarkerUV) {
      Object.values(this.gripMarkers).forEach(marker => { marker.visible = false; });
      return;
    }
    object.updateMatrixWorld(true);
    for (const [side, sign] of [['left', 1], ['right', -1]] as const) {
      const uv = this.gripMarkerUV[side];
      this.gripMarkers[side].position.copy(new THREE.Vector3(uv[0] / 2, sign / 2, uv[1] / 2).applyMatrix4(object.matrixWorld));
      this.gripMarkers[side].visible = true;
    }
    this.dirty = true;
  }
  selectSceneObject(id: string | null, mode: ObjectTransformMode = this.objectTransformMode) {
    this.selectedSceneObject = id && this.sceneObjects[id] ? id : null;
    this.selectionLocked = !!this.selectedSceneObject;
    this.objectTransformMode = mode;
    Object.entries(this.sceneObjects).forEach(([key, mesh]) => {
      mesh.traverse(node => {
        if (!(node instanceof THREE.Mesh)) return;
        const materials = (Array.isArray(node.material) ? node.material : [node.material]) as THREE.MeshStandardMaterial[];
        materials.forEach(material => {
          if ('emissive' in material) material.emissive.set(key === this.selectedSceneObject
            ? '#254e43' : mesh.userData.graspGhost ? '#174c55' : '#000000');
        });
      });
    });
    const object = this.selectedSceneObject ? this.sceneObjects[this.selectedSceneObject] : undefined;
    if (object && this.editable) {
      this.gizmo.showX = this.gizmo.showY = this.gizmo.showZ = true;
      this.gizmo.setMode(mode);
      this.gizmo.setSpace(mode === 'translate' ? this.space : 'local');
      this.gizmo.attach(object);
    } else {
      this.gizmo.detach();
      if (!id) this.select(this.selected, this.pins, this.members);
    }
    this.dirty = true;
  }
  setObjectPoses(poses: Record<string, SceneObjectPose>) {
    Object.entries(poses).forEach(([id, pose]) => {
      const object = this.sceneObjects[id];
      if (object) { object.position.fromArray(pose.position); object.quaternion.fromArray(pose.quaternion_xyzw); }
      const proxy = this.collisionProxies[id];
      if (proxy) { proxy.position.fromArray(pose.position); proxy.quaternion.fromArray(pose.quaternion_xyzw); }
    });
    this.dirty = true;
  }
  dispose() {
    this.destroyed = true;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.renderer.domElement.removeEventListener('pointermove', this.hover);
    this.renderer.domElement.removeEventListener('pointerdown', this.click, true);
    window.removeEventListener('keydown', this.keydown);
    this.orbit.dispose();
    this.gizmo.dispose();
    this.scene.traverse(node => {
      if (node instanceof THREE.Mesh || node instanceof THREE.Line) {
        node.geometry.dispose();
        (Array.isArray(node.material) ? node.material : [node.material]).forEach(m => m.dispose());
      }
    });
    this.renderer.dispose();
    this.host.replaceChildren();
  }
}
