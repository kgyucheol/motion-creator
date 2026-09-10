import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export type PoseState = {
  qpos: number[];
  handles: Record<string, { position: number[]; quaternion: number[]; label: string }>;
  geoms: Record<string, { position: number[]; quaternion: number[] }>;
  com: number[];
  floor_min_mm: number;
  hinges: Record<string, { joint_name: string; angle: number; limits: number[]; axis_world: number[]; position: number[] }>;
};
export type TransformMode = 'translate' | 'rotate';
export const ROTATABLE = ['pelvis', 'left_hand', 'right_hand', 'left_foot', 'right_foot'];
export const HINGE_HANDLES = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee'];
export const isJointHandle = (key: string) => key.endsWith('_joint');
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
  selected = 'pelvis';
  members = ['pelvis'];
  selectionLocked = false;
  pins: string[] = [];
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
  com: THREE.Mesh;
  markerVisible = true;
  handleLayer: 'body' | 'joints' = 'body';
  transformMode: TransformMode = 'translate';
  space: 'world' | 'local' = 'world';
  hingeDrag: { quaternion: THREE.Quaternion; angle: number; lastTwist: number; delta: number; limits: number[] } | null = null;

  constructor(public host: HTMLDivElement, private callbacks: Callbacks) {
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
      if (event.value) {
        const hinge = this.transformMode === 'rotate' && this.members.length === 1 ? this.state?.hinges?.[this.selected] : undefined;
        this.hingeDrag = hinge ? { quaternion: this.pivot.quaternion.clone(), angle: hinge.angle, lastTwist: 0, delta: 0, limits: hinge.limits } : null;
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
        if (this.hingeDrag) {
          const drag = this.hingeDrag;
          const relative = drag.quaternion.clone().invert().multiply(this.pivot.quaternion);
          const twist = 2 * Math.atan2(relative.z, relative.w);
          drag.delta += Math.atan2(Math.sin(twist - drag.lastTwist), Math.cos(twist - drag.lastTwist));
          drag.lastTwist = twist;
          this.callbacks.jointAngle(this.selected, THREE.MathUtils.clamp(drag.angle + drag.delta, drag.limits[0], drag.limits[1]));
        } else if (this.transformMode === 'rotate') this.callbacks.rotate(this.selected, this.pivot.quaternion.toArray());
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
    new GLTFLoader().load('/api/robot.glb', gltf => {
      if (this.destroyed) return;
      gltf.scene.traverse(node => {
        if (node.name.startsWith('geom_')) this.geoms[node.name.slice(5)] = node;
        if (node instanceof THREE.Mesh) {
          const mats = Array.isArray(node.material) ? node.material : [node.material];
          mats.forEach(mat => { if (mat instanceof THREE.MeshStandardMaterial) { mat.roughness = .65; mat.metalness = .25; } });
        }
      });
      this.scene.add(gltf.scene);
      if (this.state) this.update(this.state);
    }, undefined, () => callbacks.error('G1 모델을 불러오지 못했습니다. 서버 연결을 확인하세요.'));
  }

  private pick(event: PointerEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.ray.setFromCamera(this.pointer, this.camera);
    const keys = [...new Set(this.ray.intersectObjects(Object.values(this.markers).filter(m => m.visible), false).map(hit => hit.object.name))];
    return event.altKey ? keys[(keys.indexOf(this.selected) + 1) % keys.length] : keys[0];
  }
  hover = (event: PointerEvent) => {
    if (!this.editable || event.buttons || event.shiftKey || event.altKey || this.selectionLocked || this.members.length > 1 || this.gizmo.dragging || this.gizmo.axis) return;
    const key = this.pick(event);
    if (key && key !== this.selected) this.callbacks.select(key, false, true);
  };
  click = (event: PointerEvent) => {
    if (!this.editable || this.gizmo.dragging || event.button !== 0) return;
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
    else if (!event.shiftKey) this.selectionLocked = false;
  };

  keydown = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    if (this.keyboardEnabled === false || event.repeat || event.isComposing || event.ctrlKey || event.altKey || event.metaKey || this.gizmo.dragging
        || target?.isContentEditable || target?.closest('input, textarea, select')) return;
    const key = event.code || event.key.toLowerCase();
    if (key === 'KeyF' || key === 'f') {
      event.preventDefault();
      this.focusSelection();
    } else if (this.editable && this.state && ['KeyW', 'KeyE', 'w', 'e'].includes(key)) {
      event.preventDefault();
      this.callbacks.transformMode(key === 'KeyW' || key === 'w' ? 'translate' : 'rotate');
    }
  };

  center() {
    const center = new THREE.Vector3();
    if (this.state) this.members.forEach(k => center.add(new THREE.Vector3().fromArray(this.state!.handles[k].position)));
    return center.divideScalar(this.members.length);
  }

  focusSelection() {
    if (!this.state) return;
    const center = this.center();
    const radius = Math.max(.12, ...this.members.map(k => new THREE.Vector3().fromArray(this.state!.handles[k].position).distanceTo(center) + .10));
    const halfFov = Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * Math.min(1, this.camera.aspect));
    const distance = Math.max(.55, radius / Math.sin(halfFov));
    const direction = this.camera.position.clone().sub(this.orbit.target).normalize();
    this.orbit.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.orbit.update();
    this.dirty = true;
  }

  select(key: string, pins: string[], members = [key]) {
    this.selected = key;
    this.members = members;
    this.pins = pins;
    Object.entries(this.markers).forEach(([k, mesh]) => {
      mesh.visible = this.markerVisible && (isJointHandle(k) === (this.handleLayer === 'joints') || members.includes(k));
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.color.set(members.includes(k) ? '#80f2c7' : pins.includes(k) ? '#f1bc65' : '#56bdec');
      mat.opacity = members.includes(k) ? .64 : .28;
      this.labels[k]?.classList.toggle('chosen', members.includes(k));
      this.labels[k]?.classList.toggle('pinned', pins.includes(k));
    });
    const hinge = this.transformMode === 'rotate' && members.length === 1 ? this.state?.hinges?.[key] : undefined;
    this.gizmo.setSpace(hinge ? 'local' : this.space);
    this.gizmo.showX = this.gizmo.showY = !hinge;
    this.gizmo.showZ = true;
    if (this.state && !this.gizmo.dragging) {
      this.pivot.position.copy(hinge ? new THREE.Vector3().fromArray(hinge.position) : this.center());
      if (hinge) this.pivot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3().fromArray(hinge.axis_world).normalize());
      else this.pivot.quaternion.fromArray(this.state.handles[key].quaternion);
    }
    const blocked = this.transformMode === 'rotate'
      ? !canRotateSelection(members) || members.some(k => pins.includes(k) && (members.length > 1 || k.endsWith('_foot')))
      : members.some(k => pins.includes(k));
    if (this.editable && !blocked && this.markerVisible) this.gizmo.attach(this.pivot); else this.gizmo.detach();
    this.dirty = true;
  }

  update(state: PoseState) {
    this.state = state;
    for (const [id, pose] of Object.entries(state.geoms)) {
      const obj = this.geoms[id];
      if (obj) { obj.position.fromArray(pose.position); obj.quaternion.fromArray(pose.quaternion); }
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

  positionLabels() {
    const rect = this.host.getBoundingClientRect();
    Object.entries(this.markers).forEach(([k, mesh]) => {
      const p = mesh.position.clone().project(this.camera);
      const label = this.labels[k];
      label.style.display = mesh.visible && p.z < 1 && (this.members.includes(k) || this.pins.includes(k)) ? 'block' : 'none';
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
  setEditable(editable: boolean) { this.editable = editable; this.select(this.selected, this.pins, this.members); }
  setTransformMode(mode: TransformMode, space: 'world' | 'local') {
    this.transformMode = mode;
    this.space = space;
    this.gizmo.setMode(mode);
    this.gizmo.setSpace(space);
    this.select(this.selected, this.pins, this.members);
  }
  setBox(position: number[], size: number[], visible: boolean) {
    this.box.visible = visible;
    this.box.position.fromArray(position);
    this.box.scale.set(size[0] / .3, size[1] / .32, size[2] / .24);
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
