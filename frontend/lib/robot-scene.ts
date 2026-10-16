import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { controlKey, controlSelection } from './body-groups.ts';
import { canMirrorSelection, isJointPairSelection } from './pose-transforms.ts';
import { HEAD_CAMERA_LENS_OFFSET, PointCloudLayer } from './point-cloud.ts';
import { normalizedObjectSize, normalizedSceneAsset, type CandidateGraspPoints, type ObjectTransformMode, type SceneObject, type SceneObjectPose } from './scene-objects.ts';
import type { LabPathPreview, HandSide } from './lab-path-apply.ts';

export type TcpOverlay = {
  planned: Record<HandSide, number[][]>; actual: Record<HandSide, number[][]>; now: Record<HandSide, number[]>;
  planned_now: Record<HandSide, number[]> | null;
};

export type PoseState = {
  model_id?: string;
  qpos: number[];
  handles: Record<string, { position: number[]; quaternion: number[]; label: string }>;
  geoms: Record<string, { position: number[]; quaternion: number[] }>;
  grip_pads: Record<'left' | 'right', { position: number[]; quaternion: number[]; size: number[] }>;
  cameras?: Record<'head', { position: number[]; quaternion: number[]; label: string; calibrated_projection: boolean }>;
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
export const canRotateSelection = (members: string[]) => members.length > 0 && (members.every(k => ROTATABLE.includes(k)) || isJointPairSelection(members) || members.length === 1 && (HINGE_HANDLES.includes(members[0]) || isJointHandle(members[0])));
/** Position-pinned handles rotate in place (rotatedGroupTargets sends no position target for them), so
 * only angle pins and pinned feet, whose orientation the solver keeps fixed, block a rotation. */
export const rotationBlocked = (controls: string[], pins: string[], anglePins: string[]) =>
  !canRotateSelection(controls) || controls.some(k => anglePins.includes(k) || pins.includes(k) && k.endsWith('_foot'));
const HEAD_CAMERA_VIEW_FRAME = new THREE.Quaternion().setFromRotationMatrix(
  new THREE.Matrix4().makeBasis(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)),
);
export function headCameraViewQuaternion(linkQuaternion: number[]) {
  return new THREE.Quaternion().fromArray(linkQuaternion).multiply(HEAD_CAMERA_VIEW_FRAME);
}
/** 가상 헤드 카메라 보정값. offset은 d435_link 축(앞, 왼쪽, 위) 기준 mm, rotation은 같은 축 기준 (롤, 피치, 요) 도. */
export type HeadCameraAdjust = { offset: [number, number, number]; rotation: [number, number, number] };
/** 저장된 보정값이 기준으로 삼은 URDF 렌즈 위치의 버전. 값이 바뀌면 옛 저장값은 무시된다. */
const ADJUST_BASE = 'urdf-lens-1';
/** 확정한 점군 기준점 보정(d435_link 기준 mm, 도). 초기값과 "초기화" 버튼이 되돌아가는 값이다. */
export const defaultHeadCameraAdjust = (): HeadCameraAdjust => ({ offset: HEAD_CAMERA_LENS_OFFSET.map(meters => Math.round(meters * 1000)) as [number, number, number], rotation: [0, 0, 0] });
export const NO_HEAD_CAMERA_ADJUST: HeadCameraAdjust = { offset: [0, 0, 0], rotation: [0, 0, 0] };
/** d435_link 자세에 보정값을 적용한 가상 카메라(및 점군)의 월드 위치와 three 카메라 방향. */
export function adjustedHeadCameraPose(linkPosition: number[], linkQuaternion: number[], adjust: HeadCameraAdjust = NO_HEAD_CAMERA_ADJUST) {
  const link = new THREE.Quaternion().fromArray(linkQuaternion);
  const [forward, left, up] = adjust.offset;
  const position = new THREE.Vector3(forward, left, up).multiplyScalar(.001).applyQuaternion(link).add(new THREE.Vector3().fromArray(linkPosition));
  const [roll, pitch, yaw] = adjust.rotation.map(value => THREE.MathUtils.degToRad(value));
  const correction = new THREE.Quaternion().setFromEuler(new THREE.Euler(roll, pitch, yaw, 'ZYX'));
  return { position, quaternion: link.multiply(correction).multiply(HEAD_CAMERA_VIEW_FRAME) };
}
/** D435 컬러 스트림(848×480)의 정렬된 내부 파라미터(fx, fy)에서 계산한 화각. 실제 영상과 가상 카메라를 맞출 때 쓴다. */
export const D435_COLOR_FOV = {
  horizontal: THREE.MathUtils.radToDeg(2 * Math.atan(848 / (2 * 608.126))),
  vertical: THREE.MathUtils.radToDeg(2 * Math.atan(480 / (2 * 608.878))),
};
export function verticalFovForHorizontal(horizontalDegrees: number, aspect: number) {
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(horizontalDegrees / 2)) / aspect));
}
export function horizontalFovForVertical(verticalDegrees: number, aspect: number) {
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(verticalDegrees / 2)) * aspect));
}
export function setPerspectiveFovs(camera: THREE.PerspectiveCamera, horizontalDegrees: number, verticalDegrees: number) {
  const right = camera.near * Math.tan(THREE.MathUtils.degToRad(horizontalDegrees / 2));
  const top = camera.near * Math.tan(THREE.MathUtils.degToRad(verticalDegrees / 2));
  camera.fov = verticalDegrees;
  camera.projectionMatrix.makePerspective(-right, right, top, -top, camera.near, camera.far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}
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
  headCamera = new THREE.PerspectiveCamera(verticalFovForHorizontal(69, 16 / 9), 16 / 9, .03, 20);
  headCameraOverlay: HTMLDivElement;
  headCameraHorizontalFov = 69;
  headCameraVerticalFov = 42;
  headCameraAspect: number | null = null;
  private editorSettingsKey: string;
  private headCameraPosition: { top: number; right: number } | null = null;
  private headCameraMoving = false;
  private headCameraWidth: number | null = null;
  headCameraAvailable = false;
  headCameraSource: 'sim' | 'real' | 'compare' = 'sim';
  headCameraSplit = 50;
  headCameraCompareStyle: 'split' | 'blend' = 'split';
  headCameraBlend = 50;
  headCameraGuides = false;
  headCameraAdjust: HeadCameraAdjust = defaultHeadCameraAdjust();
  private cloud!: PointCloudLayer;
  private headCameraStatusTimer: number | null = null;
  renderer: THREE.WebGLRenderer;
  orbit: OrbitControls;
  gizmo: TransformControls;
  pivot = new THREE.Object3D();
  markers: Record<string, THREE.Mesh> = {};
  labels: Record<string, HTMLDivElement> = {};
  geoms: Record<string, THREE.Object3D> = {};
  referenceGeoms: Record<string, THREE.Object3D> = {};
  referenceOnly: Set<string> | null = null;
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
  candidateGraspMarkers: Record<'left' | 'right', THREE.Group>;
  labPathPreview = new THREE.Group();
  tcpOverlay = new THREE.Group();
  comparePaths = new THREE.Group();
  gripPads: Record<'left' | 'right', THREE.Mesh>;
  transformMode: TransformMode = 'translate';
  space: 'world' | 'local' = 'world';
  hingeDrag: { quaternion: THREE.Quaternion; angle: number; lastTwist: number; delta: number; limits: number[]; key: string; component: 'x' | 'y' | 'z' } | null = null;
  modelMismatchReported = false;

  constructor(public host: HTMLDivElement, private callbacks: Callbacks, public modelId = 'g1', visualRevision = '') {
    this.editorSettingsKey = `motioncreator-editor-view-v1-${modelId}`;
    this.scene.background = new THREE.Color('#10171f');
    this.scene.fog = new THREE.Fog('#10171f', 5, 14);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.headCameraOverlay = document.createElement('div');
    this.headCameraOverlay.className = 'head-camera-overlay';
    this.headCameraOverlay.innerHTML = '<button class="head-camera-settings-button" type="button" aria-label="PiP 카메라 설정" aria-expanded="false" aria-controls="head-camera-settings">⚙</button><img class="head-camera-feed" alt="실제 HEAD CAM 영상" hidden><div class="head-camera-resize" role="separator" aria-label="PiP 크기 조절" title="이동 모드에서 드래그해 크기 조절"></div><div class="head-camera-guides" hidden></div><b>HEAD CAM · SIM</b><button class="head-camera-move-button" type="button" aria-label="PiP 이동 시작" aria-pressed="false" title="PiP 위치 이동">⤧</button><div id="head-camera-settings" class="head-camera-controls" hidden><label class="head-camera-cloud" title="로봇 D435 깊이로 만든 3D 점군을 장면에 표시"><input type="checkbox" class="head-camera-cloud-toggle" aria-label="3D 포인트 클라우드 표시"><span>3D 포인트 클라우드</span><small></small></label><div class="head-camera-source" role="group" aria-label="PiP 카메라 소스"><button type="button" data-source="sim" aria-pressed="true">가상 카메라</button><button type="button" data-source="real" aria-pressed="false" title="로봇 D435 영상 (teleimager ZMQ)">실제 카메라</button><button type="button" data-source="compare" aria-pressed="false" title="실제 영상(왼쪽)과 가상 카메라(오른쪽)를 한 화면에서 비교">비교</button></div><div class="head-camera-compare" hidden><div class="head-camera-aspects head-camera-styles" role="group" aria-label="비교 방식"><button type="button" data-style="split" aria-pressed="true" title="경계선 왼쪽은 실제, 오른쪽은 가상">분할</button><button type="button" data-style="blend" aria-pressed="false" title="실제 영상을 가상 렌더 위에 반투명하게 겹침">겹치기</button></div><label class="head-camera-split"><span>분할</span><input type="range" min="0" max="100" step="1" value="50" aria-label="비교 분할 위치"><output>50%</output></label><label class="head-camera-blend" hidden><span>투명도</span><input type="range" min="0" max="100" step="1" value="50" aria-label="실제 영상 불투명도"><output>50%</output></label><label class="head-camera-guides-toggle"><input type="checkbox" aria-label="가이드선 표시"><span>가이드선 (중앙 십자·3분할)</span></label></div><details class="head-camera-adjust"><summary>포인트클라우드 기준점 보정 (d435_link 기준)</summary><div class="head-camera-adjust-rows"></div><div class="head-camera-adjust-actions"><button type="button" data-action="reset">초기화</button><button type="button" data-action="copy" title="점군 기준점 보정값과 화각을 클립보드에 JSON으로 복사">값 복사</button></div></details><div class="head-camera-controls-title"><strong>PiP 카메라 화각 (°)</strong><button type="button" aria-label="D435 RGB 화각으로 초기화" title="D435 RGB · 가로 69° / 세로 42°">↺</button></div><div class="head-camera-aspects" role="group" aria-label="PiP 화면 비율"><button type="button" data-aspect="free" aria-pressed="true">자유</button><button type="button" data-aspect="4:3" aria-pressed="false">4:3</button><button type="button" data-aspect="16:9" aria-pressed="false">16:9</button></div><label><span>가로</span><input type="range" min="1" max="120" step="0.01" value="69" aria-label="PiP 가로 화각 슬라이더"><input type="text" inputmode="decimal" min="1" max="120" step="0.01" value="69" aria-label="PiP 가로 화각 수치"></label><label><span>세로</span><input type="range" min="1" max="120" step="0.01" value="42" aria-label="PiP 세로 화각 슬라이더"><input type="text" inputmode="decimal" min="1" max="120" step="0.01" value="42" aria-label="PiP 세로 화각 수치"></label></div>';
    const moveButton = this.headCameraOverlay.querySelector<HTMLButtonElement>('.head-camera-move-button')!;
    moveButton.addEventListener('click', () => {
      this.headCameraMoving = !this.headCameraMoving;
      this.headCameraOverlay.classList.toggle('moving', this.headCameraMoving);
      moveButton.setAttribute('aria-pressed', String(this.headCameraMoving));
      moveButton.setAttribute('aria-label', this.headCameraMoving ? 'PiP 위치 고정' : 'PiP 이동 시작');
      moveButton.title = this.headCameraMoving ? '현재 위치에 고정' : 'PiP 위치 이동';
      if (!this.headCameraMoving) this.saveEditorViewSettings();
    });
    let drag: { x: number; y: number; top: number; right: number } | null = null;
    let resize: { x: number; width: number; right: number } | null = null;
    this.headCameraOverlay.addEventListener('pointerdown', event => {
      if (!this.headCameraMoving) return;
      if ((event.target as HTMLElement).closest('.head-camera-resize')) {
        const start = this.headCameraRect(this.host.clientWidth, this.host.clientHeight);
        resize = { x: event.clientX, width: start.width, right: start.right };
        this.headCameraOverlay.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }
      if ((event.target as HTMLElement).closest('button,input,.head-camera-controls')) return;
      const rect = this.headCameraRect(this.host.clientWidth, this.host.clientHeight);
      drag = { x: event.clientX, y: event.clientY, top: rect.top, right: rect.right };
      this.headCameraOverlay.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    this.headCameraOverlay.addEventListener('pointermove', event => {
      if (resize) {
        // 오른쪽 아래 핸들: 왼쪽 가장자리를 고정하고 오른쪽으로 늘어난 만큼 right 여백을 줄인다.
        const grow = Math.min(event.clientX - resize.x, resize.right - 4);
        this.headCameraWidth = resize.width + grow;
        this.headCameraPosition = { top: this.headCameraPosition?.top ?? 52, right: resize.right - grow };
        const rect = this.headCameraRect(this.host.clientWidth, this.host.clientHeight);
        this.headCameraWidth = rect.width;
        this.headCameraPosition = { top: rect.top, right: resize.right - (rect.width - resize.width) };
        this.dirty = true;
        return;
      }
      if (!drag) return;
      this.headCameraPosition = { top: drag.top + event.clientY - drag.y, right: drag.right - (event.clientX - drag.x) };
      const rect = this.headCameraRect(this.host.clientWidth, this.host.clientHeight);
      this.headCameraPosition = { top: rect.top, right: rect.right };
      this.dirty = true;
    });
    const finishMove = () => { if (drag || resize) { drag = null; resize = null; this.saveEditorViewSettings(); } };
    this.headCameraOverlay.addEventListener('pointerup', finishMove);
    this.headCameraOverlay.addEventListener('pointercancel', finishMove);
    const settingsButton = this.headCameraOverlay.querySelector<HTMLButtonElement>('.head-camera-settings-button')!;
    const settings = this.headCameraOverlay.querySelector<HTMLDivElement>('.head-camera-controls')!;
    settingsButton.addEventListener('click', () => {
      settings.hidden = !settings.hidden;
      settingsButton.setAttribute('aria-expanded', String(!settings.hidden));
    });
    const fovInput = (label: string) => this.headCameraOverlay.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
    const horizontalRange = fovInput('PiP 가로 화각 슬라이더'), verticalRange = fovInput('PiP 세로 화각 슬라이더');
    const horizontalValue = fovInput('PiP 가로 화각 수치'), verticalValue = fovInput('PiP 세로 화각 수치');
    const aspectButtons = this.headCameraOverlay.querySelectorAll<HTMLButtonElement>('.head-camera-aspects button');
    const showFovs = () => {
      horizontalRange.value = String(this.headCameraHorizontalFov);
      verticalRange.value = String(this.headCameraVerticalFov);
      horizontalValue.value = String(Number(this.headCameraHorizontalFov.toFixed(2)));
      verticalValue.value = String(Number(this.headCameraVerticalFov.toFixed(2)));
      this.dirty = true;
    };
    this.syncHeadCameraFovControls = () => {
      showFovs();
      const choice = this.headCameraAspect === 4 / 3 ? '4:3' : this.headCameraAspect === 16 / 9 ? '16:9' : 'free';
      aspectButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.aspect === choice)));
    };
    const setAspect = (choice: string) => {
      this.headCameraAspect = choice === '4:3' ? 4 / 3 : choice === '16:9' ? 16 / 9 : null;
      aspectButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.aspect === choice)));
      if (this.headCameraAspect) {
        this.headCameraHorizontalFov = Math.max(this.headCameraHorizontalFov, horizontalFovForVertical(1, this.headCameraAspect));
        this.headCameraVerticalFov = verticalFovForHorizontal(this.headCameraHorizontalFov, this.headCameraAspect);
      }
      showFovs();
      this.saveEditorViewSettings();
    };
    aspectButtons.forEach(button => button.addEventListener('click', () => setAspect(button.dataset.aspect!)));
    this.headCameraOverlay.querySelector<HTMLButtonElement>('.head-camera-controls-title button')!.addEventListener('click', () => {
      this.headCameraHorizontalFov = 69;
      this.headCameraVerticalFov = 42;
      setAspect('free');
    });
    const setHorizontal = (value: number) => {
      if (!Number.isFinite(value)) { showFovs(); return; }
      this.headCameraHorizontalFov = Math.min(120, Math.max(1, value));
      if (this.headCameraAspect) {
        this.headCameraHorizontalFov = Math.max(this.headCameraHorizontalFov, horizontalFovForVertical(1, this.headCameraAspect));
        this.headCameraVerticalFov = verticalFovForHorizontal(this.headCameraHorizontalFov, this.headCameraAspect);
      }
      showFovs();
      this.saveEditorViewSettings();
    };
    const setVertical = (value: number) => {
      if (!Number.isFinite(value)) { showFovs(); return; }
      this.headCameraVerticalFov = Math.min(120, Math.max(1, value));
      if (this.headCameraAspect) {
        this.headCameraHorizontalFov = Math.min(120, Math.max(1, horizontalFovForVertical(this.headCameraVerticalFov, this.headCameraAspect)));
        this.headCameraVerticalFov = verticalFovForHorizontal(this.headCameraHorizontalFov, this.headCameraAspect);
      }
      showFovs();
      this.saveEditorViewSettings();
    };
    horizontalRange.addEventListener('input', () => setHorizontal(Number(horizontalRange.value)));
    verticalRange.addEventListener('input', () => setVertical(Number(verticalRange.value)));
    horizontalValue.addEventListener('change', () => setHorizontal(horizontalValue.value.trim() ? Number(horizontalValue.value) : NaN));
    verticalValue.addEventListener('change', () => setVertical(verticalValue.value.trim() ? Number(verticalValue.value) : NaN));
    this.headCameraOverlay.querySelectorAll<HTMLButtonElement>('.head-camera-source button').forEach(button =>
      button.addEventListener('click', () => {
        this.setHeadCameraSource(button.dataset.source === 'real' ? 'real' : button.dataset.source === 'compare' ? 'compare' : 'sim');
        this.saveEditorViewSettings();
      }));
    this.setupHeadCameraCompareControls();
    const splitRange = this.headCameraOverlay.querySelector<HTMLInputElement>('.head-camera-split input')!;
    splitRange.addEventListener('input', () => this.setHeadCameraSplit(Number(splitRange.value)));
    splitRange.addEventListener('change', () => this.saveEditorViewSettings());
    this.headCameraOverlay.hidden = true;
    const cloudToggle = this.headCameraOverlay.querySelector<HTMLInputElement>('.head-camera-cloud-toggle')!;
    const cloudStatus = this.headCameraOverlay.querySelector<HTMLElement>('.head-camera-cloud small')!;
    this.cloud = new PointCloudLayer(() => { this.dirty = true; }, text => { cloudStatus.textContent = text; });
    cloudToggle.addEventListener('change', () => {
      this.setPointCloudEnabled(cloudToggle.checked);
      this.saveEditorViewSettings();
    });
    host.appendChild(this.headCameraOverlay);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.4, -2.8, 1.85);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.target.set(0, 0, .7);
    this.loadEditorViewSettings();
    this.orbit.enableDamping = false;
    this.orbit.minDistance = .5;
    this.orbit.maxDistance = 10;
    this.orbit.maxPolarAngle = Math.PI * .49;
    this.orbit.update();
    this.orbit.addEventListener('change', () => { this.dirty = true; });
    this.orbit.addEventListener('end', () => this.saveEditorViewSettings());
    this.scene.add(this.cloud.points);
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
        const controls = controlSelection(this.members);
        const hinge = this.transformMode === 'rotate' && (controls.length === 1 || isJointPairSelection(controls)) && ring ? this.state?.hinges?.[ring.key] : undefined;
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
    this.candidateGraspMarkers = Object.fromEntries((['left', 'right'] as const).map(side => {
      const color = side === 'left' ? '#55e7c1' : '#ffbd70';
      const marker = new THREE.Group();
      const material = new THREE.MeshBasicMaterial({ color, depthTest: false });
      marker.add(new THREE.Mesh(new THREE.SphereGeometry(.018, 16, 12), material));
      marker.add(new THREE.Mesh(new THREE.TorusGeometry(.04, .004, 6, 24), material.clone()));
      marker.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), .09, 0xff0000, .025, .012));
      marker.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), .06, 0x00ff00, .018, .009));
      marker.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), .06, 0x0000ff, .018, .009));
      marker.traverse(node => { node.renderOrder = 6; });
      marker.visible = false; this.scene.add(marker); return [side, marker];
    })) as unknown as Record<'left' | 'right', THREE.Group>;
    this.scene.add(this.labPathPreview, this.tcpOverlay, this.comparePaths);
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
        const size = this.renderer.getSize(new THREE.Vector2());
        this.renderer.setScissorTest(false);
        this.renderer.setViewport(0, 0, size.x, size.y);
        this.renderer.render(this.scene, this.camera);
        if (this.headCameraAvailable) this.renderHeadCamera(size.x, size.y);
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
      this.setReferenceOnly(this.referenceOnly);
      this.referenceRoot = reference;
      this.scene.add(reference);
      if (this.referenceState) this.updateReference(this.referenceState);
      if (this.state) this.update(this.state);
    }, undefined, () => callbacks.error('G1 모델을 불러오지 못했습니다. 서버 연결을 확인하세요.'));
  }

  private saveEditorViewSettings() {
    try {
      window.localStorage.setItem(this.editorSettingsKey, JSON.stringify({
        camera: { position: this.camera.position.toArray(), target: this.orbit.target.toArray() },
        pip: { position: this.headCameraPosition, width: this.headCameraWidth, horizontalFov: (this.preCompareView ?? { horizontalFov: this.headCameraHorizontalFov }).horizontalFov,
          verticalFov: (this.preCompareView ?? { verticalFov: this.headCameraVerticalFov }).verticalFov, aspect: this.preCompareView ? this.preCompareView.aspect : this.headCameraAspect, source: this.headCameraSource, split: this.headCameraSplit,
          compareStyle: this.headCameraCompareStyle, blend: this.headCameraBlend, guides: this.headCameraGuides, adjust: this.headCameraAdjust, adjustBase: ADJUST_BASE, cloud: this.cloud.running },
      }));
    } catch { /* 에디터 설정 저장이 불가능해도 편집은 계속합니다. */ }
  }

  private loadEditorViewSettings() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(this.editorSettingsKey) ?? 'null');
      const vector = (value: unknown): value is number[] => Array.isArray(value) && value.length === 3
        && value.every(component => typeof component === 'number' && Number.isFinite(component));
      if (vector(saved?.camera?.position) && vector(saved?.camera?.target)) {
        this.camera.position.fromArray(saved.camera.position);
        this.orbit.target.fromArray(saved.camera.target);
        this.orbit.update();
      }
      const pip = saved?.pip;
      if (Number.isFinite(pip?.position?.top) && Number.isFinite(pip?.position?.right)) {
        this.headCameraPosition = { top: pip.position.top, right: pip.position.right };
      }
      if (Number.isFinite(pip?.width) && pip.width > 0) this.headCameraWidth = pip.width;
      if (Number.isFinite(pip?.horizontalFov) && pip.horizontalFov >= 1 && pip.horizontalFov <= 120) this.headCameraHorizontalFov = pip.horizontalFov;
      if (Number.isFinite(pip?.verticalFov) && pip.verticalFov >= 1 && pip.verticalFov <= 120) this.headCameraVerticalFov = pip.verticalFov;
      if (pip?.aspect === null || pip?.aspect === 4 / 3 || pip?.aspect === 16 / 9) this.headCameraAspect = pip.aspect;
      this.syncHeadCameraFovControls();
      if (Number.isFinite(pip?.split) && pip.split >= 0 && pip.split <= 100) this.setHeadCameraSplit(pip.split);
      if (Number.isFinite(pip?.blend) && pip.blend >= 0 && pip.blend <= 100) this.setHeadCameraBlend(pip.blend);
      if (pip?.compareStyle === 'blend') this.setHeadCameraCompareStyle('blend');
      if (pip?.guides === true) this.setHeadCameraGuides(true);
      // 렌즈 위치가 URDF에 반영되기 전에 저장된 보정값은 같은 오프셋을 한 번 더 더하게 되므로 버린다.
      const adjustment = pip?.adjustBase === ADJUST_BASE ? pip?.adjust : undefined;
      if (vector(adjustment?.offset) && vector(adjustment?.rotation)) {
        this.headCameraAdjust = { offset: [...adjustment.offset] as [number, number, number], rotation: [...adjustment.rotation] as [number, number, number] };
        this.refreshHeadCameraAdjust();
      }
      if (pip?.source === 'real' || pip?.source === 'compare') this.setHeadCameraSource(pip.source);
      if (pip?.cloud === true) this.setPointCloudEnabled(true);
      const choice = this.headCameraAspect === 4 / 3 ? '4:3' : this.headCameraAspect === 16 / 9 ? '16:9' : 'free';
      this.headCameraOverlay.querySelectorAll<HTMLButtonElement>('.head-camera-aspects button').forEach(button =>
        button.setAttribute('aria-pressed', String(button.dataset.aspect === choice)));
    } catch { /* 손상된 로컬 설정은 기본값으로 시작합니다. */ }
  }

  private setupHeadCameraCompareControls() {
    const overlay = this.headCameraOverlay;
    const rows = overlay.querySelector<HTMLDivElement>('.head-camera-adjust-rows')!;
    const specs = [
      { group: 'offset', index: 0, label: '앞뒤', unit: 'mm', limit: 100, step: .5, hint: '점군 기준점을 d435_link 앞(+)/뒤(−)로: +는 카메라에서 멀어져 작게, −는 가까워져 크게 보임' },
      { group: 'offset', index: 1, label: '왼쪽', unit: 'mm', limit: 100, step: .5, hint: '기준점을 왼쪽(+)/오른쪽(−)으로' },
      { group: 'offset', index: 2, label: '위', unit: 'mm', limit: 100, step: .5, hint: '기준점을 위(+)/아래(−)로' },
      { group: 'rotation', index: 0, label: '롤', unit: '°', limit: 10, step: .05, hint: '점군을 앞 축 기준으로 기울임: 왼쪽이 올라가면 +' },
      { group: 'rotation', index: 1, label: '피치', unit: '°', limit: 10, step: .05, hint: '점군의 시선 방향을 아래로 숙이면 +' },
      { group: 'rotation', index: 2, label: '요', unit: '°', limit: 10, step: .05, hint: '점군의 시선 방향을 왼쪽으로 돌리면 +' },
    ] as const;
    const refresh: Array<() => void> = [];
    for (const spec of specs) {
      const row = document.createElement('label');
      row.title = spec.hint;
      row.innerHTML = `<span>${spec.label}</span><input type="range" min="${-spec.limit}" max="${spec.limit}" step="${spec.step}" value="0" aria-label="${spec.label} 보정 슬라이더"><button type="button" class="head-camera-step" data-direction="-1" aria-label="${spec.label} 감소">−</button><input type="text" inputmode="decimal" value="0" aria-label="${spec.label} 보정 수치"><button type="button" class="head-camera-step" data-direction="1" aria-label="${spec.label} 증가">+</button><small>${spec.unit}</small>`;
      const [range, text] = row.querySelectorAll<HTMLInputElement>('input');
      const nudge = spec.group === 'offset' ? 1 : .1;
      const show = () => {
        const value = this.headCameraAdjust[spec.group][spec.index];
        range.value = String(value);
        text.value = String(Number(value.toFixed(2)));
      };
      const apply = (value: number) => {
        if (!Number.isFinite(value)) { show(); return; }
        this.headCameraAdjust[spec.group][spec.index] = Math.min(spec.limit * 5, Math.max(-spec.limit * 5, value));
        show();
        this.applyHeadCameraPose();
      };
      row.querySelectorAll<HTMLButtonElement>('.head-camera-step').forEach(button => button.addEventListener('click', () => {
        const next = this.headCameraAdjust[spec.group][spec.index] + Number(button.dataset.direction) * nudge;
        apply(Math.round(next * 100) / 100);
        this.saveEditorViewSettings();
      }));
      range.addEventListener('input', () => apply(Number(range.value)));
      range.addEventListener('change', () => this.saveEditorViewSettings());
      text.addEventListener('change', () => { apply(text.value.trim() ? Number(text.value) : NaN); this.saveEditorViewSettings(); });
      refresh.push(show);
      rows.appendChild(row);
    }
    this.refreshHeadCameraAdjust = () => refresh.forEach(show => show());
    overlay.querySelector<HTMLButtonElement>('[data-action="reset"]')!.addEventListener('click', () => {
      this.headCameraAdjust = defaultHeadCameraAdjust();
      this.refreshHeadCameraAdjust();
      this.applyHeadCameraPose();
      this.saveEditorViewSettings();
    });
    overlay.querySelector<HTMLButtonElement>('[data-action="copy"]')!.addEventListener('click', () => {
      const text = JSON.stringify({ offset_mm: this.headCameraAdjust.offset, rotation_deg: this.headCameraAdjust.rotation,
        horizontal_fov: this.headCameraHorizontalFov, vertical_fov: this.headCameraVerticalFov });
      navigator.clipboard?.writeText(text).catch(() => {});
    });
    overlay.querySelectorAll<HTMLButtonElement>('.head-camera-styles button').forEach(button =>
      button.addEventListener('click', () => {
        this.setHeadCameraCompareStyle(button.dataset.style === 'blend' ? 'blend' : 'split');
        this.saveEditorViewSettings();
      }));
    const blend = overlay.querySelector<HTMLInputElement>('.head-camera-blend input')!;
    blend.addEventListener('input', () => this.setHeadCameraBlend(Number(blend.value)));
    blend.addEventListener('change', () => this.saveEditorViewSettings());
    const guides = overlay.querySelector<HTMLInputElement>('.head-camera-guides-toggle input')!;
    guides.addEventListener('change', () => { this.setHeadCameraGuides(guides.checked); this.saveEditorViewSettings(); });
  }

  refreshHeadCameraAdjust: () => void = () => {};
  syncHeadCameraFovControls: () => void = () => {};
  /** 비교 모드에 들어가기 전 사용자의 PiP 화각/비율. 비교 모드에서는 실제 D435 RGB 값으로 고정된다. */
  private preCompareView: { horizontalFov: number; verticalFov: number; aspect: number | null } | null = null;

  /** 마지막 d435_link 자세로 가상 카메라를 놓고, 보정값을 적용한 기준점에 점군을 놓는다. */
  private applyHeadCameraPose() {
    const head = this.state?.cameras?.head;
    if (head) {
      const pose = adjustedHeadCameraPose(head.position, head.quaternion);     // 가상 카메라는 d435_link 그대로 둔다
      this.headCamera.position.copy(pose.position);
      this.headCamera.quaternion.copy(pose.quaternion);
    }
    this.placeCloud();
    this.dirty = true;
  }

  setHeadCameraCompareStyle(style: 'split' | 'blend') {
    this.headCameraCompareStyle = style;
    this.headCameraOverlay.classList.toggle('blend', style === 'blend');
    this.headCameraOverlay.querySelectorAll<HTMLButtonElement>('.head-camera-styles button').forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.style === style)));
    this.headCameraOverlay.querySelector<HTMLElement>('.head-camera-split')!.hidden = style !== 'split';
    this.headCameraOverlay.querySelector<HTMLElement>('.head-camera-blend')!.hidden = style !== 'blend';
    this.dirty = true;
  }

  setHeadCameraBlend(percent: number) {
    this.headCameraBlend = THREE.MathUtils.clamp(Math.round(percent), 0, 100);
    this.headCameraOverlay.style.setProperty('--blend', String(this.headCameraBlend / 100));
    this.headCameraOverlay.querySelector<HTMLInputElement>('.head-camera-blend input')!.value = String(this.headCameraBlend);
    this.headCameraOverlay.querySelector<HTMLOutputElement>('.head-camera-blend output')!.value = `${this.headCameraBlend}%`;
    this.dirty = true;
  }

  setHeadCameraGuides(on: boolean) {
    this.headCameraGuides = on;
    this.headCameraOverlay.querySelector<HTMLElement>('.head-camera-guides')!.hidden = !on;
    this.headCameraOverlay.querySelector<HTMLInputElement>('.head-camera-guides-toggle input')!.checked = on;
    this.dirty = true;
  }

  /** 비교 모드에서 실제 영상이 차지하는 왼쪽 비율(%). 나머지 오른쪽은 가상 카메라 렌더가 보인다. */
  setHeadCameraSplit(percent: number) {
    this.headCameraSplit = THREE.MathUtils.clamp(Math.round(percent), 0, 100);
    this.headCameraOverlay.style.setProperty('--split', `${this.headCameraSplit}%`);
    this.headCameraOverlay.querySelector<HTMLInputElement>('.head-camera-split input')!.value = String(this.headCameraSplit);
    this.headCameraOverlay.querySelector<HTMLOutputElement>('.head-camera-split output')!.value = `${this.headCameraSplit}%`;
    this.dirty = true;
  }

  setHeadCameraSource(source: 'sim' | 'real' | 'compare') {
    const wasCompare = this.headCameraSource === 'compare';
    this.headCameraSource = source;
    if (source === 'compare' && !wasCompare) {
      this.preCompareView = { horizontalFov: this.headCameraHorizontalFov, verticalFov: this.headCameraVerticalFov, aspect: this.headCameraAspect };
      this.headCameraHorizontalFov = D435_COLOR_FOV.horizontal;
      this.headCameraVerticalFov = D435_COLOR_FOV.vertical;
      this.headCameraAspect = 16 / 9;
    } else if (source !== 'compare' && wasCompare && this.preCompareView) {
      ({ horizontalFov: this.headCameraHorizontalFov, verticalFov: this.headCameraVerticalFov, aspect: this.headCameraAspect } = this.preCompareView);
      this.preCompareView = null;
    }
    this.syncHeadCameraFovControls();
    const real = source === 'real';
    const live = source !== 'sim';
    const prefix = source === 'compare' ? 'HEAD CAM · 비교' : 'HEAD CAM · REAL';
    const feed = this.headCameraOverlay.querySelector<HTMLImageElement>('.head-camera-feed')!;
    const label = this.headCameraOverlay.querySelector<HTMLElement>('b')!;
    this.headCameraOverlay.classList.toggle('real', real);
    this.headCameraOverlay.classList.toggle('compare', source === 'compare');
    this.headCameraOverlay.querySelector<HTMLElement>('.head-camera-compare')!.hidden = source !== 'compare';
    this.headCameraOverlay.style.setProperty('--split', `${this.headCameraSplit}%`);
    this.headCameraOverlay.querySelectorAll<HTMLButtonElement>('.head-camera-source button').forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.source === source)));
    this.headCameraOverlay.querySelectorAll<HTMLInputElement | HTMLButtonElement>('.head-camera-controls input:not(.head-camera-cloud-toggle):not(.head-camera-compare input):not(.head-camera-adjust input), .head-camera-controls-title button').forEach(control => { control.disabled = real || source === 'compare'; });
    this.headCameraOverlay.querySelectorAll<HTMLButtonElement>('.head-camera-aspects:not(.head-camera-styles) button').forEach(button => { button.disabled = source === 'compare'; });
    if (this.headCameraStatusTimer !== null) { window.clearInterval(this.headCameraStatusTimer); this.headCameraStatusTimer = null; }
    if (!live) {
      feed.hidden = true;
      feed.removeAttribute('src');
      label.textContent = 'HEAD CAM · SIM';
    } else {
      feed.hidden = false;
      feed.src = '/api/head-camera/stream';
      label.textContent = `${prefix} · 연결 중`;
      const poll = async () => {
        try {
          const status = await (await fetch('/api/head-camera/status')).json() as { receiving: boolean; width: number | null; height: number | null; fps: number | null };
          const size = status.receiving && status.width && status.height ? ` ${status.width}×${status.height}` : '';
          const rate = status.receiving && status.fps ? ` · ${status.fps.toFixed(1)}fps` : '';
          label.textContent = `${prefix} · ${status.receiving ? `LIVE${size}${rate}` : '신호 없음'}`;
        } catch { label.textContent = `${prefix} · 서버 오류`; }
      };
      this.headCameraStatusTimer = window.setInterval(() => { void poll(); }, 1000);
      void poll();
    }
    this.dirty = true;
  }

  setPointCloudEnabled(enabled: boolean) {
    this.headCameraOverlay.querySelector<HTMLInputElement>('.head-camera-cloud-toggle')!.checked = enabled;
    if (enabled) this.cloud.start(); else this.cloud.stop();
    this.placeCloud();
    this.dirty = true;
  }

  private placeCloud() {
    const head = this.state?.cameras?.head;
    if (head) {
      // 점군의 기준점만 d435_link에서 보정값만큼 옮긴다. 보정값은 링크 기준 실제 컬러 렌즈의 위치/자세 추정치다.
      const pose = adjustedHeadCameraPose(head.position, head.quaternion, this.headCameraAdjust);
      this.cloud.place(pose.position.toArray(), pose.quaternion);
    }
    this.cloud.points.visible = this.cloud.running && !!head;
  }

  private headCameraRect(width: number, height: number) {
    const maxWidth = Math.max(120, Math.min(width - 8, (height - 8) * (this.headCameraAspect ?? 16 / 9), 960));
    const frameWidth = this.headCameraWidth ? THREE.MathUtils.clamp(this.headCameraWidth, 120, maxWidth) : Math.min(220, Math.max(150, width * .28));
    const frameHeight = frameWidth / (this.headCameraAspect ?? 16 / 9);
    const right = THREE.MathUtils.clamp(this.headCameraPosition?.right ?? 18, 4, Math.max(4, width - frameWidth - 4));
    const top = THREE.MathUtils.clamp(this.headCameraPosition?.top ?? 52, 4, Math.max(4, height - frameHeight - 4));
    return { x: width - frameWidth - right, y: height - frameHeight - top,
      width: frameWidth, height: frameHeight, right, top };
  }

  private renderHeadCamera(width: number, height: number) {
    const rect = this.headCameraRect(width, height);
    this.headCamera.aspect = rect.width / rect.height;
    setPerspectiveFovs(this.headCamera, this.headCameraHorizontalFov, this.headCameraVerticalFov);
    this.headCameraOverlay.style.width = `${rect.width}px`;
    this.headCameraOverlay.style.height = `${rect.height}px`;
    this.headCameraOverlay.style.right = `${rect.right}px`;
    this.headCameraOverlay.style.top = `${rect.top}px`;
    if (this.headCameraSource === 'real') return;
    const hidden = [this.gizmo.getHelper(), ...Object.values(this.markers), ...Object.values(this.gripMarkers),
      ...Object.values(this.gripPads), this.referenceRoot, ...Object.values(this.collisionProxies)]
      .filter((value): value is THREE.Object3D => !!value)
      .map(object => [object, object.visible] as const);
    hidden.forEach(([object]) => { object.visible = false; });
    this.renderer.setViewport(rect.x, rect.y, rect.width, rect.height);
    this.renderer.setScissor(rect.x, rect.y, rect.width, rect.height);
    this.renderer.setScissorTest(true);
    this.renderer.render(this.scene, this.headCamera);
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, width, height);
    hidden.forEach(([object, visible]) => { object.visible = visible; });
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
    this.saveEditorViewSettings();
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
    const hinge = this.transformMode === 'rotate' && (controls.length === 1 || isJointPairSelection(controls)) ? this.state?.hinges?.[activeControl] : undefined;
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
      ? rotationBlocked(controls, pins, anglePins)
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
    const headCamera = state.cameras?.head;
    this.headCameraAvailable = !!headCamera;
    this.headCameraOverlay.hidden = !headCamera;
    this.applyHeadCameraPose();
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

  /** Show only these geoms (by id) of the reference robot, e.g. just the arms; null shows all of it. */
  setReferenceOnly(ids: Iterable<string> | null) {
    this.referenceOnly = ids ? new Set(ids) : null;
    for (const [id, object] of Object.entries(this.referenceGeoms)) object.visible = !this.referenceOnly || this.referenceOnly.has(id);
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
    this.saveEditorViewSettings();
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
  setCandidateGraspMarkers(points: CandidateGraspPoints | null) {
    for (const side of ['left', 'right'] as const) {
      const marker = this.candidateGraspMarkers[side];
      marker.visible = !!points;
      if (points) {
        marker.position.fromArray(points[side]);
        marker.quaternion.fromArray(points.orientation[side]);
      }
    }
    this.dirty = true;
  }
  /** Planned (dim) versus actual (bright) tool-tip paths, the current tips and the gap between them. */
  setTcpOverlay(overlay: TcpOverlay | null) {
    this.tcpOverlay.traverse(node => {
      if (node instanceof THREE.Mesh || node instanceof THREE.Line) {
        node.geometry.dispose();
        (Array.isArray(node.material) ? node.material : [node.material]).forEach(material => material.dispose());
      }
    });
    this.tcpOverlay.clear();
    if (overlay) for (const side of ['left', 'right'] as HandSide[]) {
      const color = side === 'left' ? 0x55e7c1 : 0xffbd70;
      const vector = (point: number[]) => new THREE.Vector3().fromArray(point);
      const line = (points: number[][], opacity: number, lineColor: number) => {
        if (points.length < 2) return;
        const object = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(vector)),
          new THREE.LineBasicMaterial({ color: lineColor, depthTest: false, transparent: true, opacity }));
        object.renderOrder = 7; this.tcpOverlay.add(object);
      };
      const dot = (point: number[], radius: number, dotColor: number) => {
        const marker = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 8), new THREE.MeshBasicMaterial({ color: dotColor, depthTest: false }));
        marker.position.fromArray(point); marker.renderOrder = 8; this.tcpOverlay.add(marker);
      };
      line(overlay.planned[side], .45, color);
      line(overlay.actual[side], 1, 0xffffff);
      dot(overlay.now[side], .014, 0xffffff);
      const planned = overlay.planned_now?.[side];
      if (planned) { dot(planned, .011, color); line([planned, overlay.now[side]], .95, 0xff5a4d); }
    }
    this.dirty = true;
  }

  /** Hand paths of the original motion (grey) and the dynamic grasp result (coloured), one polyline per hand. */
  setComparePaths(paths: { original: Record<HandSide, number[][]>; resolved: Record<HandSide, number[][]> } | null) {
    this.comparePaths.traverse(node => {
      if (node instanceof THREE.Line) { node.geometry.dispose(); (node.material as THREE.Material).dispose(); }
    });
    this.comparePaths.clear();
    if (paths) for (const side of ['left', 'right'] as HandSide[]) {
      const draw = (points: number[][], color: number, opacity: number) => {
        if (points.length < 2) return;
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(point => new THREE.Vector3().fromArray(point))),
          new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity }));
        line.renderOrder = 7; this.comparePaths.add(line);
      };
      draw(paths.original[side], 0xb8c2cc, .8);
      draw(paths.resolved[side], side === 'left' ? 0x55e7c1 : 0xffbd70, 1);
    }
    this.dirty = true;
  }

  setLabPathPreview(preview: LabPathPreview | null) {
    this.labPathPreview.traverse(node => {
      if (node instanceof THREE.Mesh || node instanceof THREE.Line) {
        node.geometry.dispose();
        (Array.isArray(node.material) ? node.material : [node.material]).forEach(material => material.dispose());
      }
    });
    this.labPathPreview.clear();
    if (preview) for (const side of ['left', 'right'] as HandSide[]) {
      const track = preview.tracks[side];
      if (!track) continue;
      const color = side === 'left' ? 0x55e7c1 : 0xffbd70;
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(track.points.map(point =>
        new THREE.Vector3().fromArray(point.position))), new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: .9 }));
      line.renderOrder = 5;
      this.labPathPreview.add(line);
      for (const point of track.points.filter((value, index) => index === 0 || value.anchor)) {
        const marker = new THREE.Mesh(new THREE.SphereGeometry(point.time === 0 ? .023 : .009, 12, 8),
          new THREE.MeshBasicMaterial({ color, depthTest: false }));
        marker.position.fromArray(point.position); marker.renderOrder = 6;
        this.labPathPreview.add(marker);
      }
      const axes = new THREE.AxesHelper(.09);
      axes.position.fromArray(track.grasp.position);
      axes.quaternion.fromArray(track.grasp.quaternion_xyzw);
      axes.renderOrder = 6;
      this.labPathPreview.add(axes);
    }
    this.dirty = true;
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
    if (this.headCameraStatusTimer !== null) window.clearInterval(this.headCameraStatusTimer);
    this.cloud.dispose();
    this.headCameraOverlay.querySelector('.head-camera-feed')?.removeAttribute('src');
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
