import * as THREE from 'three';
import { LengthPrefixedFramer, cloudPlacement } from './point-cloud.ts';

/** 컵 용기의 기본 반지름(m). 줄을 이룬 컵의 눈에 보이는 폭은 약 13cm이다. */
export const DEFAULT_CONTAINER_RADIUS = 0.065;
const STREAM_URL = '/api/head-camera/objects/stream';
const RETRY_MS = 1500;
const CLIPPED_COLOR = 0xffb454;
const ENDPOINT_COLOR = 0x6bffb8;

/** 로봇 인식기가 보내는 한 줄(옆으로 누운 컵들)의 눈에 보이는 능선. 좌표는 컬러 카메라 광학 프레임(x 오른쪽, y 아래, z 앞) 미터. */
export type DetectedRow = {
  label: string; confidence: number;
  start: [number, number, number]; end: [number, number, number];
  normal: [number, number, number]; axis: [number, number, number];
  length: number; width: number; depth_extent: number; points: number; clipped: [boolean, boolean];
};
export type DetectionMessage = { type: 'objects'; frame: 'camera_optical'; stamp: number; inference_ms: number; objects: DetectedRow[] };

const isVector = (value: unknown): value is [number, number, number] =>
  Array.isArray(value) && value.length === 3 && value.every(item => typeof item === 'number' && Number.isFinite(item));

export function parseDetections(message: Uint8Array): DetectionMessage | null {
  try {
    const data = JSON.parse(new TextDecoder().decode(message));
    if (data?.type !== 'objects' || data.frame !== 'camera_optical' || !Array.isArray(data.objects)) return null;
    const objects = data.objects.filter((row: DetectedRow) => isVector(row?.start) && isVector(row?.end) && isVector(row?.normal)
      && Array.isArray(row?.clipped) && Number.isFinite(row?.width));
    return { ...data, objects };
  } catch { return null; }
}

/** 보이는 능선을 카메라 반대쪽(-normal)으로 `radius`만큼 옮긴 용기 중심축의 양 끝점. */
export function axisEndpoints(row: Pick<DetectedRow, 'start' | 'end' | 'normal'>, radius: number) {
  const shift = (point: readonly number[]) => point.map((value, index) => value - radius * row.normal[index]) as [number, number, number];
  return { start: shift(row.start), end: shift(row.end) };
}

/** 두 점 사이에 놓인 Y축 방향 원기둥의 위치와 방향. */
export function segmentPlacement(start: ArrayLike<number>, end: ArrayLike<number>) {
  const from = new THREE.Vector3().fromArray(start as number[]);
  const to = new THREE.Vector3().fromArray(end as number[]);
  const direction = to.clone().sub(from);
  const length = direction.length();
  const quaternion = length > 0 ? new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.divideScalar(length)) : new THREE.Quaternion();
  return { position: from.add(to).multiplyScalar(.5), quaternion, length };
}

/** 로봇이 인식한 용기 줄을 점군과 같은 프레임에 그린다: 중심축 선분, 양끝 점, 반지름을 보여주는 와이어 원통. */
export class DetectedObjectsLayer {
  readonly root = new THREE.Group();
  private radius = DEFAULT_CONTAINER_RADIUS;
  private message: DetectionMessage | null = null;
  private abort: AbortController | null = null;
  private retry: number | null = null;

  constructor(private readonly onChange: () => void, private readonly onStatus: (text: string) => void = () => {}) {
    this.root.visible = false;
    this.root.name = 'detected-container-rows';
  }

  get running() { return this.abort !== null; }
  get containerRadius() { return this.radius; }

  place(headPosition: ArrayLike<number>, viewQuaternion: THREE.Quaternion) {
    const placement = cloudPlacement(headPosition, viewQuaternion);
    this.root.position.copy(placement.position);
    this.root.quaternion.copy(placement.quaternion);
    this.onChange();
  }

  setRadius(radius: number) {
    if (!Number.isFinite(radius) || radius <= 0) return;
    this.radius = radius;
    this.redraw();
  }

  start() {
    if (this.abort) return;
    this.abort = new AbortController();
    this.root.visible = true;
    this.onStatus('연결 중');
    void this.run(this.abort);
  }

  stop() {
    this.abort?.abort();
    this.abort = null;
    if (this.retry !== null) { window.clearTimeout(this.retry); this.retry = null; }
    this.message = null;
    this.root.visible = false;
    this.clear();
    this.onStatus('');
    this.onChange();
  }

  dispose() { this.stop(); }

  private async run(control: AbortController) {
    try {
      const response = await fetch(STREAM_URL, { signal: control.signal });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader();
      const framer = new LengthPrefixedFramer();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const message of framer.push(value)) this.apply(parseDetections(message));
      }
    } catch (error) {
      if (control.signal.aborted) return;
      this.onStatus(`연결 실패 (${error instanceof Error ? error.message : error})`);
    }
    if (control.signal.aborted) return;
    this.abort = null;
    this.retry = window.setTimeout(() => { this.retry = null; this.start(); }, RETRY_MS);
  }

  private apply(message: DetectionMessage | null) {
    if (!message) return;
    this.message = message;
    this.redraw();
  }

  private clear() {
    for (const child of [...this.root.children]) {
      this.root.remove(child);
      child.traverse(object => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose();
        (mesh.material as THREE.Material | undefined)?.dispose();
      });
    }
  }

  private redraw() {
    this.clear();
    const message = this.message;
    if (!message) return;
    for (const row of message.objects) {
      const { start, end } = axisEndpoints(row, this.radius);
      const placement = segmentPlacement(start, end);
      const body = new THREE.Mesh(new THREE.CylinderGeometry(this.radius, this.radius, placement.length, 24, 1, true),
        new THREE.MeshBasicMaterial({ color: ENDPOINT_COLOR, wireframe: true, transparent: true, opacity: .25, depthWrite: false }));
      const axis = new THREE.Mesh(new THREE.CylinderGeometry(.004, .004, placement.length, 8),
        new THREE.MeshBasicMaterial({ color: ENDPOINT_COLOR, depthTest: false }));
      for (const mesh of [body, axis]) {
        mesh.position.copy(placement.position);
        mesh.quaternion.copy(placement.quaternion);
        mesh.renderOrder = 10;
        this.root.add(mesh);
      }
      row.clipped.forEach((clipped, index) => {
        const ball = new THREE.Mesh(new THREE.SphereGeometry(clipped ? .01 : .014, 16, 12),
          new THREE.MeshBasicMaterial({ color: clipped ? CLIPPED_COLOR : ENDPOINT_COLOR, depthTest: false, wireframe: clipped }));
        ball.position.fromArray(index === 0 ? start : end);
        ball.renderOrder = 11;
        this.root.add(ball);
      });
    }
    this.onStatus(`${message.objects.length}줄 · 추론 ${Math.round(message.inference_ms)}ms`);
    this.onChange();
  }
}
