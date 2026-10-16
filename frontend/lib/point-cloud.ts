import * as THREE from 'three';

/**
 * 점군 기준점의 d435_link 원점 기준 오프셋. link 프레임(x 앞, y 왼쪽, z 위)의 미터 값이다.
 * PiP 비교 화면에서 실제 영상과 겹쳐 맞춘 값(앞 2mm, 왼쪽 -5mm, 위 -16mm, 회전 0)이 기본값이다.
 * 이 값은 물리적인 렌즈 위치가 아니라, 가상 카메라가 컬러 영상의 광학 중심(cx 428.9, cy 256.1)이 아닌 화면 중앙을 쓰는
 * 차이(0.47m 거리에서 약 -4mm, -12mm)를 대부분 보상한다. 그래서 URDF의 d435_joint에는 반영하지 않는다.
 */
export const HEAD_CAMERA_LENS_OFFSET: [number, number, number] = [0.002, -0.005, -0.016];

const HEADER_BYTES = 12;
const MAX_POINTS = 150_000;
const MAX_MESSAGE_BYTES = 32 * 1024 * 1024;
const RETRY_MS = 1500;
const STREAM_URL = '/api/head-camera/cloud/stream';
/** RealSense 광학 프레임(x 오른쪽, y 아래, z 앞)을 three 카메라 프레임(x 오른쪽, y 위, z 뒤)으로: x축 기준 180° 회전. */
const OPTICAL_TO_VIEW = new THREE.Quaternion(1, 0, 0, 0);

export type PointCloud = { count: number; positions: Float32Array; colors: Uint8Array };

/** 로봇 송출 메시지: b'PC01' | uint32 n | uint16 stride | uint16 reserved | int16[n,3] xyz(mm) | uint8[n,3] rgb (little endian) */
export function parsePointCloud(message: Uint8Array): PointCloud | null {
  if (message.length < HEADER_BYTES) return null;
  const view = new DataView(message.buffer, message.byteOffset, message.byteLength);
  if (view.getUint32(0, false) !== 0x50433031) return null;
  const count = view.getUint32(4, true);
  if (message.length !== HEADER_BYTES + count * 9) return null;
  const millimeters = new Int16Array(message.slice(HEADER_BYTES, HEADER_BYTES + count * 6).buffer);
  const positions = new Float32Array(count * 3);
  for (let index = 0; index < positions.length; index++) positions[index] = millimeters[index] / 1000;
  return { count, positions, colors: message.slice(HEADER_BYTES + count * 6) };
}

/** `uint32(LE) 길이 + 본문` 으로 이어진 바이트 스트림에서 완성된 메시지를 꺼낸다. */
export class LengthPrefixedFramer {
  private buffer = new Uint8Array(0);

  push(chunk: Uint8Array): Uint8Array[] {
    const joined = new Uint8Array(this.buffer.length + chunk.length);
    joined.set(this.buffer);
    joined.set(chunk, this.buffer.length);
    const messages: Uint8Array[] = [];
    let offset = 0;
    while (joined.length - offset >= 4) {
      const length = new DataView(joined.buffer, joined.byteOffset + offset, 4).getUint32(0, true);
      if (length > MAX_MESSAGE_BYTES) throw new Error(`포인트 클라우드 메시지가 너무 큽니다: ${length}바이트`);
      if (joined.length - offset - 4 < length) break;
      messages.push(joined.slice(offset + 4, offset + 4 + length));
      offset += 4 + length;
    }
    this.buffer = joined.slice(offset);
    return messages;
  }
}

/** 점군(광학 프레임 좌표)을 놓을 월드 자세. `viewQuaternion`은 헤드 카메라의 three 카메라 방향이다. */
export function cloudPlacement(headPosition: ArrayLike<number>, viewQuaternion: THREE.Quaternion, lensOffset: readonly number[] = [0, 0, 0]) {
  const [forward, left, up] = lensOffset;
  const offset = new THREE.Vector3(-left, up, -forward).applyQuaternion(viewQuaternion);
  return {
    position: new THREE.Vector3().fromArray(headPosition as number[]).add(offset),
    quaternion: viewQuaternion.clone().multiply(OPTICAL_TO_VIEW),
  };
}

/** 로봇에서 오는 포인트 클라우드 스트림을 THREE.Points 하나로 그린다. */
export class PointCloudLayer {
  readonly points: THREE.Points;
  private readonly positionAttribute: THREE.BufferAttribute;
  private readonly colorAttribute: THREE.BufferAttribute;
  private abort: AbortController | null = null;
  private retry: number | null = null;
  private arrivals: number[] = [];

  constructor(private readonly onChange: () => void, private readonly onStatus: (text: string) => void = () => {}) {
    const geometry = new THREE.BufferGeometry();
    this.positionAttribute = new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3);
    this.colorAttribute = new THREE.BufferAttribute(new Uint8Array(MAX_POINTS * 3), 3, true);
    geometry.setAttribute('position', this.positionAttribute);
    geometry.setAttribute('color', this.colorAttribute);
    geometry.setDrawRange(0, 0);
    this.points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: .006, vertexColors: true, sizeAttenuation: true }));
    this.points.frustumCulled = false;
    this.points.visible = false;
    this.points.name = 'head-camera-point-cloud';
  }

  get running() { return this.abort !== null; }

  place(headPosition: ArrayLike<number>, viewQuaternion: THREE.Quaternion) {
    const placement = cloudPlacement(headPosition, viewQuaternion);
    this.points.position.copy(placement.position);
    this.points.quaternion.copy(placement.quaternion);
    this.onChange();
  }

  start() {
    if (this.abort) return;
    this.abort = new AbortController();
    this.points.visible = true;
    this.onStatus('연결 중');
    void this.run(this.abort);
  }

  stop() {
    this.abort?.abort();
    this.abort = null;
    if (this.retry !== null) { window.clearTimeout(this.retry); this.retry = null; }
    this.points.visible = false;
    this.points.geometry.setDrawRange(0, 0);
    this.arrivals = [];
    this.onStatus('');
    this.onChange();
  }

  dispose() {
    this.stop();
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
  }

  private async run(control: AbortController) {
    try {
      const response = await fetch(STREAM_URL, { signal: control.signal });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader();
      const framer = new LengthPrefixedFramer();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const message of framer.push(value)) this.apply(parsePointCloud(message));
      }
    } catch (error) {
      if (control.signal.aborted) return;
      this.onStatus(`연결 실패 (${error instanceof Error ? error.message : error})`);
    }
    if (control.signal.aborted) return;
    this.abort = null;
    this.retry = window.setTimeout(() => { this.retry = null; this.start(); }, RETRY_MS);
  }

  private apply(cloud: PointCloud | null) {
    if (!cloud) return;
    const count = Math.min(cloud.count, MAX_POINTS);
    this.positionAttribute.array.set(cloud.positions.subarray(0, count * 3));
    this.colorAttribute.array.set(cloud.colors.subarray(0, count * 3));
    this.positionAttribute.needsUpdate = true;
    this.colorAttribute.needsUpdate = true;
    this.points.geometry.setDrawRange(0, count);
    const now = performance.now();
    this.arrivals = this.arrivals.filter(time => now - time < 2000).concat(now);
    const span = (now - this.arrivals[0]) / 1000;
    this.onStatus(`${count.toLocaleString('ko-KR')}점${this.arrivals.length > 1 && span > 0 ? ` · ${((this.arrivals.length - 1) / span).toFixed(1)}fps` : ''}`);
    this.onChange();
  }
}
