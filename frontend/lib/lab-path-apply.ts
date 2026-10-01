import * as THREE from 'three';
import { autoSteps, drawnWaypoints, type DrawnPoint, type DrawnStepping, DEFAULT_STEPPING } from './drawn-path';
import { candidateRamenGraspPoints, DEFAULT_RAMEN_GRASP_COEFFICIENTS,
  type GraspPointCoefficients, type RamenGraspCoefficients, type SceneObject } from './scene-objects';

const DRAWINGS_KEY = 'motioncreator-cylinder-lab-v2';
const SETTINGS_KEY = 'motioncreator-cylinder-lab-tool-settings-v1';
const GRASPS_KEY = 'motioncreator-cylinder-lab-tool-grasps-v1';
export type HandSide = 'left' | 'right';
export type LabPathSource = { center: number[]; radius: number; planeY: number };
export type LabToolPath = { side: HandSide; toolId: string; projectName: string; points: DrawnPoint[]; flip: boolean;
  source: LabPathSource | null; stepping: DrawnStepping; coefficients: GraspPointCoefficients };
export type ToolPose = { position: number[]; quaternion_xyzw: number[] };
export type ToolPathPoint = ToolPose & { time: number; label: string; anchor: boolean };
export type ToolPathTrack = { side: HandSide; toolId: string; points: ToolPathPoint[]; drawnCount: number; grasp: ToolPose };
export type LabPathPreview = { targetId: string; tracks: Partial<Record<HandSide, ToolPathTrack>>; times: number[];
  sourcePaths: Partial<Record<HandSide, LabToolPath>> };

function objectMap(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function parseMap(storage: Pick<Storage, 'getItem'>, key: string) {
  try { return objectMap(JSON.parse(storage.getItem(key) ?? '{}')); } catch { return {}; }
}
function finiteVector(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(v => typeof v === 'number' && Number.isFinite(v));
}
function coefficients(value: unknown, fallback: GraspPointCoefficients) {
  const item = objectMap(value);
  return (['lateral', 'clearance', 'height', 'roll', 'pitch', 'yaw'] as const).every(field =>
    typeof item[field] === 'number' && Number.isFinite(item[field]) &&
    (field !== 'lateral' && field !== 'clearance' || (item[field] as number) >= 0))
    ? item as GraspPointCoefficients : fallback;
}
function stepping(value: unknown): DrawnStepping {
  const item = objectMap(value);
  const keys = ['stepM', 'stepDeg', 'speedMS', 'turnDegS', 'minStepS'] as const;
  return keys.every(key => typeof item[key] === 'number' && Number.isFinite(item[key]) && (item[key] as number) > 0)
    ? item as DrawnStepping : DEFAULT_STEPPING;
}

/** Read the two mounted tools' drawings for this exact cylinder ID. An absent drawing leaves that hand idle. */
export function readLabToolPaths(targetId: string, mounted: [string, string], editorCoefficients: RamenGraspCoefficients,
  storage: Pick<Storage, 'getItem'> = localStorage): Partial<Record<HandSide, LabToolPath>> {
  const saved = parseMap(storage, DRAWINGS_KEY);
  const drawings = objectMap(saved.drawings);
  const sections = parseMap(storage, SETTINGS_KEY);
  const grasps = parseMap(storage, GRASPS_KEY);
  const preferredProject = typeof saved.project === 'string' ? saved.project : '';
  const result: Partial<Record<HandSide, LabToolPath>> = {};
  for (const [index, side] of (['left', 'right'] as const).entries()) {
    const toolId = mounted[index];
    const matches = Object.keys(drawings).filter(key => key.startsWith(`${toolId}|`) && key.endsWith(`|${targetId}`));
    matches.sort((a, b) => Number(b === `${toolId}|${preferredProject}|${targetId}`) - Number(a === `${toolId}|${preferredProject}|${targetId}`));
    const legacyKey = `${preferredProject}|${targetId}`;
    const key = matches[0] ?? (saved.legacyDrawingTool === toolId && legacyKey in drawings ? legacyKey : undefined);
    const drawing = objectMap(key ? drawings[key] : undefined);
    if (!key || !Array.isArray(drawing.points) || drawing.points.length < 2) continue;
    const points = drawing.points.map(value => {
      const item = objectMap(value);
      return { x: Number(item.x), z: Number(item.z),
        angle: item.angle === null ? null : Number(item.angle), steps: item.steps === null ? null : Number(item.steps) };
    });
    if (points.length > 200 || points.some(point => !Number.isFinite(point.x) || !Number.isFinite(point.z)
      || point.angle !== null && !Number.isFinite(point.angle) || point.steps !== null && (!Number.isInteger(point.steps) || point.steps < 1 || point.steps > 100))) continue;
    const source = objectMap(drawing.source);
    const section = objectMap(objectMap(sections[toolId]).stepping);
    const savedStepping = { stepM: Number(section['step-m']), stepDeg: Number(section['step-deg']),
      speedMS: Number(section.speed), turnDegS: Number(section.turn), minStepS: Number(section['min-step']) };
    result[side] = { side, toolId, projectName: key === legacyKey ? preferredProject : key.slice(toolId.length + 1, -targetId.length - 1), points,
      flip: drawing.flip === true,
      source: finiteVector(source.center, 2) && typeof source.radius === 'number' && source.radius > 0 &&
        typeof source.planeY === 'number' && Number.isFinite(source.planeY)
        ? { center: source.center, radius: source.radius, planeY: source.planeY } : null,
      stepping: stepping(savedStepping), coefficients: coefficients(grasps[toolId], editorCoefficients[side]),
    };
  }
  return result;
}

function poseAt(track: ToolPathTrack, time: number): ToolPose {
  const points = track.points;
  if (time <= 0) return points[0];
  const next = points.findIndex(point => point.time >= time - 1e-8);
  if (next <= 0) return next < 0 ? points[points.length - 1] : points[0];
  const a = points[next - 1], b = points[next];
  const f = Math.max(0, Math.min(1, (time - a.time) / (b.time - a.time)));
  return { position: a.position.map((v, i) => v + (b.position[i] - v) * f),
    quaternion_xyzw: new THREE.Quaternion().fromArray(a.quaternion_xyzw)
      .slerp(new THREE.Quaternion().fromArray(b.quaternion_xyzw), f).toArray() };
}
export const toolPoseAt = poseAt;

/** Convert the Lab's XZ TCP traces into world XYZ positions and full mounted-tool orientations. */
export function buildLabPathPreview(target: SceneObject, robotPosition: number[], robotQuaternion: number[],
  sourcePaths: Partial<Record<HandSide, LabToolPath>>, maxFrames: number): LabPathPreview {
  const sides = ['left', 'right'] as const;
  const active = sides.filter(side => !!sourcePaths[side]);
  if (!active.length) throw new Error('선택한 용기에 장착 툴의 저장 경로가 없습니다. Cylinder Lab에서 경로를 먼저 그려 주세요.');
  if (maxFrames < 1) throw new Error('키프레임 최대 100개에 도달했습니다. 기존 프레임을 줄여 주세요.');
  const configured = Object.fromEntries(sides.map(side => [side, sourcePaths[side]?.coefficients ?? DEFAULT_RAMEN_GRASP_COEFFICIENTS[side]])) as RamenGraspCoefficients;
  const grasp = candidateRamenGraspPoints(target, robotPosition, robotQuaternion, configured);
  const tracks: Partial<Record<HandSide, ToolPathTrack>> = {};
  const candidates = new Set<number>();
  for (const side of active) {
    const path = sourcePaths[side]!;
    if (!path.source) throw new Error(`${side === 'left' ? '왼손' : '오른손'} ${path.toolId} 경로에 기준 용기 위치가 없습니다. Cylinder Lab에서 해당 툴의 경로를 다시 열어 저장해 주세요.`);
    const waypoints = drawnWaypoints(path.points, path.stepping);
    if (waypoints.length > 1000) throw new Error(`${path.toolId} 경로의 보간점이 너무 많습니다. Cylinder Lab에서 간격을 늘려 주세요.`);
    const base = new THREE.Quaternion().fromArray(grasp.orientation[side]);
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(base);
    const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(base);
    const baseAngle = Math.atan2(forward.z, forward.x);
    const baseFlip = normal.x * -forward.z + normal.z * forward.x < 0;
    const faceTurn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    const points: ToolPathPoint[] = [{ time: 0, position: grasp[side], quaternion_xyzw: grasp.orientation[side],
      label: '파지점', anchor: true }];
    let elapsed = 0, controlIndex = 0, controlEnd = 0;
    for (let i = 0; i < waypoints.length; i++) {
      const waypoint = waypoints[i];
      elapsed += Math.max(.1, waypoint.duration);
      if (i === 0) controlEnd = 0;
      else if (i > controlEnd) {
        controlIndex++;
        controlEnd += Math.max(1, Math.round(path.points[controlIndex].steps ?? autoSteps(path.points, controlIndex, path.stepping)));
      }
      const anchor = i === controlEnd;
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -(waypoint.angle - baseAngle));
      const rotation = yaw.multiply(base.clone());
      if (path.flip !== baseFlip) rotation.multiply(faceTurn);
      const position = [target.position[0] + waypoint.tcp[0] - path.source.center[0], grasp[side][1],
        target.position[2] + waypoint.tcp[1] - path.source.center[1]];
      if (i === 0 && Math.hypot(position[0] - grasp[side][0], position[2] - grasp[side][2]) < .005
        && Math.abs(waypoint.angle - baseAngle) < Math.PI / 36 && path.flip === baseFlip) continue;
      const time = Number(elapsed.toFixed(6));
      points.push({ time, position, quaternion_xyzw: rotation.toArray(), label: anchor ? `그린 점 ${controlIndex + 1}` : waypoint.label, anchor });
      candidates.add(time);
    }
    tracks[side] = { side, toolId: path.toolId, points, drawnCount: path.points.length, grasp: points[0] };
  }
  const times = [0, ...candidates].sort((a, b) => a - b);
  if (times.length > maxFrames) throw new Error(`양손 경로의 웨이포인트 ${times.length}개가 남은 키프레임 ${maxFrames}개를 넘습니다. Cylinder Lab에서 보간 간격을 늘려 주세요.`);
  return { targetId: target.id, tracks, times, sourcePaths };
}
