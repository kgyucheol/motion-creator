import * as THREE from 'three';
import { autoSteps, drawnWaypoints, type DrawnPoint, type DrawnStepping, DEFAULT_STEPPING } from './drawn-path';
import { candidateRamenGraspPoints, DEFAULT_RAMEN_GRASP_COEFFICIENTS,
  type GraspPointCoefficients, type RamenGraspCoefficients, type SceneObject } from './scene-objects';

const DRAWINGS_KEY = 'motioncreator-cylinder-lab-v2';
const SETTINGS_KEY = 'motioncreator-cylinder-lab-tool-settings-v1';
export type HandSide = 'left' | 'right';
export type LabPathSource = { center: number[]; radius: number; planeY: number; far?: number[]; up?: number[]; boxAligned?: boolean; grasp?: number[] };
export type LabToolPath = { side: HandSide; toolId: string; projectName: string; points: DrawnPoint[]; flip: boolean;
  source: LabPathSource | null; stepping: DrawnStepping; coefficients: GraspPointCoefficients };
export type ToolPose = { position: number[]; quaternion_xyzw: number[] };
export type ToolPathPoint = ToolPose & { time: number; label: string; anchor: boolean };
export type ToolPathTrack = { side: HandSide; toolId: string; points: ToolPathPoint[]; keyPoints: ToolPathPoint[]; drawnCount: number; grasp: ToolPose };
export type LabLateralOffsets = Partial<Record<HandSide, number>>;
export type LabPathPreview = { targetId: string; tracks: Partial<Record<HandSide, ToolPathTrack>>; times: number[];
  sourcePaths: Partial<Record<HandSide, LabToolPath>>; lateralOffsets: LabLateralOffsets; lateralDirections: Record<HandSide, number> };

/** Shift every TCP sample of a hand along world Y, leaving XZ positions and rotations intact. */
export function withLabLateralOffsets(preview: LabPathPreview, offsets: LabLateralOffsets): LabPathPreview {
  const tracks = { ...preview.tracks };
  for (const side of ['left', 'right'] as const) {
    const track = tracks[side];
    if (!track) continue;
    const previous = preview.lateralOffsets[side] ?? 0;
    const desired = offsets[side] ?? 0;
    const shift = (desired - previous) * preview.lateralDirections[side];
    if (!shift) continue;
    const moved = (pose: ToolPathPoint): ToolPathPoint => ({ ...pose,
      position: [pose.position[0], pose.position[1] + shift, pose.position[2]] });
    const points = track.points.map(moved);
    tracks[side] = { ...track, points, keyPoints: track.keyPoints.map(moved), grasp: points[0] };
  }
  return { ...preview, tracks, lateralOffsets: { ...offsets } };
}

function objectMap(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function parseMap(storage: Pick<Storage, 'getItem'>, key: string) {
  try { return objectMap(JSON.parse(storage.getItem(key) ?? '{}')); } catch { return {}; }
}
function finiteVector(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(v => typeof v === 'number' && Number.isFinite(v));
}
function stepping(value: unknown): DrawnStepping {
  const item = objectMap(value);
  const keys = ['stepM', 'stepDeg', 'speedMS', 'turnDegS', 'minStepS'] as const;
  return keys.every(key => typeof item[key] === 'number' && Number.isFinite(item[key]) && (item[key] as number) > 0)
    ? item as DrawnStepping : DEFAULT_STEPPING;
}

/** Read the two mounted tools' drawings for this exact cylinder ID. An absent drawing leaves that hand idle.
 * Grasp coefficients always come from the editor's shared values, never from per-tool Lab copies. */
export function readLabToolPaths(targetId: string, mounted: [string, string], editorCoefficients: RamenGraspCoefficients,
  storage: Pick<Storage, 'getItem'> = localStorage): Partial<Record<HandSide, LabToolPath>> {
  const saved = parseMap(storage, DRAWINGS_KEY);
  const drawings = objectMap(saved.drawings);
  const sections = parseMap(storage, SETTINGS_KEY);
  const preferredProject = typeof saved.project === 'string' ? saved.project : '';
  const templates = objectMap(saved.pathTemplates), overrides = objectMap(saved.pathOverrides);
  const toolPaths = objectMap(saved.toolPaths);
  const result: Partial<Record<HandSide, LabToolPath>> = {};
  for (const [index, side] of (['left', 'right'] as const).entries()) {
    const toolId = mounted[index];
    const matches = Object.keys(drawings).filter(key => key.startsWith(`${toolId}|`) && key.endsWith(`|${targetId}`));
    matches.sort((a, b) => Number(b === `${toolId}|${preferredProject}|${targetId}`) - Number(a === `${toolId}|${preferredProject}|${targetId}`));
    const legacyKey = `${preferredProject}|${targetId}`;
    const templateId = templates[`${toolId}|${preferredProject}`];
    const sharedKey = typeof templateId === 'string' && templateId !== targetId && overrides[`${toolId}|${preferredProject}|${targetId}`] !== true
      ? `${toolId}|${preferredProject}|${templateId}` : undefined;
    const key = sharedKey && sharedKey in drawings ? sharedKey : matches[0]
      ?? (saved.legacyDrawingTool === toolId && legacyKey in drawings ? legacyKey : undefined);
    const global = objectMap(toolPaths[toolId]);
    const useGlobal = Array.isArray(global.points) && overrides[`${toolId}|${preferredProject}|${targetId}`] !== true;
    const drawing = useGlobal ? global : objectMap(key ? drawings[key] : undefined);
    if ((!key && !useGlobal) || !Array.isArray(drawing.points) || drawing.points.length < 2) continue;
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
    result[side] = { side, toolId, projectName: useGlobal ? '툴 공통 파지법' : key === legacyKey ? preferredProject : key!.slice(toolId.length + 1, key!.lastIndexOf('|')), points,
      flip: drawing.flip === true,
      source: finiteVector(source.center, 2) && typeof source.radius === 'number' && source.radius > 0 &&
        typeof source.planeY === 'number' && Number.isFinite(source.planeY)
        ? { center: source.center, radius: source.radius, planeY: source.planeY,
          ...(finiteVector(source.far, 2) && finiteVector(source.up, 2) ? { far: source.far, up: source.up,
            boxAligned: source.boxAligned === true } : {}),
          ...(finiteVector(source.grasp, 2) ? { grasp: source.grasp } : {}) } : null,
      stepping: stepping(savedStepping), coefficients: editorCoefficients[side],
    };
  }
  return result;
}

export function toolPoseAtKeyPoint(track: ToolPathTrack, index: number): ToolPose {
  return track.keyPoints[Math.min(index, track.keyPoints.length - 1)];
}

/** Convert the Lab's XZ TCP traces into world XYZ positions and full mounted-tool orientations. */
export function buildLabPathPreview(target: SceneObject, robotPosition: number[], robotQuaternion: number[],
  sourcePaths: Partial<Record<HandSide, LabToolPath>>, maxFrames: number): LabPathPreview {
  const sides = ['left', 'right'] as const;
  const active = sides.filter(side => !!sourcePaths[side]);
  if (!active.length) throw new Error('선택한 용기에 장착 툴의 저장 경로가 없습니다. Cylinder Lab에서 경로를 먼저 그려 주세요.');
  if (maxFrames < 1) throw new Error('키프레임 최대 100개에 도달했습니다. 기존 프레임을 줄여 주세요.');
  const configured = Object.fromEntries(sides.map(side => [side, sourcePaths[side]?.coefficients ?? DEFAULT_RAMEN_GRASP_COEFFICIENTS[side]])) as RamenGraspCoefficients;
  const grasp = candidateRamenGraspPoints(target, robotPosition, robotQuaternion, configured);
  const targetRotation = new THREE.Quaternion().fromArray(target.quaternion_xyzw).normalize();
  const targetFar3 = new THREE.Vector3(1, 0, 0).applyQuaternion(targetRotation);
  const targetUp3 = new THREE.Vector3(0, 0, 1).applyQuaternion(targetRotation);
  const targetFar = [targetFar3.x, targetFar3.z], targetUp = [targetUp3.x, targetUp3.z];
  if ((robotPosition[0] - target.position[0]) * targetFar[0] > 0) {
    targetFar[0] *= -1; targetFar[1] *= -1;
  }
  const tracks: Partial<Record<HandSide, ToolPathTrack>> = {};
  for (const side of active) {
    const path = sourcePaths[side]!;
    if (!path.source) throw new Error(`${side === 'left' ? '왼손' : '오른손'} ${path.toolId} 경로에 기준 용기 위치가 없습니다. Cylinder Lab에서 해당 툴의 경로를 다시 열어 저장해 주세요.`);
    const sourceFar = path.source.boxAligned ? path.source.far : undefined;
    const sourceUp = path.source.boxAligned ? path.source.up : undefined;
    const waypoints = drawnWaypoints(path.points, path.stepping);
    if (waypoints.length > 1000) throw new Error(`${path.toolId} 경로의 보간점이 너무 많습니다. Cylinder Lab에서 간격을 늘려 주세요.`);
    const base = new THREE.Quaternion().fromArray(grasp.orientation[side]);
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(base);
    const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(base);
    const baseAngle = Math.atan2(forward.z, forward.x);
    const baseFlip = normal.x * -forward.z + normal.z * forward.x < 0;
    const faceTurn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    const placeOnTarget = (waypoint: { tcp: number[] }) => {
      const dx = waypoint.tcp[0] - path.source!.center[0], dz = waypoint.tcp[1] - path.source!.center[1];
      return sourceFar && sourceUp
        ? [target.position[0] + (dx * sourceFar[0] + dz * sourceFar[1]) * targetFar[0]
            + (dx * sourceUp[0] + dz * sourceUp[1]) * targetUp[0],
          target.position[2] + (dx * sourceFar[0] + dz * sourceFar[1]) * targetFar[1]
            + (dx * sourceUp[0] + dz * sourceUp[1]) * targetUp[1]]
        : [target.position[0] + dx, target.position[2] + dz];
    };
    // The drawing is relative to the grasp point it was drawn against (the first point for older drawings):
    // move the whole path so that reference lands on the editor's grasp point.
    const [firstX, firstZ] = placeOnTarget({ tcp: path.source.grasp ?? waypoints[0].tcp });
    const shift = [grasp[side][0] - firstX, grasp[side][2] - firstZ];
    const points: ToolPathPoint[] = [{ time: 0, position: grasp[side], quaternion_xyzw: grasp.orientation[side],
      label: '파지점', anchor: true }];
    const keyPoints: ToolPathPoint[] = [];
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
      const directionFar = sourceFar && sourceUp ? Math.cos(waypoint.angle) * sourceFar[0] + Math.sin(waypoint.angle) * sourceFar[1] : 0;
      const directionUp = sourceFar && sourceUp ? Math.cos(waypoint.angle) * sourceUp[0] + Math.sin(waypoint.angle) * sourceUp[1] : 0;
      const angle = sourceFar && sourceUp ? Math.atan2(directionFar * targetFar[1] + directionUp * targetUp[1],
        directionFar * targetFar[0] + directionUp * targetUp[0]) : waypoint.angle;
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -(angle - baseAngle));
      const rotation = yaw.multiply(base.clone());
      if (path.flip !== baseFlip) rotation.multiply(faceTurn);
      const [x, z] = placeOnTarget(waypoint);
      const position = [x + shift[0], grasp[side][1], z + shift[1]];
      const time = Number(elapsed.toFixed(6));
      const point = { time, position, quaternion_xyzw: rotation.toArray(),
        label: anchor ? `그린 점 ${controlIndex + 1}` : waypoint.label, anchor };
      if (anchor) keyPoints.push(point);
      if (i === 0 && Math.hypot(position[0] - grasp[side][0], position[2] - grasp[side][2]) < .005
        && Math.abs(angle - baseAngle) < Math.PI / 36 && path.flip === baseFlip) continue;
      points.push(point);
    }
    tracks[side] = { side, toolId: path.toolId, points, keyPoints, drawnCount: path.points.length, grasp: points[0] };
  }
  const count = Math.max(...active.map(side => tracks[side]!.keyPoints.length));
  if (count > maxFrames) throw new Error(`양손 경로의 그린 점 ${count}개가 남은 키프레임 ${maxFrames}개를 넘습니다. Cylinder Lab에서 점을 줄여 주세요.`);
  const times: number[] = [];
  for (let index = 0; index < count; index++) {
    const duration = Math.max(.1, ...active.map(side => {
      const keyPoints = tracks[side]!.keyPoints;
      if (index >= keyPoints.length) return 0;
      return keyPoints[index].time - (index ? keyPoints[index - 1].time : 0);
    }));
    times.push(Number(((times[index - 1] ?? 0) + duration).toFixed(6)));
  }
  const leftDirection = Math.sign(grasp.left[1] - grasp.right[1]) || 1;
  return { targetId: target.id, tracks, times, sourcePaths, lateralOffsets: {},
    lateralDirections: { left: leftDirection, right: -leftDirection } };
}
