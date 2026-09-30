// A hand-drawn TCP path: points the user places in the cross-section, cut into small steps.
import type { Waypoint } from './cylinder-sim';

/** `angle` null follows the direction of travel; `steps` null splits the segment into this point automatically. */
export type DrawnPoint = { x: number; z: number; angle: number | null; steps: number | null };
export type DrawnStepping = { stepM: number; stepDeg: number; speedMS: number; turnDegS: number; minStepS: number };
export const DEFAULT_STEPPING: DrawnStepping = { stepM: .02, stepDeg: 10, speedMS: .06, turnDegS: 40, minStepS: .15 };

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** Tool angle at each point: the user's angle, or the travel direction (into the point; out of the first). */
export function pointAngles(points: DrawnPoint[]) {
  return points.map((point, i) => {
    if (point.angle !== null) return point.angle;
    const [a, b] = i === 0 ? [points[0], points[1]] : [points[i - 1], point];
    if (!a || !b || Math.hypot(b.x - a.x, b.z - a.z) < 1e-9) return 0;
    return Math.atan2(b.z - a.z, b.x - a.x);
  });
}

/** Automatic step count of the segment ending at point `i`. */
export function autoSteps(points: DrawnPoint[], i: number, stepping: DrawnStepping = DEFAULT_STEPPING) {
  const angles = pointAngles(points);
  const distance = Math.hypot(points[i].x - points[i - 1].x, points[i].z - points[i - 1].z);
  const turn = Math.abs(wrap(angles[i] - angles[i - 1])) * 180 / Math.PI;
  return Math.max(1, Math.ceil(distance / stepping.stepM - 1e-9), Math.ceil(turn / stepping.stepDeg - 1e-9));
}

/** Waypoints for the simulator (and later the editor): one per step, the drawn points included. */
export function drawnWaypoints(points: DrawnPoint[], stepping: DrawnStepping = DEFAULT_STEPPING): Waypoint[] {
  if (!points.length) return [];
  const angles = pointAngles(points);
  const waypoints: Waypoint[] = [{ stage: 'draw', label: '점 1', tcp: [points[0].x, points[0].z], angle: angles[0], duration: .1 }];
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    const turn = wrap(angles[i] - angles[i - 1]);
    const steps = Math.max(1, Math.round(b.steps ?? autoSteps(points, i, stepping)));
    const distance = Math.hypot(b.x - a.x, b.z - a.z);
    const seconds = Math.max(stepping.minStepS, distance / steps / stepping.speedMS, Math.abs(turn) * 180 / Math.PI / steps / stepping.turnDegS);
    for (let step = 1; step <= steps; step++) {
      const f = step / steps;
      waypoints.push({ stage: 'draw', label: steps === 1 ? `점 ${i} → ${i + 1}` : `점 ${i} → ${i + 1} · ${step}/${steps}`,
        tcp: [a.x + (b.x - a.x) * f, a.z + (b.z - a.z) * f], angle: angles[i - 1] + turn * f, duration: seconds });
    }
  }
  return waypoints;
}

/** Snap a point onto a circle (e.g. the target surface minus the wedge) when it is within `tolerance`. */
export function snapToCircle(x: number, z: number, center: number[], radius: number, tolerance: number) {
  const dx = x - center[0], dz = z - center[1];
  const distance = Math.hypot(dx, dz);
  if (distance < 1e-9 || Math.abs(distance - radius) > tolerance) return null;
  return { x: center[0] + dx / distance * radius, z: center[1] + dz / distance * radius };
}
