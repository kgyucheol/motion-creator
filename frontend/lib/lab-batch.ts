// Take cylinders out of the box one after another with one path: carry the path over to each chosen
// cylinder and simulate, starting each attempt from where the previous one left the pile.
import { restingScene, simulateScoop, type BladePath, type Scene, type SimOptions, type SimResult, type BladeProfile } from './cylinder-sim.ts';

export type Numbered = { id: string; name: string; order: number; layer: number };

/**
 * Number the cylinders the way the robot takes them: the layer nearest the box opening first (the
 * frame's `up`), and inside a layer the higher one first (toward the upper wall, the frame's `far`).
 */
export function numberCylinders(scene: Scene): Numbered[] {
  const [ux, uz] = scene.frame.up, [fx, fz] = scene.frame.far;
  const items = scene.circles.map(c => ({ id: c.id, name: c.name, up: c.center[0] * ux + c.center[1] * uz,
    far: c.center[0] * fx + c.center[1] * fz, radius: c.radius }));
  items.sort((a, b) => b.up - a.up);
  const layers: number[] = [];
  let layer = -1, layerUp = Infinity;
  for (const item of items) {
    if (layerUp - item.up > item.radius * .5) { layer++; layerUp = item.up; }
    layers.push(layer);
  }
  return items.map((item, index) => ({ ...item, layer: layers[index] }))
    .sort((a, b) => a.layer - b.layer || b.far - a.far)
    .map((item, index) => ({ id: item.id, name: item.name, order: index + 1, layer: item.layer + 1 }));
}

/** The same scene aimed at another cylinder: frame moved to it, rim height kept relative to the box. */
export function retarget<T extends Scene>(scene: T, targetId: string): T {
  const target = scene.circles.find(c => c.id === targetId);
  if (!target) throw new Error(`Unknown cylinder ${targetId}`);
  const [ox, oz] = scene.frame.origin, [nx, nz] = target.center, [ux, uz] = scene.frame.up;
  return { ...scene, target_id: targetId, target,
    frame: { ...scene.frame, origin: [nx, nz], rim_up_m: scene.frame.rim_up_m - ((nx - ox) * ux + (nz - oz) * uz) } };
}

/**
 * Carry blade paths drawn for one cylinder over to another. The box frame is shared, so a path
 * relative to the cylinder centre (and therefore to its grasp point) only needs a translation.
 */
export function translatePaths(paths: BladePath[], from: number[], to: number[]): BladePath[] {
  const dx = to[0] - from[0], dz = to[1] - from[1];
  return paths.map(path => ({ ...path, waypoints: path.waypoints.map(w => ({ ...w, tcp: [w.tcp[0] + dx, w.tcp[1] + dz] })) }));
}

export type Attempt = { order: number; id: string; name: string; scene: Scene; paths: BladePath[]; result: SimResult };
export type BatchReport = { attempts: Attempt[]; extracted: number; total: number; stoppedAt: Numbered | null };

/**
 * One extraction: aim the pile at `targetId`, carry the path over from the cylinder it was made for
 * (at `referenceCenter`) and simulate. Pass `settle` when the pile is not at rest yet (after a run).
 */
export function runAttempt(pile: Scene, targetId: string, paths: BladePath[], referenceCenter: number[], order: number,
  options: Partial<SimOptions> = {}, settle = false, profile?: BladeProfile): Attempt {
  const aimed = retarget(pile, targetId);
  const scene = settle ? restingScene(aimed) : aimed;
  const moved = translatePaths(paths, referenceCenter, scene.target.center);
  const result = simulateScoop(scene, moved, profile, options);
  return { order, id: targetId, name: scene.circles.find(c => c.id === targetId)?.name ?? targetId, scene, paths: moved, result };
}

/** The pile after a successful attempt: the target gone, the rest where the simulation left them. */
export function pileAfter(attempt: Attempt): Scene | null {
  const { scene, result } = attempt;
  const last = result.frames[result.frames.length - 1].bodies;
  const circles = scene.circles.filter(c => c.id !== attempt.id).map(c => ({ ...c, center: last[c.id] ? [last[c.id].x, last[c.id].z] : c.center }));
  if (!circles.length) return null;
  const rectangles = scene.rectangles.map(r => ({ ...r, center: last[r.id] ? [last[r.id].x, last[r.id].z] : r.center, angle: last[r.id]?.angle ?? r.angle }));
  return { ...scene, circles, rectangles, target: circles[0], target_id: circles[0].id };
}

/**
 * Take the cylinders out in the numbered order with one path; `start` must be at rest
 * (restingScene). A failure leaves the cylinder count unchanged, so the run stops there.
 */
export function runBatch(start: Scene, paths: BladePath[], referenceId: string, options: Partial<SimOptions> = {},
  profile?: BladeProfile, onAttempt: (attempt: Attempt, index: number, total: number) => void = () => {}): BatchReport {
  const reference = start.circles.find(c => c.id === referenceId);
  if (!reference) throw new Error('The path reference cylinder is not in the scene');
  const plan = numberCylinders(start);
  let pile: Scene | null = start;
  const attempts: Attempt[] = [];
  for (const item of plan) {
    if (!pile) break;
    // `start` is already at rest; later piles are where the last run left them, so settle those again.
    const attempt = runAttempt(pile, item.id, paths, reference.center, item.order, options, attempts.length > 0, profile);
    attempts.push(attempt);
    onAttempt(attempt, attempts.length - 1, plan.length);
    if (!attempt.result.success) return { attempts, extracted: attempts.length - 1, total: plan.length, stoppedAt: item };
    pile = pileAfter(attempt);
  }
  return { attempts, extracted: attempts.length, total: plan.length, stoppedAt: null };
}
