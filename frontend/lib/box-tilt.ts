// Change the tilt of the open box in the cylinder section: the box and everything inside turn together.
import type { Scene } from './cylinder-sim.ts';

type TiltBox = { id?: string; center: number[]; angle: number; width: number; height: number; thickness: number; e?: number[]; u?: number[] };
type TiltScene = Scene & { box?: TiltBox | null; toward_robot?: number[] };

/** The box tilt as the 3-D editor shows it: pitch (rotation about +Y), which is minus the section angle. */
export const boxPitchDeg = (scene: TiltScene) => (scene.box ? -scene.box.angle * 180 / Math.PI : null);

/**
 * Rotate the box, its walls and the objects inside it about the box centre so the box pitch becomes
 * `pitchDeg`. The target frame turns with the box (its rim height is unchanged); `far` keeps pointing
 * away from the robot. Returns the new scene and the rotation applied (radians, +X toward +Z).
 */
export function tiltScene<T extends TiltScene>(scene: T, pitchDeg: number): { scene: T; delta: number; pivot: number[] } {
  const box = scene.box;
  if (!box) throw new Error('이 장면에는 기울일 상자가 없습니다.');
  const angle = -pitchDeg * Math.PI / 180, delta = angle - box.angle, [cx, cz] = box.center;
  const c = Math.cos(delta), s = Math.sin(delta);
  const turn = (p: number[]) => [cx + (p[0] - cx) * c - (p[1] - cz) * s, cz + (p[0] - cx) * s + (p[1] - cz) * c];
  const inside = (p: number[], margin: number) => {
    const dx = p[0] - cx, dz = p[1] - cz, bc = Math.cos(box.angle), bs = Math.sin(box.angle);
    return Math.abs(dx * bc + dz * bs) <= box.width / 2 + margin && Math.abs(-dx * bs + dz * bc) <= box.height / 2 + margin;
  };
  const boxId = box.id;
  const walls = scene.walls.map(w => (boxId ? w.id === boxId : inside(w.center, 0)) ? { ...w, center: turn(w.center), angle: w.angle + delta } : w);
  const circles = scene.circles.map(o => inside(o.center, o.radius) ? { ...o, center: turn(o.center) } : o);
  const rectangles = scene.rectangles.map(r => inside(r.center, 0) ? { ...r, center: turn(r.center), angle: r.angle + delta } : r);
  const up = [-Math.sin(angle), Math.cos(angle)];
  let far = [Math.cos(angle), Math.sin(angle)];
  const toward = scene.toward_robot ?? [-1, 0];
  if (far[0] * toward[0] + far[1] * toward[1] > 0) far = [-far[0], -far[1]];
  const target = circles.find(o => o.id === scene.target_id) ?? scene.target;
  return {
    scene: { ...scene, walls, circles, rectangles, target,
      box: { ...box, angle, e: [Math.cos(angle), Math.sin(angle)], u: up },
      frame: { ...scene.frame, origin: [...target.center], far, up } },
    delta, pivot: [cx, cz],
  };
}
