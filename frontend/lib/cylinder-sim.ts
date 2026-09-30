// 2-D rigid-body check of a scoop path in the cylinder's cross-section (Planck.js / Box2D).
// Runs unchanged in the browser lab page and in the Node sweep script.
import { World, Vec2, Circle, Box, Polygon, Settings, MotorJoint, type Body } from 'planck';

// Box2D is tuned for objects around 0.1–10 units; working in decimetres keeps a 13.5 cm cylinder
// in that range and makes the solver's 5 mm default slop 0.5 mm, small enough for a few-mm wedge.
const SCALE = 10;
Settings.linearSlop = .005;

export type Waypoint = { stage: string; label: string; tcp: number[]; angle: number; duration: number };
export type Scene = {
  target_id: string;
  target: { id: string; center: number[]; radius: number };
  circles: { id: string; name: string; center: number[]; radius: number; mass_kg: number; friction: number }[];
  rectangles: { id: string; center: number[]; half: number[]; angle: number; mass_kg: number; friction: number }[];
  walls: { id: string; center: number[]; half: number[]; angle: number; friction: number }[];
  frame: { origin: number[]; far: number[]; up: number[]; rim_up_m: number };
};
/** Tool section in its own frame: x forward, y = top-face normal, TCP at the origin on the top face. */
export type BladeProfile = { tipAhead: number; length: number; thickness: number; heelLength: number; heelHeight: number; friction: number };
// NSE scoop: blade 0.025–0.32 m, neck 0.018–0.065 m, TCP at the tip.
export const NSE_SCOOP_PROFILE: BladeProfile = { tipAhead: 0, length: .295, thickness: .006, heelLength: .047, heelHeight: .02, friction: .5 };
/** The robot moves the blade through a force-limited pose tracker; a jammed blade falls behind its path. */
export type SimOptions = { settleSeconds: number; holdSeconds: number; dt: number; forceLimitN: number; torqueLimitNm: number;
  trackingLimitM: number; trackingLimitDeg: number; neighborLimitM: number; bladeMassKg: number; frameEvery: number };
export const DEFAULT_SIM: SimOptions = { settleSeconds: 1, holdSeconds: 1.5, dt: 1 / 240, forceLimitN: 60, torqueLimitNm: 8,
  trackingLimitM: .01, trackingLimitDeg: 5, neighborLimitM: .02, bladeMassKg: .3, frameEvery: 8 };

export type Pose = { x: number; z: number; angle: number };
export type Frame = { t: number; stage: string; blade: Pose; command: Pose; blades: Pose[]; commands: Pose[]; bodies: Record<string, Pose>; forceN: number };
/** Stages where the blade is meant to press on the pile; stopping short there is the wedge seating, not a failure. */
export const CONTACT_STAGES = new Set(['insert', 'squeeze', 'pry', 'wrap', 'slide', 'tilt']);
export type SimResult = {
  success: boolean; reason: string; steps: number; seconds: number;
  liftM: number; requiredLiftM: number; targetOnBlade: boolean; targetTouchesOthers: boolean;
  neighborMaxM: number; peakForceN: number; trackingMaxM: number; trackingMaxDeg: number; jamStage: string | null; stalledM: number; finalErrorM: number; frames: Frame[];
};

const ease = (u: number) => u * u * u * (10 + u * (-15 + 6 * u)); // quintic, like the editor's keyframes
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** TCP pose at time `t` after the path starts; each waypoint's duration is its travel time. */
export function poseAt(waypoints: Waypoint[], t: number): Pose & { stage: string } {
  let elapsed = 0;
  for (let i = 1; i < waypoints.length; i++) {
    const a = waypoints[i - 1], b = waypoints[i];
    if (t <= elapsed + b.duration) {
      const s = ease(Math.max(0, Math.min(1, (t - elapsed) / b.duration)));
      return { x: a.tcp[0] + (b.tcp[0] - a.tcp[0]) * s, z: a.tcp[1] + (b.tcp[1] - a.tcp[1]) * s,
        angle: a.angle + wrap(b.angle - a.angle) * s, stage: b.stage };
    }
    elapsed += b.duration;
  }
  const last = waypoints[waypoints.length - 1];
  return { x: last.tcp[0], z: last.tcp[1], angle: last.angle, stage: last.stage };
}

export const pathSeconds = (waypoints: Waypoint[]) => waypoints.slice(1).reduce((sum, w) => sum + w.duration, 0);

function bladeFixtures(body: Body, profile: BladeProfile, flip: boolean, density: number) {
  const s = SCALE;
  const y = (value: number) => (flip ? -value : value) * s; // a rolled-over tool has its top face on the other side
  const back = profile.tipAhead - profile.length;
  const quad = (x0: number, y0: number, x1: number, y1: number) => {
    const points = [Vec2(x0 * s, y(y0)), Vec2(x1 * s, y(y0)), Vec2(x1 * s, y(y1)), Vec2(x0 * s, y(y1))];
    return new Polygon(flip ? points.reverse() : points);
  };
  body.createFixture(quad(back, -profile.thickness, profile.tipAhead, 0), { friction: profile.friction, density });
  if (profile.heelLength > 0 && profile.heelHeight > 0) {
    body.createFixture(quad(back - profile.heelLength, -profile.thickness, back, profile.heelHeight), { friction: profile.friction, density });
  }
}

export type BladePath = { waypoints: Waypoint[]; flip: boolean; side?: string };

/**
 * Simulate one or more blades following their paths. Several blades stand for tools at different
 * points along the cylinder axis: they never touch each other and share the robot's force limit.
 */
export function simulateScoop(scene: Scene, paths: BladePath[] | Waypoint[], profile: BladeProfile = NSE_SCOOP_PROFILE,
  options: Partial<SimOptions> = {}, flip = false): SimResult {
  const opt = { ...DEFAULT_SIM, ...options };
  const plans: BladePath[] = paths.length && 'waypoints' in paths[0] ? paths as BladePath[] : [{ waypoints: paths as Waypoint[], flip }];
  const s = SCALE;
  const world = new World({ gravity: Vec2(0, -9.81 * s) });
  for (const wall of scene.walls) {
    world.createBody().createFixture(new Box(wall.half[0] * s, wall.half[1] * s, Vec2(wall.center[0] * s, wall.center[1] * s), wall.angle),
      { friction: wall.friction });
  }
  const bodies: Record<string, Body> = {};
  for (const circle of scene.circles) {
    const body = world.createDynamicBody({ position: Vec2(circle.center[0] * s, circle.center[1] * s), bullet: true });
    const area = Math.PI * (circle.radius * s) ** 2;
    body.createFixture(new Circle(circle.radius * s), { density: circle.mass_kg / area, friction: circle.friction });
    bodies[circle.id] = body;
  }
  for (const rect of scene.rectangles) {
    const body = world.createDynamicBody({ position: Vec2(rect.center[0] * s, rect.center[1] * s), angle: rect.angle });
    const area = 4 * rect.half[0] * rect.half[1] * s * s;
    body.createFixture(new Box(rect.half[0] * s, rect.half[1] * s), { density: rect.mass_kg / area, friction: rect.friction });
    bodies[rect.id] = body;
  }
  const ground = world.createBody();
  const bladeArea = (profile.length * profile.thickness + profile.heelLength * (profile.thickness + profile.heelHeight)) * s * s;
  const blades = plans.map(plan => {
    const start = poseAt(plan.waypoints, 0);
    const body = world.createDynamicBody({ position: Vec2(start.x * s, start.z * s), angle: start.angle, bullet: true });
    bladeFixtures(body, profile, plan.flip, opt.bladeMassKg / bladeArea);
    for (let fixture = body.getFixtureList(); fixture; fixture = fixture.getNext()) fixture.setFilterGroupIndex(-1); // blades pass each other
    body.setGravityScale(0); // the arm carries the tool's weight
    const motor = world.createJoint(new MotorJoint({ maxForce: opt.forceLimitN / plans.length * s, maxTorque: opt.torqueLimitNm / plans.length * s * s,
      correctionFactor: 1, linearOffset: Vec2(start.x * s, start.z * s), angularOffset: start.angle }, ground, body))!;
    return { plan, body, motor };
  });
  const target = bodies[scene.target_id];
  const touching = (first: Body, second: Body) => {
    for (let edge = first.getContactList(); edge; edge = edge.next) {
      if (edge.other === second && edge.contact.isTouching()) return true;
    }
    return false;
  };
  const snapshot = () => Object.fromEntries(Object.entries(bodies).map(([id, body]) => {
    const p = body.getPosition();
    return [id, { x: p.x / s, z: p.y / s, angle: body.getAngle() }];
  }));

  const frames: Frame[] = [];
  const motion = Math.max(...plans.map(plan => pathSeconds(plan.waypoints)));
  const total = opt.settleSeconds + motion + opt.holdSeconds;
  let settled: Record<string, Pose> = {};
  let peakForce = 0, trackingMaxM = 0, trackingMaxDeg = 0;
  let jamStage: string | null = null, stalledM = 0, finalErrorM = 0, finalErrorDeg = 0;
  const steps = Math.ceil(total / opt.dt);
  for (let step = 0; step <= steps; step++) {
    const t = step * opt.dt;
    const pathTime = Math.max(0, t - opt.settleSeconds);
    const commands = blades.map(({ plan, body, motor }) => {
      const command = poseAt(plan.waypoints, Math.max(0, t + opt.dt - opt.settleSeconds));
      motor.setLinearOffset(Vec2(command.x * s, command.z * s));
      motor.setAngularOffset(body.getAngle() + wrap(command.angle - body.getAngle()));
      return command;
    });
    world.step(opt.dt, 16, 8);
    let force = 0;
    const poses = blades.map(({ body, motor }, index) => {
      force += motor.getReactionForce(1 / opt.dt).length() / s;
      const p = body.getPosition();
      const command = commands[index];
      const errorM = Math.hypot(p.x / s - command.x, p.y / s - command.z);
      const errorDeg = Math.abs(wrap(body.getAngle() - command.angle)) * 180 / Math.PI;
      if (t >= opt.settleSeconds && CONTACT_STAGES.has(command.stage)) stalledM = Math.max(stalledM, errorM);
      else if (t >= opt.settleSeconds) {
        trackingMaxM = Math.max(trackingMaxM, errorM); trackingMaxDeg = Math.max(trackingMaxDeg, errorDeg);
        if (!jamStage && (errorM > opt.trackingLimitM || errorDeg > opt.trackingLimitDeg)) jamStage = command.stage;
      }
      if (step === steps) finalErrorM = Math.max(finalErrorM, errorM), finalErrorDeg = Math.max(finalErrorDeg, errorDeg);
      return { x: p.x / s, z: p.y / s, angle: body.getAngle() };
    });
    if (t >= opt.settleSeconds) peakForce = Math.max(peakForce, force);
    if (Math.abs(t - opt.settleSeconds) < opt.dt / 2) settled = snapshot();
    if (step % opt.frameEvery === 0 || step === steps) {
      frames.push({ t, stage: t < opt.settleSeconds ? 'settle' : poseAt(plans[0].waypoints, pathTime).stage,
        blade: poses[0], command: commands[0], blades: poses, commands: commands.map(({ x, z, angle }) => ({ x, z, angle })),
        bodies: snapshot(), forceN: force });
    }
  }
  const final = snapshot();
  const [ux, uz] = scene.frame.up;
  const liftM = (final[scene.target_id].x - settled[scene.target_id].x) * ux + (final[scene.target_id].z - settled[scene.target_id].z) * uz;
  const requiredLiftM = Math.max(0, scene.frame.rim_up_m) + scene.target.radius;
  const targetOnBlade = blades.some(({ body }) => touching(target, body));
  const targetTouchesOthers = Object.entries(bodies).some(([id, body]) => id !== scene.target_id && touching(target, body));
  const neighborMaxM = Math.max(0, ...Object.keys(bodies).filter(id => id !== scene.target_id)
    .map(id => Math.hypot(final[id].x - settled[id].x, final[id].z - settled[id].z)));
  const reasons = [
    liftM < requiredLiftM ? `상자 밖으로 못 나옴 (들린 높이 ${(liftM * 1000).toFixed(0)} / 필요 ${(requiredLiftM * 1000).toFixed(0)} mm)` : '',
    !targetOnBlade ? '끝에서 날 위에 없음' : '',
    targetTouchesOthers ? '다른 원통에 걸려 있음' : '',
    neighborMaxM > opt.neighborLimitM ? `이웃 원통이 ${(neighborMaxM * 1000).toFixed(0)} mm 밀림` : '',
    // A blade may lag while it drags the load; it has jammed only if it never reaches the end of its path.
    finalErrorM > opt.trackingLimitM || finalErrorDeg > opt.trackingLimitDeg
      ? `날이 끝 자세에 못 감 (${(finalErrorM * 1000).toFixed(0)} mm / ${finalErrorDeg.toFixed(0)}°, 처음 밀린 단계 '${jamStage}')` : '',
  ].filter(Boolean);
  return { success: reasons.length === 0, reason: reasons.join(' · ') || '성공', steps: plans[0].waypoints.length - 1, seconds: motion,
    liftM, requiredLiftM, targetOnBlade, targetTouchesOthers, neighborMaxM, peakForceN: peakForce, trackingMaxM, trackingMaxDeg, jamStage, stalledM, finalErrorM, frames };
}

/** Let the authored scene settle under gravity (no blade) and return it with the resting positions. */
export function settleScene(scene: Scene, seconds = 1.5, dt = 1 / 240): Scene {
  const s = SCALE;
  const world = new World({ gravity: Vec2(0, -9.81 * s) });
  for (const wall of scene.walls) {
    world.createBody().createFixture(new Box(wall.half[0] * s, wall.half[1] * s, Vec2(wall.center[0] * s, wall.center[1] * s), wall.angle),
      { friction: wall.friction });
  }
  const bodies = new Map<string, Body>();
  for (const circle of scene.circles) {
    const body = world.createDynamicBody({ position: Vec2(circle.center[0] * s, circle.center[1] * s), bullet: true });
    body.createFixture(new Circle(circle.radius * s), { density: circle.mass_kg / (Math.PI * (circle.radius * s) ** 2), friction: circle.friction });
    bodies.set(circle.id, body);
  }
  for (const rect of scene.rectangles) {
    const body = world.createDynamicBody({ position: Vec2(rect.center[0] * s, rect.center[1] * s), angle: rect.angle });
    body.createFixture(new Box(rect.half[0] * s, rect.half[1] * s), { density: rect.mass_kg / (4 * rect.half[0] * rect.half[1] * s * s), friction: rect.friction });
    bodies.set(rect.id, body);
  }
  for (let t = 0; t < seconds; t += dt) world.step(dt, 16, 8);
  const at = (id: string) => { const p = bodies.get(id)!.getPosition(); return [p.x / s, p.y / s]; };
  const circles = scene.circles.map(circle => ({ ...circle, center: at(circle.id) }));
  const rectangles = scene.rectangles.map(rect => ({ ...rect, center: at(rect.id), angle: bodies.get(rect.id)!.getAngle() }));
  return { ...scene, circles, rectangles, target: circles.find(circle => circle.id === scene.target_id)! };
}
