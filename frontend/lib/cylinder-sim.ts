// 2-D check of a scoop path in the cylinder's cross-section (Planck.js / Box2D).
// Runs unchanged in the browser lab page and in the Node sweep script.
import { World, Vec2, Circle, Box, Polygon, Settings, MotorJoint, DistanceJoint, type Body } from 'planck';

// Box2D is tuned for objects around 0.1–10 units; working in decimetres keeps a 13.5 cm cylinder
// in that range and makes the solver's 5 mm default slop 0.5 mm, small enough for a few-mm wedge.
const SCALE = 10;
Settings.linearSlop = .005;

export type Waypoint = { stage: string; label: string; tcp: number[]; angle: number; duration: number };
export type Scene = {
  target_id: string;
  target: { id: string; center: number[]; radius: number; name?: string };
  circles: { id: string; name: string; center: number[]; radius: number; mass_kg: number; friction: number }[];
  rectangles: { id: string; center: number[]; half: number[]; angle: number; mass_kg: number; friction: number }[];
  walls: { id: string; center: number[]; half: number[]; angle: number; friction: number }[];
  frame: { origin: number[]; far: number[]; up: number[]; rim_up_m: number };
  /** The open box the target sits in (null when it is not in one). */
  box?: { center: number[]; angle: number; width: number; height: number; thickness: number } | null;
};
/** Tool section in its own frame: x forward, y = top-face normal, TCP at the origin on the top face. */
export type BladeProfile = { tipAhead: number; length: number; thickness: number; heelLength: number; heelHeight: number; friction: number };
// NSE scoop: blade 0.025–0.32 m, neck 0.018–0.065 m, TCP at the tip.
export const NSE_SCOOP_PROFILE: BladeProfile = { tipAhead: 0, length: .295, thickness: .006, heelLength: .047, heelHeight: .02, friction: .5 };
/**
 * Paper cups give way: a soft cylinder is a ring of discs sprung to a hub. `mmPer10N` is how far the
 * cylinder squashes between two flat plates under 10 N; `scope` makes only the target and the
 * cylinders touching it soft (faster) or every cylinder. `shape` is the outline: 'smooth' packs 48
 * overlapping discs for a round surface (about 1 mm ripple), 'beads' uses 16 (about 5 mm bumps).
 */
export type SoftShape = 'smooth' | 'beads';
export type Softness = { mmPer10N: number; scope: 'neighbors' | 'all'; shape: SoftShape };
export const DEFAULT_SOFTNESS: Softness = { mmPer10N: 5, scope: 'neighbors', shape: 'smooth' };
/** Friction overrides (Box2D mixes two surfaces as sqrt(a*b)); `rolling` damps a rigid cylinder's spin (1/s). */
export type Friction = { cylinder?: number; blade?: number; wall?: number; rolling?: number };
/** The robot moves the blade through a force-limited pose tracker; a jammed blade falls behind its path. */
export type SimOptions = { settleSeconds: number; holdSeconds: number; dt: number; forceLimitN: number; torqueLimitNm: number;
  trackingLimitM: number; trackingLimitDeg: number; neighborLimitM: number; bladeMassKg: number; frameEvery: number; soft: Softness | null;
  friction: Friction;
  /** Mass of every cylinder in kg (the whole bundle, carried by all tools together); null keeps the project's values. */
  cylinderMassKg: number | null };
export const DEFAULT_SIM: SimOptions = { settleSeconds: 1, holdSeconds: 1.5, dt: 1 / 240, forceLimitN: 60, torqueLimitNm: 8,
  trackingLimitM: .01, trackingLimitDeg: 5, neighborLimitM: .02, bladeMassKg: .3, frameEvery: 8, soft: null, friction: {},
  cylinderMassKg: null };

export type Pose = { x: number; z: number; angle: number };
export type Frame = { t: number; stage: string; blade: Pose; command: Pose; blades: Pose[]; commands: Pose[];
  bodies: Record<string, Pose>; shapes: Record<string, number[][]>; forceN: number };
/** Stages where the blade is meant to press on the pile; stopping short there is the wedge seating, not a failure. */
export const CONTACT_STAGES = new Set(['insert', 'squeeze', 'pry', 'wrap', 'slide', 'tilt', 'draw']);
export type SimResult = {
  success: boolean; reason: string; steps: number; seconds: number;
  liftM: number; requiredLiftM: number; targetOnBlade: boolean; targetTouchesOthers: boolean;
  neighborMaxM: number; peakForceN: number; trackingMaxM: number; trackingMaxDeg: number; jamStage: string | null; stalledM: number;
  finalErrorM: number; maxSquashMm: number; outsideBox: boolean; warnings: string[]; frames: Frame[];
};

/**
 * Whether a cylinder has left the box: its whole section is above the rim, or it is past a side wall.
 * Without a box, being lifted by the required height counts as out.
 */
export function isOutsideBox(scene: Scene, center: number[], radius: number, liftM: number, requiredLiftM: number) {
  const box = scene.box;
  if (!box) return liftM >= requiredLiftM;
  const dx = center[0] - box.center[0], dz = center[1] - box.center[1];
  const c = Math.cos(box.angle), s = Math.sin(box.angle);
  const along = dx * c + dz * s, up = -dx * s + dz * c; // box axes: along the floor, out of the opening
  return up > box.height / 2 + radius || Math.abs(along) > box.width / 2 + radius;
}

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

// ---------------------------------------------------------------- soft cylinders

type SoftCylinder = { hub: Body; nodes: Body[]; radius: number; nodeRadius: number };

/** Ring layout for a shape: node count, ring radius and disc radius, with the outline at `radius`. */
export function softLayout(radius: number, shape: SoftShape) {
  if (shape === 'smooth') {
    const nodes = 48;
    const ring = radius / (1 + 2 * Math.PI / nodes); // discs twice as wide as their spacing
    return { nodes, ring, nodeRadius: radius - ring };
  }
  const nodes = 16, nodeRadius = Math.PI * radius / nodes * 1.05;
  return { nodes, ring: radius - nodeRadius, nodeRadius };
}

/** Build a soft cylinder: discs on a ring, neighbours tied rigidly, each sprung to a stiff hub. */
function softCylinder(world: World, center: number[], radius: number, mass: number, friction: number, shape: SoftShape,
  frequencyHz: number, group: number): SoftCylinder {
  const s = SCALE;
  const { nodes, ring, nodeRadius } = softLayout(radius, shape);
  const hubRadius = radius * .35; // the nested cups' walls: the stack cannot collapse flat
  const hub = world.createDynamicBody({ position: Vec2(center[0] * s, center[1] * s), bullet: true });
  hub.createFixture(new Circle(hubRadius * s), { density: mass * .4 / (Math.PI * (hubRadius * s) ** 2), friction, filterGroupIndex: group });
  const nodeBodies: Body[] = [];
  for (let i = 0; i < nodes; i++) {
    const a = 2 * Math.PI * i / nodes;
    const body = world.createDynamicBody({ position: Vec2((center[0] + ring * Math.cos(a)) * s, (center[1] + ring * Math.sin(a)) * s), bullet: true });
    body.createFixture(new Circle(nodeRadius * s), { density: mass * .6 / nodes / (Math.PI * (nodeRadius * s) ** 2), friction, filterGroupIndex: group });
    body.setFixedRotation(true);
    nodeBodies.push(body);
  }
  const bend = Math.max(2, Math.round(nodes / 8)); // bending springs span about 45 degrees whatever the node count
  for (let i = 0; i < nodes; i++) {
    const a = nodeBodies[i], b = nodeBodies[(i + 1) % nodes], c = nodeBodies[(i + bend) % nodes];
    world.createJoint(new DistanceJoint({ frequencyHz: 0, dampingRatio: 0 }, a, b, a.getPosition(), b.getPosition())); // paper does not stretch
    world.createJoint(new DistanceJoint({ frequencyHz: frequencyHz * 1.5, dampingRatio: .7 }, a, c, a.getPosition(), c.getPosition())); // bending
    world.createJoint(new DistanceJoint({ frequencyHz, dampingRatio: .7 }, hub, a, hub.getPosition(), a.getPosition()));
  }
  return { hub, nodes: nodeBodies, radius, nodeRadius };
}

const frequencyCache = new Map<string, number>();

/** Squash (mm) of a soft cylinder pressed with `forceN` between a floor and a plate, zero gravity. */
function squashUnder(radius: number, mass: number, shape: SoftShape, frequencyHz: number, forceN: number) {
  const s = SCALE;
  const world = new World({ gravity: Vec2(0, 0) });
  world.createBody().createFixture(new Box(.3 * s, .01 * s, Vec2(0, -.01 * s), 0));
  softCylinder(world, [0, radius], radius, mass, .7, shape, frequencyHz, -2);
  const plate = world.createDynamicBody({ position: Vec2(0, (2 * radius + .01) * s), fixedRotation: true });
  plate.createFixture(new Box(.3 * s, .01 * s), { density: .5 / (.6 * .02 * s * s) });
  for (let t = 0; t < 1.2; t += 1 / 240) {
    plate.applyForceToCenter(Vec2(0, -forceN * s));
    world.step(1 / 240, 16, 8);
  }
  return (2 * radius - (plate.getPosition().y / s - .01)) * 1000;
}

/** Spring frequency that gives the requested squash, found by bisection and cached. */
export function softFrequency(radius: number, mass: number, soft: Softness) {
  const key = `${radius.toFixed(4)}:${mass}:${soft.shape}:${soft.mmPer10N}`;
  const cached = frequencyCache.get(key);
  if (cached) return cached;
  let low = .5, high = 150; // softer below, stiffer above
  for (let i = 0; i < 14; i++) {
    const middle = Math.sqrt(low * high);
    if (squashUnder(radius, mass, soft.shape, middle, 10) > soft.mmPer10N) low = middle; else high = middle;
  }
  const result = Math.sqrt(low * high);
  frequencyCache.set(key, result);
  return result;
}

// ---------------------------------------------------------------- scene

type Built = { world: World; bodies: Record<string, Body>; soft: Record<string, SoftCylinder> };

function softIds(scene: Scene, soft: Softness | null) {
  if (!soft) return new Set<string>();
  if (soft.scope === 'all') return new Set(scene.circles.map(c => c.id));
  const [tx, tz] = scene.target.center;
  return new Set(scene.circles.filter(c => Math.hypot(c.center[0] - tx, c.center[1] - tz) < c.radius + scene.target.radius + .02).map(c => c.id));
}

function buildScene(scene: Scene, soft: Softness | null, friction: Friction = {}, massKg: number | null = null): Built {
  const s = SCALE;
  const world = new World({ gravity: Vec2(0, -9.81 * s) });
  for (const wall of scene.walls) {
    world.createBody().createFixture(new Box(wall.half[0] * s, wall.half[1] * s, Vec2(wall.center[0] * s, wall.center[1] * s), wall.angle),
      { friction: friction.wall ?? wall.friction });
  }
  const bodies: Record<string, Body> = {}, softBodies: Record<string, SoftCylinder> = {};
  const softSet = softIds(scene, soft);
  let group = -2; // -1 is the blades
  for (const source of scene.circles) {
    const circle = massKg ? { ...source, mass_kg: massKg } : source;
    const mu = friction.cylinder ?? circle.friction;
    if (soft && softSet.has(circle.id)) {
      const cylinder = softCylinder(world, circle.center, circle.radius, circle.mass_kg, mu, soft.shape,
        softFrequency(circle.radius, circle.mass_kg, soft), group--);
      softBodies[circle.id] = cylinder;
      bodies[circle.id] = cylinder.hub;
      continue;
    }
    const body = world.createDynamicBody({ position: Vec2(circle.center[0] * s, circle.center[1] * s), bullet: true,
      angularDamping: friction.rolling ?? 0 });
    body.createFixture(new Circle(circle.radius * s), { density: circle.mass_kg / (Math.PI * (circle.radius * s) ** 2), friction: mu });
    bodies[circle.id] = body;
  }
  for (const rect of scene.rectangles) {
    const body = world.createDynamicBody({ position: Vec2(rect.center[0] * s, rect.center[1] * s), angle: rect.angle });
    body.createFixture(new Box(rect.half[0] * s, rect.half[1] * s), { density: rect.mass_kg / (4 * rect.half[0] * rect.half[1] * s * s), friction: friction.cylinder ?? rect.friction });
    bodies[rect.id] = body;
  }
  return { world, bodies, soft: softBodies };
}

/** Bodies of one scene object (a soft cylinder is its hub plus its ring). */
function partsOf(built: Built, id: string) {
  const soft = built.soft[id];
  return soft ? [soft.hub, ...soft.nodes] : [built.bodies[id]];
}

function bladeFixtures(body: Body, profile: BladeProfile, flip: boolean, density: number) {
  const s = SCALE;
  const y = (value: number) => (flip ? -value : value) * s; // a rolled-over tool has its top face on the other side
  const back = profile.tipAhead - profile.length;
  const quad = (x0: number, y0: number, x1: number, y1: number) => {
    const points = [Vec2(x0 * s, y(y0)), Vec2(x1 * s, y(y0)), Vec2(x1 * s, y(y1)), Vec2(x0 * s, y(y1))];
    return new Polygon(flip ? points.reverse() : points);
  };
  body.createFixture(quad(back, -profile.thickness, profile.tipAhead, 0), { friction: profile.friction, density, filterGroupIndex: -1 });
  if (profile.heelLength > 0 && profile.heelHeight > 0) {
    body.createFixture(quad(back - profile.heelLength, -profile.thickness, back, profile.heelHeight), { friction: profile.friction, density, filterGroupIndex: -1 });
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
  const built = buildScene(scene, opt.soft, opt.friction, opt.cylinderMassKg);
  const { world, bodies } = built;
  if (opt.friction.blade !== undefined) profile = { ...profile, friction: opt.friction.blade };
  const ground = world.createBody();
  const bladeArea = (profile.length * profile.thickness + profile.heelLength * (profile.thickness + profile.heelHeight)) * s * s;
  const blades = plans.map(plan => {
    const start = poseAt(plan.waypoints, 0);
    const body = world.createDynamicBody({ position: Vec2(start.x * s, start.z * s), angle: start.angle, bullet: true });
    bladeFixtures(body, profile, plan.flip, opt.bladeMassKg / bladeArea);
    body.setGravityScale(0); // the arm carries the tool's weight
    const motor = world.createJoint(new MotorJoint({ maxForce: opt.forceLimitN / plans.length * s, maxTorque: opt.torqueLimitNm / plans.length * s * s,
      correctionFactor: 1, linearOffset: Vec2(start.x * s, start.z * s), angularOffset: start.angle }, ground, body))!;
    return { plan, body, motor };
  });
  const targetParts = new Set(partsOf(built, scene.target_id));
  const touchingAny = (parts: Body[], others: Set<Body>) => parts.some(part => {
    for (let edge = part.getContactList(); edge; edge = edge.next) if (others.has(edge.other!) && edge.contact.isTouching()) return true;
    return false;
  });
  const snapshot = () => Object.fromEntries(Object.entries(bodies).map(([id, body]) => {
    const p = body.getPosition();
    return [id, { x: p.x / s, z: p.y / s, angle: body.getAngle() }];
  }));
  // Outline of each soft cylinder: every disc pushed out from the hub by its radius.
  const shapes = () => Object.fromEntries(Object.entries(built.soft).map(([id, cylinder]) => {
    const hub = cylinder.hub.getPosition();
    return [id, cylinder.nodes.map(node => {
      const p = node.getPosition(), d = Vec2.distance(p, hub) || 1, k = (d + cylinder.nodeRadius * s) / d;
      return [(hub.x + (p.x - hub.x) * k) / s, (hub.y + (p.y - hub.y) * k) / s];
    })];
  }));
  const squashMm = () => Math.max(0, ...Object.values(built.soft).map(cylinder => {
    const hub = cylinder.hub.getPosition();
    const radii = cylinder.nodes.map(node => Vec2.distance(node.getPosition(), hub) / s);
    return (Math.max(...radii) - Math.min(...radii)) * 1000;
  }));

  const frames: Frame[] = [];
  const motion = Math.max(...plans.map(plan => pathSeconds(plan.waypoints)));
  const total = opt.settleSeconds + motion + opt.holdSeconds;
  let settled: Record<string, Pose> = {};
  let peakForce = 0, trackingMaxM = 0, trackingMaxDeg = 0, maxSquashMm = 0;
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
      if (step === steps) { finalErrorM = Math.max(finalErrorM, errorM); finalErrorDeg = Math.max(finalErrorDeg, errorDeg); }
      return { x: p.x / s, z: p.y / s, angle: body.getAngle() };
    });
    if (t >= opt.settleSeconds) {
      peakForce = Math.max(peakForce, force);
      if (step % 8 === 0) maxSquashMm = Math.max(maxSquashMm, squashMm());
    }
    if (Math.abs(t - opt.settleSeconds) < opt.dt / 2) settled = snapshot();
    if (step % opt.frameEvery === 0 || step === steps) {
      frames.push({ t, stage: t < opt.settleSeconds ? 'settle' : poseAt(plans[0].waypoints, pathTime).stage,
        blade: poses[0], command: commands[0], blades: poses, commands: commands.map(({ x, z, angle }) => ({ x, z, angle })),
        bodies: snapshot(), shapes: shapes(), forceN: force });
    }
  }
  const final = snapshot();
  const [ux, uz] = scene.frame.up;
  const liftM = (final[scene.target_id].x - settled[scene.target_id].x) * ux + (final[scene.target_id].z - settled[scene.target_id].z) * uz;
  const requiredLiftM = Math.max(0, scene.frame.rim_up_m) + scene.target.radius;
  const targetList = [...targetParts];
  const targetOnBlade = touchingAny(targetList, new Set(blades.map(({ body }) => body)));
  const others = new Set(Object.keys(bodies).filter(id => id !== scene.target_id).flatMap(id => partsOf(built, id)));
  const targetTouchesOthers = touchingAny(targetList, others);
  const neighborMaxM = Math.max(0, ...Object.keys(bodies).filter(id => id !== scene.target_id)
    .map(id => Math.hypot(final[id].x - settled[id].x, final[id].z - settled[id].z)));
  // Success is only whether the target ended up out of the box; everything else is a warning.
  const finalCenter = [final[scene.target_id].x, final[scene.target_id].z];
  const outsideBox = isOutsideBox(scene, finalCenter, scene.target.radius, liftM, requiredLiftM);
  const warnings = [
    outsideBox && !targetOnBlade ? '날 위가 아니라 상자 밖으로 떨어짐' : '',
    targetTouchesOthers ? '끝에서 다른 원통에 닿아 있음' : '',
    neighborMaxM > opt.neighborLimitM ? `이웃 원통이 ${(neighborMaxM * 1000).toFixed(0)} mm 밀림` : '',
    // A blade may lag while it drags the load; it has jammed only if it never reaches the end of its path.
    finalErrorM > opt.trackingLimitM || finalErrorDeg > opt.trackingLimitDeg
      ? `날이 끝 자세에 못 감 (${(finalErrorM * 1000).toFixed(0)} mm / ${finalErrorDeg.toFixed(0)}°${jamStage ? `, 처음 밀린 단계 '${jamStage}'` : ''})` : '',
  ].filter(Boolean);
  const reason = outsideBox ? ['성공 · 상자 밖으로 나옴', ...warnings].join(' · ')
    : [`경로가 끝났는데 상자 안에 있음 (들린 높이 ${(liftM * 1000).toFixed(0)} / 필요 ${(requiredLiftM * 1000).toFixed(0)} mm)`, ...warnings].join(' · ');
  return { success: outsideBox, reason, steps: plans[0].waypoints.length - 1, seconds: motion,
    liftM, requiredLiftM, targetOnBlade, targetTouchesOthers, neighborMaxM, peakForceN: peakForce, trackingMaxM, trackingMaxDeg, jamStage,
    stalledM, finalErrorM, maxSquashMm, outsideBox, warnings, frames };
}

/** Let the authored scene settle under gravity (no blade) and return it with the resting positions. */
export function settleScene(scene: Scene, seconds = 1.5, dt = 1 / 240, soft: Softness | null = null, friction: Friction = {}): Scene {
  const s = SCALE;
  const built = buildScene(scene, soft, friction);
  for (let t = 0; t < seconds; t += dt) built.world.step(dt, 16, 8);
  const at = (id: string) => { const p = built.bodies[id].getPosition(); return [p.x / s, p.y / s]; };
  const circles = scene.circles.map(circle => ({ ...circle, center: at(circle.id) }));
  const rectangles = scene.rectangles.map(rect => ({ ...rect, center: at(rect.id), angle: built.bodies[rect.id].getAngle() }));
  return { ...scene, circles, rectangles, target: circles.find(circle => circle.id === scene.target_id)! };
}

export { squashUnder as softSquashMm };

/** The scene at rest with its target frame moved along (mirrors cylinder_extraction.with_positions). */
export function restingScene<T extends Scene>(source: T): T {
  const rest = settleScene(source);
  const [ox, oz] = source.frame.origin, [nx, nz] = rest.target.center, [ux, uz] = source.frame.up;
  return { ...source, ...rest,
    frame: { ...source.frame, origin: [nx, nz], rim_up_m: source.frame.rim_up_m - ((nx - ox) * ux + (nz - oz) * uz) } };
}
