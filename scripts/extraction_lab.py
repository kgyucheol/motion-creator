#!/usr/bin/env python
"""Design and physically test a ramen-stack extraction motion for the sample scene.

Pipeline: the project's scene (stack cross-sections optionally scaled to open a gap) -> `cylinder_extraction.plan_scoop`
section path -> world TCP poses for both tools (Cylinder Lab's conversion) -> whole-body IK on the *WBC stand* with the
pelvis and feet pinned -> kinematic playback in the editor's decoupled-WBC MuJoCo model (robot prescribed, stacks free)
-> extraction metrics. Not part of the shipped runtime; a design aid.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motioncreator.cylinder_extraction import plan_scoop, scene_2d
from motioncreator.decoupled_wbc import DecoupledSimulation
from motioncreator.robot import FEET, Robot

ROOT = Path(__file__).resolve().parents[1]
HANDS = ("left_hand", "right_hand")
FPS = 30


def sample_project(folder: str | None = None) -> dict:
    folder = Path(folder) if folder else next(ROOT.glob("motions/왼기본주걱_오기본받침*"))
    return json.loads((folder / "project.json").read_text())


def scale_stacks(project: dict, factor: float, target_name: str = "Mupama_Stack") -> dict:
    """Shrink every stack's cross-section about its own centre (length untouched): opens a gap between neighbours."""
    project = copy.deepcopy(project)
    for item in project["scene_objects"]:
        if item["name"].startswith(target_name):
            item["size"] = [item["size"][0] * factor, item["size"][1], item["size"][2] * factor]
    return project


def target_of(project: dict, name: str = "Mupama_Stack_L03_R04") -> dict:
    return next(item for item in project["scene_objects"] if item["name"] == name)


def tool_quaternion(angle: float, flip: bool) -> np.ndarray:
    """Cylinder Lab's rule with the identity base frame (robot straight behind the box): a rotation about world Y that
    turns the tool's forward axis to `angle` (measured from +X toward +Z), rolled over when `flip`."""
    rotation = Rotation.from_rotvec([0.0, -angle, 0.0])
    if flip:
        rotation = rotation * Rotation.from_rotvec([math.pi, 0.0, 0.0])
    return rotation.as_quat()


def tool_tracks(project: dict, params: dict, lateral: float = 0.25, side_lateral: dict | None = None):
    """{side: [(duration, position xyz, quaternion xyzw, stage)]} for both tools from the section path."""
    target = target_of(project)
    scene = scene_2d(project, target["id"])
    plan = plan_scoop(scene, params)
    blades = plan["blades"]
    tracks = {}
    for index, side in enumerate(("left", "right")):
        blade = blades[min(index, len(blades) - 1)]
        offset = (side_lateral or {}).get(side, lateral)
        y = target["position"][1] + (offset if side == "left" else -offset)
        tracks[side] = [(w["duration"], np.array([w["tcp"][0], y, w["tcp"][1]]), tool_quaternion(w["angle"], blade["flip"]),
                         w["stage"]) for w in blade["waypoints"]]
    return plan, tracks


def solve_tracks(robot: Robot, base_q: np.ndarray, tracks: dict, lock_waist: bool = False, nfev: int = 300,
                 seed_q: np.ndarray | None = None):
    """IK at every waypoint from the pinned lower body. Returns (qpos list, failures)."""
    count = len(tracks["left"])
    angle_pins = ["pelvis", "waist"] if lock_waist else ["pelvis"]
    seed, solved, failures = robot.validate_q(base_q if seed_q is None else seed_q), [], []
    for index in range(count):
        targets = {hand: tracks[side][index][1] for hand, side in zip(HANDS, ("left", "right"))}
        orientations = {hand: tracks[side][index][2] for hand, side in zip(HANDS, ("left", "right"))}
        best = None
        initial = seed
        for weight, evaluations in ((.8, nfev // 2), (.3, nfev), (.08, nfev * 2)):
            candidate, info = robot.solve(initial, base_q, pins=(*FEET, "pelvis"), angle_pins=angle_pins, mode="free",
                                          resistance=0., selected_targets=targets, orientation_targets=orientations,
                                          posture_reference=seed, posture_weight=weight, max_nfev=evaluations)
            ok = info["converged"] and not info.get("rejected")
            if best is None or ok or info["target_error_mm"] < best[1]["target_error_mm"]:
                best = (candidate, info)
            if ok:
                break
            initial = candidate if not info.get("rejected") else seed
        candidate, info = best
        if not (info["converged"] and not info.get("rejected")):
            failures.append((index, tracks["left"][index][3], round(float(info["target_error_mm"]), 1), round(float(info["angle_error_deg"]), 1)))
        solved.append(candidate)
        seed = candidate
    return solved, failures


class KinematicRig:
    """Editor decoupled-WBC model with the robot prescribed and every scene object dynamic."""

    def __init__(self, robot: Robot, project: dict):
        self.sim = DecoupledSimulation(robot, project, autostart=False)
        self.model = self.sim.model
        self.bodies = dict(self.sim.object_bodies)
        self.robot = robot
        self.sides = {side: _descendants(self.model, self.model.body(f"{side}_wrist_yaw_link").id) for side in ("left", "right")}
        self.object_of_geom = {g: i for i, geoms in self.sim.object_geoms.items() for g in geoms}

    def _object_pos(self, data, identifier):
        return data.xpos[self.bodies[identifier]].copy()

    def run(self, pose_at, duration: float, target_id: str, settle: float = 1.0, sample_every: int = 10):
        """Prescribe the robot, integrate the objects. Returns start/end poses, force statistics and the target's trace."""
        model = self.model
        data = mujoco.MjData(model)
        data.qpos[:] = self.sim.data.qpos
        data.qpos[:36] = pose_at(0.0)
        mujoco.mj_forward(model, data)
        for _ in range(int(settle / model.opt.timestep)):          # let the packed stacks settle under gravity
            data.qpos[:36] = pose_at(0.0)
            data.qvel[:35] = 0
            mujoco.mj_step(model, data)
        start = {i: self._object_pos(data, i) for i in self.bodies}
        steps = int(duration / model.opt.timestep)
        stats = {"max_any_n": 0.0, "max_target_n": {"left": 0.0, "right": 0.0}, "end_target_n": {"left": 0.0, "right": 0.0}}
        trace, wrench = [], np.zeros(6)
        tail = max(1, int(0.5 / model.opt.timestep / sample_every))
        window = []
        for step in range(steps):
            t = step * model.opt.timestep
            data.qpos[:36] = pose_at(t)
            data.qvel[:35] = 0
            mujoco.mj_step(model, data)
            if step % sample_every == 0:
                forces = {"left": 0.0, "right": 0.0}
                for index in range(data.ncon):
                    contact = data.contact[index]
                    for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                        body = int(model.geom_bodyid[mine])
                        side = next((k for k, v in self.sides.items() if body in v), None)
                        if side is None or other not in self.object_of_geom:
                            continue
                        mujoco.mj_contactForce(model, data, index, wrench)
                        normal = abs(float(wrench[0]))
                        stats["max_any_n"] = max(stats["max_any_n"], normal)
                        if self.object_of_geom[other] == target_id:
                            forces[side] += normal
                for side in forces:
                    stats["max_target_n"][side] = max(stats["max_target_n"][side], forces[side])
                window.append(forces)
                window = window[-tail:]
                trace.append((t, self._object_pos(data, target_id)))
            if not np.isfinite(data.qpos).all():
                break
        stats["end_target_n"] = {side: float(np.mean([w[side] for w in window])) for side in ("left", "right")} if window else stats["end_target_n"]
        end = {i: self._object_pos(data, i) for i in self.bodies}
        return {"start": start, "end": end, "trace": trace, **stats}


def _descendants(model, root_body):
    found, changed = {root_body}, True
    while changed:
        changed = False
        for body in range(model.nbody):
            if body not in found and model.body_parentid[body] in found:
                found.add(body)
                changed = True
    return found


def pose_function(qposes: list[np.ndarray], durations: list[float], hold_start: float = 0.5, speed: float = 1.0):
    """Prescribed robot pose through keyed qpos: key i is reached `durations[i]` after key i-1 (key 0 after `hold_start`)."""
    times = hold_start + np.concatenate([[0.0], np.cumsum(np.maximum(durations[1:], 0.05) / speed)])
    times = np.concatenate([[0.0], times])
    keys = np.stack([qposes[0], *qposes])
    rotations = Slerp(times, Rotation.from_quat(keys[:, 3:7], scalar_first=True))

    def pose(t):
        t = float(np.clip(t, times[0], times[-1]))
        q = np.empty(keys.shape[1])
        for column in (*range(3), *range(7, keys.shape[1])):
            q[column] = np.interp(t, times, keys[:, column])
        q[3:7] = rotations([t]).as_quat(scalar_first=True)[0]
        return q
    return pose, float(times[-1])


def extraction_report(project: dict, result: dict, target_id: str) -> dict:
    names = {item["id"]: item["name"] for item in project["scene_objects"]}
    shift = result["end"][target_id] - result["start"][target_id]
    moved = {names[i]: float(np.linalg.norm(result["end"][i] - result["start"][i])) for i in result["end"]
             if i != target_id and np.linalg.norm(result["end"][i] - result["start"][i]) > 0.005 and names[i].startswith("Mupama")}
    return {"target_shift_m": np.round(shift, 3).tolist(), "rise_m": round(float(shift[2]), 3), "toward_robot_m": round(float(-shift[0]), 3),
            "max_any_n": round(result["max_any_n"], 1), "max_target_n": {k: round(v, 1) for k, v in result["max_target_n"].items()},
            "end_target_n": {k: round(v, 1) for k, v in result["end_target_n"].items()},
            "others_moved_m": {k[-7:]: round(v, 3) for k, v in moved.items()}}


# ---- the hand-described technique: horizontal plate, small pushes with a wiggle, lift while tilting toward the body ----

LIFT_STEPS, CARRY_STEPS, APPROACH_STEPS = 4, 3, 4        # compact on purpose: every step is one editor keyframe
MANEUVER_DEFAULTS = {
    "dx0": -0.15, "dz0": -0.05, "th0_deg": 0.0, "phi_deg": 0.0, "step_m": 0.008, "depth_m": 0.10, "push_s": 0.5,
    "wiggle_z_m": 0.004, "wiggle_th_deg": 4.0, "lift_m": 0.04, "tilt_pull_m": 0.02, "th1_deg": 30.0,
    "carry_pull_m": 0.12, "carry_lift_m": 0.15, "move_s": 0.8,
}


def maneuver_points(params: dict, center_xz) -> list[tuple[float, np.ndarray, float, str]]:
    """(duration, tcp xz, tool pitch, stage) for: hover -> place -> N x (push, wiggle) -> lift+tilt -> carry."""
    q = {**MANEUVER_DEFAULTS, **params}
    p = np.array([center_xz[0] + q["dx0"], center_xz[1] + q["dz0"]], dtype=float)
    th = math.radians(q["th0_deg"])
    hover = p + np.array([-0.06, 0.05])
    out = [(0.1, hover + np.array([0.0, 0.16]), th, "clear"), (1.2, hover, th, "hover"), (q["move_s"], p.copy(), th, "place")]
    direction = np.array([math.cos(math.radians(q["phi_deg"])), math.sin(math.radians(q["phi_deg"]))])
    pushes = max(1, round(q["pushes"])) if "pushes" in q else max(1, math.ceil(q["depth_m"] / q["step_m"]))
    q["step_m"] = q["depth_m"] / pushes
    for _ in range(pushes):
        p = p + direction * q["step_m"]
        out.append((q["push_s"], p.copy(), th, "push"))
        if q["wiggle_z_m"] > 0 or q["wiggle_th_deg"] > 0:
            wz, wt = q["wiggle_z_m"], math.radians(q["wiggle_th_deg"])
            out += [(0.12, p + [0, wz], th + wt, "wiggle"), (0.12, p - [0, wz], th - wt, "wiggle"), (0.12, p.copy(), th, "wiggle")]
    th1 = math.radians(q["th1_deg"])
    base = p.copy()
    for k in range(1, LIFT_STEPS + 1):
        f = k / LIFT_STEPS
        out.append((q["move_s"] / LIFT_STEPS * 2, base + np.array([-q["tilt_pull_m"] * f, q["lift_m"] * f]), th + (th1 - th) * f, "lift_tilt"))
    base = out[-1][1]
    for k in range(1, CARRY_STEPS + 1):
        f = k / CARRY_STEPS
        out.append((q["move_s"] / CARRY_STEPS, base + np.array([-q["carry_pull_m"] * f, q["carry_lift_m"] * f]), th1, "carry"))
    return out


def maneuver_tracks(params: dict, project: dict, lateral: float = 0.25) -> dict:
    target = target_of(project)
    points = maneuver_points(params, (target["position"][0], target["position"][2]))
    tracks = {}
    for side in ("left", "right"):
        y = target["position"][1] + (lateral if side == "left" else -lateral)
        tracks[side] = [(d, np.array([xz[0], y, xz[1]]), tool_quaternion(th, False), stage) for d, xz, th, stage in points]
    return tracks


APPROACH_S = 2.5
RETURN_S = 3.0
PLACE_HOLD_S = 1.5


def neutral_pose(robot: Robot, pelvis_x: float, forward_m: float = 0.20, height_m: float = 1.05, tip_up_deg: float = 70.0,
                 lateral: float = 0.25) -> np.ndarray:
    """Guard pose to start and settle in: WBC stand, both tools in front of the chest with the blades pointing up, well
    clear of the stacks. (Hanging arms put the blades at thigh height, too far below the box to reach it safely.)"""
    base = robot.wbc_stand.copy()
    base[0] = pelvis_x
    seed = base.copy()
    for side in ("left", "right"):
        for joint in ("shoulder_roll", "elbow"):
            address = robot.model.joint(f"{side}_{joint}_joint").qposadr[0]
            seed[address] = robot.attention[address]
    targets = {hand: np.array([pelvis_x + forward_m, sign * lateral, height_m]) for hand, sign in zip(HANDS, (1.0, -1.0))}
    orientations = {hand: tool_quaternion(math.radians(tip_up_deg), False) for hand in HANDS}
    q, info = robot.solve(robot.validate_q(seed), base, pins=(*FEET, "pelvis"), angle_pins=["pelvis"], mode="free", resistance=0.,
                          selected_targets=targets, orientation_targets=orientations, posture_reference=seed,
                          posture_weight=0.3, max_nfev=600)
    if not (info["converged"] and not info.get("rejected")):
        raise RuntimeError(f"guard pose unreachable: {info['target_error_mm']:.0f} mm")
    return q


def neutral_tips(robot: Robot, neutral: np.ndarray) -> dict:
    data = robot.data(neutral)
    out = {}
    for side, hand in zip(("left", "right"), HANDS):
        position, matrix = robot.point(data, hand)
        out[side] = (np.asarray(position, dtype=float), Rotation.from_matrix(matrix).as_quat())
    return out


def build_keyframes(robot: Robot, params: dict, project: dict, pelvis_x: float, lateral: float = 0.25, nfev: int = 200,
                    approach_s: float = APPROACH_S):
    """Guard pose -> straight line over the box to the maneuver start -> maneuver -> hold -> return.
    Returns (qposes, durations, stage names, ik failures)."""
    base = robot.wbc_stand.copy()
    base[0] = pelvis_x
    neutral = neutral_pose(robot, pelvis_x, lateral=lateral)
    tracks = maneuver_tracks(params, project, lateral)
    tips = neutral_tips(robot, neutral)
    steps = APPROACH_STEPS
    pre = {}
    for side in tracks:
        start_position, start_quat = tips[side]
        end_duration, end_position, end_quat, _ = tracks[side][0]
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([start_quat, end_quat]))
        walk = [(0.0, start_position + (end_position - start_position) * (k / steps), slerp([k / steps]).as_quat()[0], "approach")
                for k in range(1, steps)]
        pre[side] = walk + tracks[side]
    stages = ["approach"] * (steps - 1) + [w[3] for w in tracks["left"]]
    durations = [approach_s / steps] * steps + [w[0] for w in tracks["left"][1:]]
    qs, failures = solve_tracks(robot, base, pre, nfev=nfev, seed_q=neutral)
    qposes = [neutral, *qs, qs[-1], qs[-1], neutral]
    stage_names = ["ready", *stages, "place_hold", "place_hold", "return"]
    durations = [0.1, *durations, 0.1, PLACE_HOLD_S, RETURN_S]
    return qposes, durations, stage_names, failures


MAX_KEYFRAMES = 100          # the editor's project limit
PREFERRED_KEYFRAMES = 45     # what a person can still read and edit


def keyframe_count(params: dict) -> int:
    """Keyframes build_keyframes() produces: start + approach walk + maneuver + hold, hold, return."""
    return 1 + (APPROACH_STEPS - 1) + len(maneuver_points(params, (0.0, 0.0))) + 3


def settled_scene(robot: Robot, project: dict, robot_qpos: np.ndarray, seconds: float = 1.5) -> dict:
    """The project with every free object moved to where it comes to rest under gravity (robot held at `robot_qpos`).

    Scaled stacks are loose: left at their authored centres they sink a few centimetres in the first second of any
    simulation. Saving the settled layout makes the controller see the same scene the motion was evaluated on.
    """
    rig = KinematicRig(robot, project)
    model = rig.model
    data = mujoco.MjData(model)
    data.qpos[:] = rig.sim.data.qpos
    data.qpos[:36] = robot_qpos
    mujoco.mj_forward(model, data)
    before = {i: data.xpos[b].copy() for i, b in rig.bodies.items()}
    for _ in range(int(seconds / model.opt.timestep)):
        data.qpos[:36] = robot_qpos
        data.qvel[:35] = 0
        mujoco.mj_step(model, data)
    moved = copy.deepcopy(project)
    for item in moved["scene_objects"]:
        if item.get("fixed") or item["id"] not in rig.bodies:
            continue
        body = rig.bodies[item["id"]]
        item["position"] = (np.asarray(item["position"]) + data.xpos[body] - before[item["id"]]).round(6).tolist()
        item["quaternion_xyzw"] = data.xquat[body][[1, 2, 3, 0]].round(8).tolist()
    return moved
