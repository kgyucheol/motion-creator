"""Task-controller session: the ramen_controller FSM driving the editor's in-process decoupled-WBC simulation.

This is a composition layer. It is the one place (with `controller_api`) allowed to import both the authoring
package and the runtime; the runtime itself never imports the authoring package. The FSM talks to the sim through
the same `WbcLink` seam the real-robot adapter uses, so what is tested here is the same task logic.

Clocks: the FSM runs on *simulation* time, so a slow machine never changes behaviour.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ramen_controller.auto_positioner import AutoPositioner
from ramen_controller.contracts import MotionKind, ObjectPose, RobotState, WbcCommand
from ramen_controller.fsm import TaskFsm, TaskFsmConfig
from ramen_controller.motion_bundle import load_segment, root_pose_at
from ramen_controller.motion_library import FixedGraspPlanner, MotionEntry, MotionLibrary
from ramen_controller.sim.proxy_grasp import ProxyGraspMonitor
from ramen_controller.runner import TaskRunner
from ramen_controller.vision import CameraFrame, MockPoseEstimator, VisionService

from .decoupled_wbc import DecoupledSimulation, ROOT, verify_assets
from .robot import Robot

STAGE_NAMES = ("ready", "grasp", "transfer", "place", "return")
DEFAULT_STAGES = ROOT / "configs/sim_motions.json"
CONTROL_RATE_HZ = 50.0


def load_stages(bundle_dir: Path | None) -> dict:
    """Stage time ranges for a bundle: `<bundle>/task_stages.json` if present, else the repository default."""
    for path in ((bundle_dir / "task_stages.json") if bundle_dir else None, DEFAULT_STAGES):
        if path is not None and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError("no task stage mapping found")


def _validated_stages(stages: dict) -> dict:
    for name in STAGE_NAMES:
        item = stages.get(name)
        if not isinstance(item, dict) or "start_s" not in item or "end_s" not in item:
            raise ValueError(f"stage {name!r} needs start_s and end_s")
        if float(item["end_s"]) < float(item["start_s"]):
            raise ValueError(f"stage {name!r} ends before it starts")
    if not stages.get("target_object"):
        raise ValueError("target_object (the scene object to pick) is required")
    start = stages.get("robot_start_xy_yaw")
    if start is not None and (not isinstance(start, (list, tuple)) or len(start) != 3):
        raise ValueError("robot_start_xy_yaw must be [x, y, yaw]")
    return stages


class SimSessionLink:
    """WbcLink over a DecoupledSimulation in external mode."""

    def __init__(self, simulation: DecoupledSimulation):
        self.simulation = simulation

    def read_state(self, stamp: float) -> RobotState:
        simulation = self.simulation
        with simulation.lock:
            qpos = simulation.data.qpos[:36].copy()
            fallen = simulation.phase == "fallen"
        return RobotState(stamp=stamp, joint_positions=qpos[7:36], root_position=qpos[:3],
                          root_quaternion_wxyz=qpos[3:7], fallen=fallen)

    def send(self, command: WbcCommand) -> None:
        self.simulation.set_external_command(command.arm_targets, command.waist_ypr, command.nav, command.base_height)


class TruthCamera:
    """CameraSource for the mock estimator: the in-process sim renders no image, so frames only carry a timestamp."""

    def __init__(self, simulation: DecoupledSimulation):
        self.simulation = simulation

    def latest(self) -> CameraFrame:
        return CameraFrame(stamp=self.simulation.play_time, rgb=np.zeros((2, 2, 3), np.uint8),
                           K=np.eye(3), T_base_cam=np.eye(4))


def _descendants(model, root_body: int) -> set[int]:
    parents = model.body_parentid
    found = {root_body}
    changed = True
    while changed:
        changed = False
        for body in range(model.nbody):
            if body not in found and int(parents[body]) in found:
                found.add(body)
                changed = True
    return found


class ContactGraspMonitor:
    """held() is true while the target is pressed by both arms' tools/hands (normal force >= threshold)."""

    def __init__(self, simulation: DecoupledSimulation, target_id: str, threshold_n: float = 1.0):
        model = simulation.model
        self.simulation, self.threshold_n = simulation, threshold_n
        self.target_geoms = simulation.object_geoms[target_id]
        self.side_bodies = {side: _descendants(model, model.body(f"{side}_wrist_yaw_link").id) for side in ("left", "right")}
        self.forces = {"left": 0.0, "right": 0.0}

    def update(self) -> dict[str, float]:
        simulation = self.simulation
        forces = {"left": 0.0, "right": 0.0}
        with simulation.lock:
            wrench = np.zeros(6)
            for index in range(simulation.data.ncon):
                contact = simulation.data.contact[index]
                geoms = (contact.geom1, contact.geom2)
                hit = [g in self.target_geoms for g in geoms]
                if not any(hit):
                    continue
                other = int(simulation.model.geom_bodyid[geoms[1] if hit[0] else geoms[0]])
                for side, bodies in self.side_bodies.items():
                    if other in bodies:
                        mujoco.mj_contactForce(simulation.model, simulation.data, index, wrench)
                        forces[side] += abs(float(wrench[0]))
        self.forces = forces
        return forces

    def held(self) -> bool:
        return all(force >= self.threshold_n for force in self.update().values())


class ControllerSession:
    def __init__(self, simulation: DecoupledSimulation, bundle_dir: Path, stages: dict, options: dict | None = None,
                 autostart: bool = True):
        self.simulation = simulation
        self.bundle_dir = Path(bundle_dir)
        self.stages = _validated_stages(stages)
        self.options = {"proxy_grasp": False, "vision_visible": True, **(options or {})}
        self.error = ""
        self.closed = False
        names = {item["name"]: item for item in simulation.scene_objects}
        if self.stages["target_object"] not in names:
            raise ValueError(f"target_object {self.stages['target_object']!r} is not in the scene "
                             f"(objects: {', '.join(sorted(names))})")
        self.target = names[self.stages["target_object"]]
        self._build()
        simulation.set_control_mode("external")
        start = self.stages.get("robot_start_xy_yaw", [-1.0, 0.0, 0.0])
        simulation.set_start_pose(*start)
        self._next_tick = 0.0
        self._fall_reported = False
        self.thread = threading.Thread(target=self._loop, name="task-controller", daemon=True)
        if autostart:
            self.thread.start()

    # -- assembly ---------------------------------------------------------------------------------------

    def _target_pose_in_base(self) -> ObjectPose | None:
        if not self.options["vision_visible"]:
            return None
        simulation = self.simulation
        with simulation.lock:
            base_position, base_quaternion = simulation.data.qpos[:3].copy(), simulation.data.qpos[3:7].copy()
            body = simulation.object_bodies[self.target["id"]]
            position, quaternion = simulation.data.xpos[body].copy(), simulation.data.xquat[body].copy()
        base = Rotation.from_quat(base_quaternion[[1, 2, 3, 0]])
        local = base.inv() * Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        return ObjectPose(stamp=simulation.play_time, frame_id="base", position=base.inv().apply(position - base_position),
                          quaternion_xyzw=local.as_quat(), confidence=1.0, source="sim_truth")

    def _build(self) -> None:
        stages = self.stages
        entries = {MotionKind(name): MotionEntry(self.bundle_dir, float(stages[name]["start_s"]), float(stages[name]["end_s"]))
                   for name in STAGE_NAMES}
        root_position, root_wxyz = root_pose_at(self.bundle_dir, float(stages.get("authored_root_at_s", stages["ready"]["start_s"])))
        root = Rotation.from_quat([root_wxyz[1], root_wxyz[2], root_wxyz[3], root_wxyz[0]])
        authored_position = root.inv().apply(np.asarray(self.target["position"], float) - root_position)
        grasp = entries[MotionKind.GRASP]
        planner = FixedGraspPlanner(load_segment(grasp.bundle, MotionKind.GRASP, grasp.start_s, grasp.end_s),
                                    authored_position, tolerance_m=float(stages.get("grasp_tolerance_m", 0.08)))
        library = MotionLibrary(entries, planner)
        yaw = float(root.as_euler("xyz")[2])
        operator = AutoPositioner(target_xy_yaw=(float(root_position[0]), float(root_position[1]), yaw))
        vision = VisionService(TruthCamera(self.simulation),
                               MockPoseEstimator(truth=self._target_pose_in_base, position_sigma_m=0.003,
                                                 rotation_sigma_deg=1.0, dropout_probability=0.1,
                                                 outlier_probability=0.03, seed=0))
        self.contact = ContactGraspMonitor(self.simulation, self.target["id"])
        self.fsm = TaskFsm(vision, library, operator, self.contact, TaskFsmConfig(base_height=0.74))
        self.proxy = ProxyGraspMonitor(self.fsm)
        self.fsm.grasp = self.proxy if self.options["proxy_grasp"] else self.contact
        self.runner = TaskRunner(self.fsm, SimSessionLink(self.simulation), operator,
                                 clock=lambda: self.simulation.play_time, rate_hz=CONTROL_RATE_HZ)
        self.authored_position = authored_position

    # -- control ----------------------------------------------------------------------------------------

    def poll(self) -> bool:
        """One controller iteration: tick the FSM if a control period of sim time has elapsed. True if it ticked."""
        simulation = self.simulation
        # A fall stops the simulation, but the FSM must still hear about it once so it enters RECOVERY.
        unreported_fall = simulation.phase == "fallen" and not self._fall_reported
        if not (unreported_fall or (simulation.running and simulation.play_time >= self._next_tick)):
            return False
        self._fall_reported = self._fall_reported or unreported_fall
        try:
            self.runner.tick()
        except Exception as exc:                # a controller fault must stop the robot, not the server
            self.error = f"{type(exc).__name__}: {exc}"
            simulation.stop()
        self._next_tick = max(self._next_tick + 1.0 / CONTROL_RATE_HZ, simulation.play_time)
        return True

    def _loop(self) -> None:
        while not self.closed:
            if not self.poll():
                time.sleep(0.002)

    def start(self) -> None:
        self.simulation.play()

    def stop(self) -> None:
        self.simulation.stop()

    def reset(self) -> None:
        self.simulation.reset()
        self.simulation.set_control_mode("external")
        self.error = ""
        self._next_tick = 0.0
        self._fall_reported = False
        self._build()

    def operator_reset(self) -> None:
        """The operator's acknowledgement after a stop (RECOVERY -> INITIALIZE)."""
        self.fsm.operator.request_reset()

    def set_options(self, options: dict) -> None:
        for key in ("proxy_grasp", "vision_visible"):
            if key in options:
                self.options[key] = bool(options[key])
        self.fsm.grasp = self.proxy if self.options["proxy_grasp"] else self.contact

    def close(self) -> None:
        self.closed = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.simulation.close()

    # -- view -------------------------------------------------------------------------------------------

    def snapshot(self) -> dict:
        snapshot = self.simulation.snapshot()
        task = self.runner.snapshot()
        forces = self.contact.update()
        task.update({"held": all(force >= self.contact.threshold_n for force in forces.values()),
                     "grip_forces_n": {k: round(v, 2) for k, v in forces.items()},
                     "options": dict(self.options), "target_object": self.target["name"],
                     "authored_target_base": np.round(self.authored_position, 3).tolist(),
                     "controller_error": self.error})
        snapshot["task"] = task
        return snapshot


MAX_SESSIONS = 2     # each session runs a 200 Hz simulation thread; a forgotten one would starve the others


class ControllerSessions:
    def __init__(self):
        self.robots: dict[tuple, Robot] = {}
        self.sessions: dict[str, ControllerSession] = {}
        self.lock = threading.Lock()

    def _robot_for(self, project: dict) -> Robot:
        """The robot a project was authored with: its mounted tool pair if it records one, else the server default."""
        tools = tuple(project["mounted_tools"]) if project.get("mounted_tools") else None
        key = ("g1-tools", tools) if tools else ("default", None)
        with self.lock:
            if key not in self.robots:
                self.robots[key] = Robot("g1-tools", tools) if tools else Robot()
            return self.robots[key]

    def create(self, project: dict, bundle_dir: Path, stages: dict, options: dict | None = None) -> tuple[str, ControllerSession]:
        if not verify_assets()["available"]:
            raise ValueError("Decoupled WBC assets are not installed")
        robot = self._robot_for(project)
        self._evict_oldest(MAX_SESSIONS - 1)
        simulation = DecoupledSimulation(robot, project)
        try:
            session = ControllerSession(simulation, bundle_dir, stages, options)
        except Exception:
            simulation.close()
            raise
        identifier = uuid.uuid4().hex
        with self.lock:
            self.sessions[identifier] = session
        return identifier, session

    def _evict_oldest(self, keep: int) -> None:
        """Close the oldest sessions until at most `keep` remain (dicts keep creation order)."""
        with self.lock:
            stale = [self.sessions.pop(key) for key in list(self.sessions)[:max(0, len(self.sessions) - keep)]]
        for session in stale:
            session.close()

    def get(self, identifier: str) -> ControllerSession:
        with self.lock:
            session = self.sessions.get(identifier)
        if session is None:
            raise ValueError("Controller session was not found")
        return session

    def delete(self, identifier: str) -> None:
        with self.lock:
            session = self.sessions.pop(identifier, None)
        if session:
            session.close()

    def close(self) -> None:
        with self.lock:
            sessions, self.sessions = list(self.sessions.values()), {}
        for session in sessions:
            session.close()


controller_sessions = ControllerSessions()
