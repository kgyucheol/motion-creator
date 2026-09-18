"""Headless GR00T decoupled-WBC simulation used by the browser workbench."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import atexit
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .grasp import fit_two_hand_grasp, object_signature
from .grip_geometry import grip_pad_center, grip_pad_half_size, grip_pad_quaternion_wxyz
from .hand_collision import physical_hand_geom_names
from .motion import compile_motion, project_scene_objects, validate_project
from .robot import ROOT, Robot


ASSET_ROOT = ROOT / "external/task-models/decoupled-wbc"
ASSET_MANIFEST = ROOT / "integrations/decoupled-wbc-assets.json"
PARAMETERS_PATH = ROOT / "integrations/decoupled-wbc-parameters.json"


def _parameters() -> dict:
    return json.loads(PARAMETERS_PATH.read_text(encoding="utf-8"))


def verify_assets() -> dict:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    missing, invalid = [], []
    for entry in manifest["assets"]:
        path = ASSET_ROOT / entry["file"]
        if not path.is_file():
            missing.append(entry["file"])
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            invalid.append(entry["file"])
    if not (ASSET_ROOT / "meshes").is_dir():
        missing.append("meshes/")
    available = not missing and not invalid
    return {
        "available": available,
        "missing": missing,
        "invalid": invalid,
        "source_revision": manifest["source"]["revision"],
        "setup_command": "scripts/setup-decoupled-wbc.sh ../GR00T-WholeBodyControl",
    }


def _require_assets() -> None:
    result = verify_assets()
    if not result["available"]:
        details = ", ".join([*result["missing"], *result["invalid"]]) or "unknown"
        raise ValueError(f"Decoupled WBC assets are not installed or invalid: {details}. Run {result['setup_command']}")


def _numbers(values) -> str:
    return " ".join(str(float(value)) for value in values)


def _grounded_position(item: dict) -> np.ndarray:
    position = np.asarray(item["position"], dtype=float).copy()
    size = np.asarray(item["size"], dtype=float)
    if item["shape"] == "sphere":
        extent = size[0] / 2
    else:
        x, y, z, w = np.asarray(item["quaternion_xyzw"], dtype=float)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)
        extent = (size[0] / 2 * np.hypot(r20, r21) + size[2] / 2 * abs(r22)
                  if item["shape"] == "cylinder" else
                  abs(r20) * size[0] / 2 + abs(r21) * size[1] / 2 + abs(r22) * size[2] / 2)
    position[2] = max(position[2], extent)
    return position


def build_environment_model(project: dict) -> mujoco.MjModel:
    """Load the WBC MJCF and add fully dynamic authored scene objects."""
    root = ET.parse(ASSET_ROOT / "g1_gear_wbc.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSET_ROOT / "meshes"))
    objects = project_scene_objects(project)
    if objects:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for side in ("left", "right"):
            body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
            physical_hand_geom_names(root, side, dex3=True)
            ET.SubElement(body, "geom", name=f"{side}_wbc_grip", type="box",
                          pos=_numbers(grip_pad_center(side)), quat=_numbers(grip_pad_quaternion_wxyz(side)),
                          size=_numbers(grip_pad_half_size()),
                          contype="0", conaffinity="0", group="3", density="0",
                          rgba=".15 .9 .72 .45" if side == "left" else "1 .62 .25 .45")
        world = root.find("worldbody")
        object_geoms = []
        for index, item in enumerate(objects):
            body = ET.SubElement(world, "body", name=f"wbc_object_{index}",
                                 pos=_numbers(_grounded_position(item)),
                                 quat=_numbers(np.asarray(item["quaternion_xyzw"])[[3, 0, 1, 2]]))
            ET.SubElement(body, "freejoint", name=f"wbc_object_joint_{index}")
            size = np.asarray(item["size"], dtype=float)
            mj_size = (size / 2 if item["shape"] == "box" else [size[0] / 2]
                       if item["shape"] == "sphere" else [size[0] / 2, size[2] / 2])
            geom_name = f"wbc_object_geom_{index}"
            ET.SubElement(body, "geom", name=geom_name, type=item["shape"], size=_numbers(mj_size),
                          mass=str(float(item["mass_kg"])),
                          friction=_numbers([item["friction"], .005, .0001]),
                          contype="1", conaffinity="1")
            object_geoms.append(geom_name)
            for side in ("left", "right"):
                for hand_index in range(6):
                    ET.SubElement(contact, "pair", geom1=f"{side}_physical_hand_{hand_index}", geom2=geom_name,
                                  condim="3",
                                  friction=_numbers([item["friction"], item["friction"], 0, 0, 0]),
                                  solref=".01 1", solimp=".95 .99 .001")
            ET.SubElement(contact, "pair", geom1="floor", geom2=geom_name, condim="3",
                          friction=_numbers([item["friction"], item["friction"], 0, 0, 0]))
        for first, geom1 in enumerate(object_geoms):
            for geom2 in object_geoms[first + 1:]:
                ET.SubElement(contact, "pair", geom1=geom1, geom2=geom2, condim="3")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nq != 36 + 7 * len(objects) or model.nv != 35 + 6 * len(objects) or model.nu != 29:
        raise ValueError("Unexpected decoupled-WBC environment model")
    return model


def prepare_grasp_control(robot: Robot, project: dict) -> dict | None:
    """Resolve the first authored grasp into an arm closure offset and timeline time."""
    objects = project_scene_objects(project)
    elapsed = 0.
    for index, frame in enumerate(project["keyframes"]):
        if index:
            elapsed += float(frame["duration"])
        event = frame.get("grasp")
        if event is None:
            continue
        item = next(value for value in objects if value["id"] == event["object_id"])
        authored = robot.validate_q(frame["qpos"])
        contact = authored
        closure = event.get("closure_qpos")
        if closure is None or event.get("object_signature") != object_signature(item):
            fitted = fit_two_hand_grasp(robot, frame["qpos"], frame.get("pins", []), item, event)
            contact = robot.validate_q(fitted["state"]["qpos"])
            closure = fitted["grasp"]["closure_qpos"]
        closed = robot.validate_q(closure)
        return {
            "object_id": item["id"], "start_time": elapsed,
            "closure_seconds": float(event["closure_seconds"]),
            "contact_offset": contact[22:36] - authored[22:36],
            "arm_offset": closed[22:36] - contact[22:36],
            "target_force_n": float(event["target_force_n"]),
            "max_force_n": float(event["max_force_n"]),
        }
    return None


def quat_rotate_inverse(wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).inv().apply(vector)


def authored_waist_to_torso_rpy(waist_yaw_roll_pitch: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Convert the editor's yaw/roll/pitch waist joints into policy XYZ torso RPY."""
    yaw, roll, pitch = np.asarray(waist_yaw_roll_pitch, dtype=float)
    rpy = Rotation.from_euler("zxy", [yaw, roll, pitch]).as_euler("xyz")
    return np.clip(rpy, -limits, limits).astype(np.float32)


@dataclass
class CommandState:
    nav: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    height: float = 0.74

    def apply_key(self, key: str, parameters: dict) -> bool:
        key = key.lower()
        changed = True
        if key == "w": self.nav[0] += .1
        elif key == "s": self.nav[0] -= .1
        elif key == "a": self.nav[1] += .1
        elif key == "d": self.nav[1] -= .1
        elif key == "q": self.nav[2] += .1
        elif key == "e": self.nav[2] -= .1
        elif key == "1": self.height += .05
        elif key == "2": self.height -= .05
        elif key == "z":
            self.nav[:] = 0
            self.height = float(parameters["initial_height"])
        else:
            changed = False
        limits = parameters["command_limits"]
        self.nav[:2] = np.clip(self.nav[:2], -limits["linear_velocity"], limits["linear_velocity"])
        self.nav[2] = np.clip(self.nav[2], -limits["angular_velocity"], limits["angular_velocity"])
        self.height = float(np.clip(self.height, limits["minimum_height"], limits["maximum_height"]))
        return changed


def build_observation(data: mujoco.MjData, action: np.ndarray, command: CommandState,
                      torso_rpy: np.ndarray, parameters: dict) -> np.ndarray:
    n_joints = 29
    default = np.zeros(n_joints, dtype=np.float32)
    default[:15] = parameters["default_angles"]
    command_vector = np.zeros(7, dtype=np.float32)
    command_vector[:3] = command.nav * np.asarray(parameters["command_scale"], dtype=np.float32)
    command_vector[3] = command.height
    command_vector[4:7] = torso_rpy
    observation = np.zeros(86, dtype=np.float32)
    observation[:7] = command_vector
    observation[7:10] = data.qvel[3:6] * parameters["angular_velocity_scale"]
    observation[10:13] = quat_rotate_inverse(data.qpos[3:7], np.array([0., 0., -1.]))
    observation[13:42] = (data.qpos[7:36] - default) * parameters["dof_position_scale"]
    observation[42:71] = data.qvel[6:35] * parameters["dof_velocity_scale"]
    observation[71:86] = action
    return observation


class LowerBodyPolicy:
    def __init__(self, parameters: dict):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("onnxruntime is required; run scripts/setup-task-cpu.sh") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.balance = ort.InferenceSession(str(ASSET_ROOT / "GR00T-WholeBodyControl-Balance.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.walk = ort.InferenceSession(str(ASSET_ROOT / "GR00T-WholeBodyControl-Walk.onnx"), options,
                                         providers=["CPUExecutionProvider"])
        self.parameters = parameters
        self.history: deque[np.ndarray] = deque(maxlen=parameters["observation_history"])
        self.reset()

    def reset(self) -> None:
        self.action = np.zeros(15, dtype=np.float32)
        self.history.clear()
        for _ in range(self.parameters["observation_history"]):
            self.history.append(np.zeros(86, dtype=np.float32))

    def infer(self, observation: np.ndarray, moving: bool) -> np.ndarray:
        self.history.append(observation)
        stacked = np.concatenate(tuple(self.history), dtype=np.float32)[None, :]
        session = self.walk if moving else self.balance
        name = session.get_inputs()[0].name
        self.action = np.asarray(session.run(None, {name: stacked})[0], dtype=np.float32).reshape(15)
        return np.asarray(self.parameters["default_angles"], dtype=np.float32) + self.action * self.parameters["action_scale"]


class DecoupledSimulation:
    def __init__(self, robot: Robot, project: dict, autostart: bool = True):
        _require_assets()
        validate_project(robot, project)
        self.robot = robot
        self.project = project
        self.parameters = _parameters()
        self.reference = compile_motion(robot, project, fps=50)
        self.duration = float(self.reference["time"][-1])
        self.scene_objects = copy.deepcopy(project_scene_objects(project))
        self.grasp_control = prepare_grasp_control(robot, project)
        self.model = build_environment_model(project)
        self.model.opt.timestep = self.parameters["simulation_dt"]
        self.data = mujoco.MjData(self.model)
        self.object_bodies = {item["id"]: self.model.body(f"wbc_object_{index}").id
                              for index, item in enumerate(self.scene_objects)}
        self.object_geoms = {item["id"]: self.model.geom(f"wbc_object_geom_{index}").id
                             for index, item in enumerate(self.scene_objects)}
        self.hand_geoms = {
            side: {self.model.geom(f"{side}_physical_hand_{index}").id for index in range(6)}
            for side in ("left", "right")
        } if self.grasp_control else {}
        self.policy = LowerBodyPolicy(self.parameters)
        self.command = CommandState(height=self.parameters["initial_height"])
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.running = False
        self.closed = False
        self.phase = "ready"
        self.error = ""
        self.revision = 0
        self.step_count = 0
        self.motion_time = 0.
        self.play_time = 0.
        self.torso_rpy = np.zeros(3, dtype=np.float32)
        self.recording: list[dict] = []
        self._last_record_step = -1
        self.reset()
        self.thread = threading.Thread(target=self._loop, name="decoupled-wbc", daemon=True)
        if autostart:
            self.thread.start()

    def _reference_at(self, seconds: float) -> np.ndarray:
        index = min(int(round(seconds * 50)), len(self.reference["qpos"]) - 1)
        return self.reference["qpos"][index]

    def _object_states(self) -> dict:
        return {identifier: {
            "position": self.data.xpos[body].tolist(),
            "quaternion_xyzw": self.data.xquat[body][[1, 2, 3, 0]].tolist(),
        } for identifier, body in self.object_bodies.items()}

    def _grasp_contact_forces(self) -> dict[str, float]:
        if not self.grasp_control:
            return {}
        target = self.object_geoms[self.grasp_control["object_id"]]
        forces = {side: 0. for side in ("left", "right")}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if target not in pair:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            for side, geoms in self.hand_geoms.items():
                if pair & geoms:
                    forces[side] += max(0., float(wrench[0]))
        return forces

    def reset(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[7:22] = np.asarray(self.parameters["default_angles"])
            # Start from the authored upper-body pose so the real hand meshes do
            # not sweep through nearby objects while the lower-body policy settles.
            self.data.qpos[22:36] = self._reference_at(0.)[22:36]
            mujoco.mj_forward(self.model, self.data)
            self.policy.reset()
            self.command = CommandState(height=self.parameters["initial_height"])
            self.running = False
            self.phase = "ready"
            self.error = ""
            self.step_count = 0
            self.motion_time = 0.
            self.play_time = 0.
            self.torso_rpy[:] = 0
            self.recording.clear()
            self._last_record_step = -1
            self._changed()

    def play(self) -> None:
        with self.lock:
            if self.phase == "fallen":
                self.error = "낙상 후에는 Reset이 필요합니다."
                self._changed()
                return
            self.running = True
            self.phase = "settling" if self.play_time < self.parameters["upper_body_settle_seconds"] else "playing"
            self._changed()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.command.nav[:] = 0
            if self.phase != "fallen": self.phase = "paused"
            self._changed()

    def apply_key(self, key: str) -> None:
        with self.lock:
            if self.command.apply_key(key, self.parameters): self._changed()

    def _changed(self) -> None:
        self.revision += 1
        self.condition.notify_all()

    def _arm_target(self) -> np.ndarray:
        target = self._reference_at(self.motion_time)[22:36]
        settle = self.parameters["upper_body_settle_seconds"]
        if self.play_time < settle:
            return self._reference_at(0.)[22:36]
        if self.grasp_control:
            start = self.grasp_control["start_time"]
            contact_progress = np.clip((self.motion_time - (start - .5)) / .5, 0., 1.)
            target = target + contact_progress * self.grasp_control["contact_offset"]
            if self.motion_time >= start:
                progress = np.clip((self.motion_time - start) / self.grasp_control["closure_seconds"], 0., 1.)
                target = target + progress * self.grasp_control["arm_offset"]
        return target

    @staticmethod
    def _pd(target: np.ndarray, q: np.ndarray, kp: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
        return (target - q) * kp - dq * kd

    def step(self) -> None:
        with self.lock:
            if not self.running or self.closed: return
            p = self.parameters
            reference = self._reference_at(self.motion_time)
            limits = np.asarray(p["command_limits"]["torso_rpy"], dtype=np.float32)
            self.torso_rpy = authored_waist_to_torso_rpy(reference[19:22], limits)
            lower_target = np.asarray(p["default_angles"], dtype=np.float32) + self.policy.action * p["action_scale"]
            self.data.ctrl[:15] = self._pd(lower_target, self.data.qpos[7:22], np.asarray(p["lower_kp"]),
                                           self.data.qvel[6:21], np.asarray(p["lower_kd"]))
            self.data.ctrl[15:29] = self._pd(self._arm_target(), self.data.qpos[22:36], np.asarray(p["arm_kp"]),
                                             self.data.qvel[21:35], np.asarray(p["arm_kd"]))
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            self.play_time += p["simulation_dt"]
            if self.play_time >= p["upper_body_settle_seconds"]:
                self.phase = "playing"
                self.motion_time = min(self.duration, self.motion_time + p["simulation_dt"])
            if self.step_count % p["control_decimation"] == 0:
                observation = build_observation(self.data, self.policy.action, self.command, self.torso_rpy, p)
                self.policy.infer(observation, np.linalg.norm(self.command.nav) > .05)
            # Record at 25 Hz (every eight 200 Hz physics steps).
            if self.step_count % 8 == 0 and self._last_record_step != self.step_count:
                if len(self.recording) < int(p["maximum_recording_seconds"] * 25):
                    self.recording.append({"time": self.play_time, "qpos": self.data.qpos[:36].copy(),
                                           "nav": self.command.nav.copy(), "height": self.command.height,
                                           "torso_rpy": self.torso_rpy.copy(), "upper_time": self.motion_time,
                                           "object_states": self._object_states()})
                    self._last_record_step = self.step_count
            tilt = np.linalg.norm(Rotation.from_quat(self.data.qpos[3:7][[1, 2, 3, 0]]).as_euler("xyz")[:2])
            if self.data.qpos[2] < .25 or tilt > np.deg2rad(45):
                self.running = False
                self.command.nav[:] = 0
                self.phase = "fallen"
                self.error = "낙상이 감지되었습니다. Reset 후 다시 시작하세요."
            if self.step_count % p["stream_decimation"] == 0: self._changed()

    def _loop(self) -> None:
        dt = self.parameters["simulation_dt"]
        deadline = time.perf_counter()
        while not self.closed:
            with self.lock: active = self.running
            if active:
                self.step()
                deadline += dt
                time.sleep(max(0., deadline - time.perf_counter()))
            else:
                deadline = time.perf_counter()
                time.sleep(.02)

    def snapshot(self) -> dict:
        with self.lock:
            qpos = self.data.qpos[:36].copy()
            grasp = None
            if self.grasp_control:
                start = self.grasp_control["start_time"]
                grasp = {key: value for key, value in self.grasp_control.items() if not key.endswith("_offset")}
                grasp["active"] = self.motion_time >= start
                grasp["closure"] = float(np.clip((self.motion_time - start)
                                                  / self.grasp_control["closure_seconds"], 0., 1.))
                forces = self._grasp_contact_forces()
                grasp["normal_force_n"] = forces
                grasp["bilateral_contact"] = all(force > 0 for force in forces.values())
            return {
                "revision": self.revision, "phase": self.phase, "playing": self.running,
                "error": self.error, "policy": "walk" if np.linalg.norm(self.command.nav) > .05 else "balance",
                "nav": self.command.nav.tolist(), "height": self.command.height,
                "torso_rpy": self.torso_rpy.tolist(), "upper_time": self.motion_time,
                "upper_duration": self.duration, "recording_frames": len(self.recording),
                "state": self.robot.state(qpos), "scene_objects": self.scene_objects,
                "object_states": self._object_states(), "grasp": grasp,
            }

    def recording_copy(self) -> list[dict]:
        with self.lock: return copy.deepcopy(self.recording)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.running = False
            self._changed()
        if self.thread.is_alive(): self.thread.join(timeout=1.)


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_recording_bundle(frames: list[dict], source_name: str, root: Path = ROOT / "motions", *,
                          scene_objects: list[dict] | None = None) -> dict:
    if not frames:
        raise ValueError("There is no recorded playback to save")
    identifier = uuid.uuid4().hex
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder = root / f"Decoupled_WBC_{stamp}_{identifier[:8]}"
    folder.mkdir(parents=True)
    times = np.asarray([row["time"] for row in frames])
    qpos = np.stack([row["qpos"] for row in frames])
    nav = np.stack([row["nav"] for row in frames])
    height = np.asarray([row["height"] for row in frames])
    torso_rpy = np.stack([row["torso_rpy"] for row in frames])
    upper_time = np.asarray([row["upper_time"] for row in frames])
    objects = copy.deepcopy(scene_objects or [])
    archive = {"time": times, "qpos": qpos, "navigate_command": nav,
               "base_height_command": height, "torso_rpy_command": torso_rpy,
               "upper_motion_time": upper_time, "fps": np.array(25)}
    if objects:
        identifiers = [item["id"] for item in objects]
        archive["object_ids"] = np.asarray(identifiers)
        archive["object_position"] = np.asarray([[row["object_states"][key]["position"] for key in identifiers]
                                                  for row in frames])
        archive["object_quaternion_xyzw"] = np.asarray([
            [row["object_states"][key]["quaternion_xyzw"] for key in identifiers] for row in frames])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **archive)
    _atomic_bytes(folder / "motion.npz", buffer.getvalue())
    qbuffer = io.StringIO(); np.savetxt(qbuffer, qpos, delimiter=",")
    _atomic_bytes(folder / "motion.csv", qbuffer.getvalue().encode())
    cbuffer = io.StringIO()
    writer = csv.writer(cbuffer)
    writer.writerow(["time", "vx", "vy", "yaw_rate", "height", "torso_roll", "torso_pitch", "torso_yaw", "upper_motion_time"])
    writer.writerows(np.column_stack([times, nav, height, torso_rpy, upper_time]))
    _atomic_bytes(folder / "commands.csv", cbuffer.getvalue().encode())
    environment = {
        "format": "motioncreator.environment.v1", "coordinate_system": "right-handed, +X forward, +Y left, +Z up",
        "physics": {"gravity_m_s2": [0., 0., -9.81]}, "scene_objects": objects,
    }
    _atomic_bytes(folder / "environment.json", json.dumps(environment, ensure_ascii=False, indent=2).encode())
    metadata = {
        "format": "motioncreator.decoupled-wbc.recording.v1", "recording_id": identifier,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_project": source_name,
        "fps": 25, "samples": len(frames), "joint_order": "G1 29-DoF; root qpos wxyz",
        "upper_body": "authored motion arm targets", "lower_body": "GR00T decoupled-WBC policy",
        "environment_file": "environment.json", "object_count": len(objects),
        "policy_revision": json.loads(ASSET_MANIFEST.read_text())["source"]["revision"],
    }
    _atomic_bytes(folder / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode())
    return {"folder": folder.name, "files": [f"{folder.name}/{name}" for name in
                                               ("motion.npz", "motion.csv", "commands.csv", "environment.json", "metadata.json")]}


class DecoupledSessions:
    def __init__(self):
        # Build the visualization robot lazily. Constructing a second MuJoCo model
        # while FastAPI creates its worker portal can stall some OpenMP runtimes.
        self.robot: Robot | None = None
        self.sessions: dict[str, DecoupledSimulation] = {}
        self.lock = threading.Lock()

    def create(self, project: dict) -> tuple[str, DecoupledSimulation]:
        with self.lock:
            if self.robot is None:
                self.robot = Robot()
            robot = self.robot
        session = DecoupledSimulation(robot, project)
        identifier = uuid.uuid4().hex
        with self.lock: self.sessions[identifier] = session
        return identifier, session

    def get(self, identifier: str) -> DecoupledSimulation:
        with self.lock: session = self.sessions.get(identifier)
        if session is None: raise ValueError("Decoupled WBC session was not found")
        return session

    def delete(self, identifier: str) -> None:
        with self.lock: session = self.sessions.pop(identifier, None)
        if session: session.close()

    def close(self) -> None:
        with self.lock:
            sessions, self.sessions = list(self.sessions.values()), {}
        for session in sessions: session.close()


sessions = DecoupledSessions()
atexit.register(sessions.close)
