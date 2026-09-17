"""Headless GR00T decoupled-WBC simulation used by the browser workbench."""
from __future__ import annotations

from collections import deque
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

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import compile_motion, validate_project
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
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "g1_gear_wbc.xml"))
        self.model.opt.timestep = self.parameters["simulation_dt"]
        self.data = mujoco.MjData(self.model)
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

    def reset(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[7:22] = np.asarray(self.parameters["default_angles"])
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
            blend = np.clip(self.play_time / settle, 0., 1.)
            start = self._reference_at(0)[22:36]
            return (1 - blend) * self.model.qpos0[22:36] + blend * start
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
                    self.recording.append({"time": self.play_time, "qpos": self.data.qpos.copy(),
                                           "nav": self.command.nav.copy(), "height": self.command.height,
                                           "torso_rpy": self.torso_rpy.copy(), "upper_time": self.motion_time})
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
            qpos = self.data.qpos.copy()
            return {
                "revision": self.revision, "phase": self.phase, "playing": self.running,
                "error": self.error, "policy": "walk" if np.linalg.norm(self.command.nav) > .05 else "balance",
                "nav": self.command.nav.tolist(), "height": self.command.height,
                "torso_rpy": self.torso_rpy.tolist(), "upper_time": self.motion_time,
                "upper_duration": self.duration, "recording_frames": len(self.recording),
                "state": self.robot.state(qpos),
            }

    def recording_copy(self) -> list[dict]:
        with self.lock: return [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in row.items()} for row in self.recording]

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


def save_recording_bundle(frames: list[dict], source_name: str, root: Path = ROOT / "motions") -> dict:
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
    buffer = io.BytesIO()
    np.savez_compressed(buffer, time=times, qpos=qpos, navigate_command=nav,
                        base_height_command=height, torso_rpy_command=torso_rpy,
                        upper_motion_time=upper_time, fps=np.array(25))
    _atomic_bytes(folder / "motion.npz", buffer.getvalue())
    qbuffer = io.StringIO(); np.savetxt(qbuffer, qpos, delimiter=",")
    _atomic_bytes(folder / "motion.csv", qbuffer.getvalue().encode())
    cbuffer = io.StringIO()
    writer = csv.writer(cbuffer)
    writer.writerow(["time", "vx", "vy", "yaw_rate", "height", "torso_roll", "torso_pitch", "torso_yaw", "upper_motion_time"])
    writer.writerows(np.column_stack([times, nav, height, torso_rpy, upper_time]))
    _atomic_bytes(folder / "commands.csv", cbuffer.getvalue().encode())
    metadata = {
        "format": "motioncreator.decoupled-wbc.recording.v1", "recording_id": identifier,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_project": source_name,
        "fps": 25, "samples": len(frames), "joint_order": "G1 29-DoF; root qpos wxyz",
        "upper_body": "authored motion arm targets", "lower_body": "GR00T decoupled-WBC policy",
        "policy_revision": json.loads(ASSET_MANIFEST.read_text())["source"]["revision"],
    }
    _atomic_bytes(folder / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode())
    return {"folder": folder.name, "files": [f"{folder.name}/{name}" for name in ("motion.npz", "motion.csv", "commands.csv", "metadata.json")]}


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
