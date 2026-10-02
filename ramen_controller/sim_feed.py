"""Sim-side ground truth and the camera/truth feed between the simulator process and the FSM process.

`TruthExtractor` turns MuJoCo state into what the FSM side needs: object poses in the robot base (pelvis)
frame, the head camera pose T_base_cam in OpenCV optical convention, and whether the target object is gripped
by both tools. `pack_message`/`unpack_message` define the wire format; `SimFeed` is the subscriber that
exposes it as a CameraSource, a truth provider for MockPoseEstimator, and a GraspMonitor.

On the real robot a camera driver replaces the frames and no truth exists; only the sim uses this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import ObjectPose
from .vision import BASE_FRAME, CameraFrame

DEFAULT_PORT = 5570
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])        # MuJoCo camera: x right, y up, z back  ->  OpenCV: x right, y down, z forward


def _transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, position
    return T


def _invert(T: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inverse


class TruthExtractor:
    """Ground truth from a MuJoCo model built by `scene.build_scene` (uses its scene_objects.json sidecar)."""

    def __init__(self, model, sidecar_path, tool_bodies: dict[str, str], camera_name: str = "head",
                 root_body: str = "pelvis", min_grip_force_n: float = 1.0):
        import mujoco
        self._mj = mujoco
        self.model, self.min_grip_force_n = model, min_grip_force_n
        sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        self.camera = sidecar["camera"]
        self.objects = sidecar["objects"]
        self.root = model.body(root_body).id
        self.camera_id = model.camera(camera_name).id
        self._body_ids = {o["name"]: model.body(o["body"]).id for o in self.objects}
        target = [o for o in self.objects if o["is_target"]]
        self.target_name = target[0]["name"] if target else None
        self._target_geoms = {model.geom(g).id for o in target for g in o["geoms"]}
        self._tool_subtrees = {side: self._subtree(model.body(name).id) for side, name in tool_bodies.items()}

    def _subtree(self, root_id: int) -> set[int]:
        parents = self.model.body_parentid
        return {b for b in range(self.model.nbody)
                if any(a == root_id for a in self._ancestors(b, parents))}

    @staticmethod
    def _ancestors(body: int, parents) -> list[int]:
        chain = [body]
        while body != 0:
            body = int(parents[body])
            chain.append(body)
        return chain

    def base_pose_world(self, data) -> np.ndarray:
        return _transform(data.xmat[self.root].reshape(3, 3), data.xpos[self.root])

    def camera_pose_in_base(self, data) -> np.ndarray:
        world_cam = _transform(data.cam_xmat[self.camera_id].reshape(3, 3) @ _MJ_TO_CV, data.cam_xpos[self.camera_id])
        return _invert(self.base_pose_world(data)) @ world_cam

    def object_pose_in_base(self, data, name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self._body_ids[name]
        T = _invert(self.base_pose_world(data)) @ _transform(data.xmat[body].reshape(3, 3), data.xpos[body])
        return T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()

    def in_view(self, data, name: str) -> bool:
        """Target centre in front of the camera and inside the image (no occlusion test)."""
        T_base_cam = self.camera_pose_in_base(data)
        position, _ = self.object_pose_in_base(data, name)
        in_camera = _invert(T_base_cam) @ np.append(position, 1.0)
        if in_camera[2] <= 0.05:
            return False
        K = np.asarray(self.camera["K"])
        u, v = (K @ in_camera[:3])[:2] / in_camera[2]
        return 0 <= u < self.camera["width"] and 0 <= v < self.camera["height"]

    def grip_forces(self, data) -> dict[str, float]:
        """Summed normal force (N) between the target object and each tool."""
        forces = {side: 0.0 for side in self._tool_subtrees}
        wrench = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            geoms = (contact.geom1, contact.geom2)
            hit = [g in self._target_geoms for g in geoms]
            if not any(hit):
                continue
            other_body = int(self.model.geom_bodyid[geoms[1] if hit[0] else geoms[0]])
            for side, subtree in self._tool_subtrees.items():
                if other_body in subtree:
                    self._mj.mj_contactForce(self.model, data, i, wrench)
                    forces[side] += abs(float(wrench[0]))
        return forces

    def truth(self, data) -> dict:
        forces = self.grip_forces(data)
        message = {"sim_time": float(data.time), "grip_forces_n": forces,
                   "held": bool(forces and all(f >= self.min_grip_force_n for f in forces.values())),
                   "objects": {}, "target": self.target_name, "target_in_view": None}
        for obj in self.objects:
            position, quat = self.object_pose_in_base(data, obj["name"])
            message["objects"][obj["name"]] = {"position": position, "quaternion_xyzw": quat}
        if self.target_name:
            message["target_in_view"] = self.in_view(data, self.target_name)
        return message


def pack_message(stamp: float, rgb: np.ndarray, depth_m: np.ndarray, K, T_base_cam, truth: dict) -> bytes:
    """Depth is stored as uint16 millimetres (depth_scale 0.001), like a real D435."""
    import msgpack
    import msgpack_numpy
    depth_mm = np.clip(np.nan_to_num(depth_m, nan=0.0, posinf=0.0) * 1000.0, 0, 65535).astype(np.uint16)
    return msgpack.packb({"v": 1, "stamp": float(stamp), "rgb": np.ascontiguousarray(rgb, np.uint8),
                          "depth_mm": depth_mm, "K": np.asarray(K, float), "T_base_cam": np.asarray(T_base_cam, float),
                          "truth": truth}, default=msgpack_numpy.encode)


def unpack_message(payload: bytes) -> dict:
    import msgpack
    import msgpack_numpy
    message = msgpack.unpackb(payload, object_hook=msgpack_numpy.decode)
    if message.get("v") != 1:
        raise ValueError(f"unsupported feed version {message.get('v')}")
    return message


class SimFeed:
    """Subscriber for the simulator's feed. Keeps only the latest message."""

    def __init__(self, endpoint: str = f"tcp://127.0.0.1:{DEFAULT_PORT}", require_depth: bool = False):
        import zmq
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.connect(endpoint)
        self._zmq = zmq
        self._message: dict | None = None
        self.require_depth = require_depth

    def poll(self) -> dict | None:
        try:
            while True:
                self._message = unpack_message(self._socket.recv(self._zmq.NOBLOCK))
        except self._zmq.Again:
            return self._message

    # CameraSource
    def latest(self) -> CameraFrame | None:
        message = self.poll()
        if message is None:
            return None
        depth = message["depth_mm"].astype(np.float32) / 1000.0 if self.require_depth else None
        return CameraFrame(stamp=message["stamp"], rgb=message["rgb"], K=message["K"],
                           T_base_cam=message["T_base_cam"], depth=depth)

    # truth provider for MockPoseEstimator (None when the target is not in view)
    def target_pose(self) -> ObjectPose | None:
        message = self.poll()
        truth = None if message is None else message["truth"]
        if truth is None or not truth.get("target") or not truth.get("target_in_view"):
            return None
        pose = truth["objects"][truth["target"]]
        return ObjectPose(stamp=message["stamp"], frame_id=BASE_FRAME, position=pose["position"],
                          quaternion_xyzw=pose["quaternion_xyzw"], confidence=1.0, source="sim_truth")

    # GraspMonitor
    def held(self) -> bool:
        message = self.poll()
        return bool(message and message["truth"]["held"])
