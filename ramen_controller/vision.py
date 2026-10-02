"""Vision boundary: a replaceable pose-estimation backend behind a stable contract.

Any backend (FoundationPose-Shap, an RGB-only estimator, a mock) implements `PoseEstimator`: one
`CameraFrame` in, 6D `ObjectPose` detections out. `VisionService` adapts that to the FSM's `VisionPort`:
it pulls frames, runs the estimator, and expresses every pose in the robot base frame. Swapping the
backend never touches the FSM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import ObjectPose

BASE_FRAME = "base"
CAMERA_FRAME = "camera"        # OpenCV optical frame: x right, y down, z forward


@dataclass(frozen=True)
class CameraFrame:
    """One camera capture. `depth` (metres, HxW) is None for RGB-only cameras."""
    stamp: float
    rgb: np.ndarray                          # (H, W, 3) uint8
    K: np.ndarray                            # (3, 3) intrinsics
    T_base_cam: np.ndarray                   # (4, 4) base <- camera optical frame, at capture time
    depth: np.ndarray | None = None

    def __post_init__(self):
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {self.rgb.shape}")
        if np.asarray(self.K).shape != (3, 3) or np.asarray(self.T_base_cam).shape != (4, 4):
            raise ValueError("K must be (3, 3) and T_base_cam (4, 4)")
        if self.depth is not None and self.depth.shape != self.rgb.shape[:2]:
            raise ValueError(f"depth {self.depth.shape} does not match rgb {self.rgb.shape[:2]}")


class CameraSource(Protocol):
    def latest(self) -> CameraFrame | None:
        """Most recent frame, or None before the first capture."""


class PoseEstimator(Protocol):
    """Replaceable backend. Returned poses may be in CAMERA_FRAME or BASE_FRAME (see ObjectPose.frame_id)."""
    needs_depth: bool

    def estimate(self, frame: CameraFrame) -> Sequence[ObjectPose]: ...


def to_base_frame(pose: ObjectPose, T_base_cam: np.ndarray) -> ObjectPose:
    if pose.frame_id == BASE_FRAME:
        return pose
    if pose.frame_id != CAMERA_FRAME:
        raise ValueError(f"cannot express pose from frame {pose.frame_id!r} in {BASE_FRAME!r}")
    T = np.asarray(T_base_cam, dtype=float)
    rotation = Rotation.from_matrix(T[:3, :3]) * Rotation.from_quat(pose.quaternion_xyzw)
    return ObjectPose(stamp=pose.stamp, frame_id=BASE_FRAME, position=T[:3, :3] @ pose.position + T[:3, 3],
                      quaternion_xyzw=rotation.as_quat(), confidence=pose.confidence, source=pose.source)


class VisionService:
    """VisionPort backed by a CameraSource and a PoseEstimator."""

    def __init__(self, camera: CameraSource, estimator: PoseEstimator):
        self.camera, self.estimator = camera, estimator
        self.frames_processed = 0
        self.estimator_errors = 0
        self.last_error = ""
        self._last_stamp = float("-inf")

    def detections(self, since: float) -> Sequence[ObjectPose]:
        frame = self.camera.latest()
        if frame is None or frame.stamp <= self._last_stamp:
            return []
        if self.estimator.needs_depth and frame.depth is None:
            raise ValueError("the pose estimator needs depth but the camera provides RGB only")
        self._last_stamp = frame.stamp
        try:
            poses = self.estimator.estimate(frame)
        except Exception as error:      # a backend failure must degrade to "nothing seen", not stop the FSM
            self.estimator_errors += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return []
        self.frames_processed += 1
        result = [to_base_frame(pose, frame.T_base_cam) for pose in poses]
        return [pose for pose in result if pose.stamp > since]


@dataclass
class MockPoseEstimator:
    """Ground-truth pose with configurable imperfections, for sim2sim and FSM tests (no depth needed)."""
    truth: Callable[[], ObjectPose | None]   # current true pose in BASE_FRAME, None if not in view
    position_sigma_m: float = 0.003
    rotation_sigma_deg: float = 1.0
    dropout_probability: float = 0.0
    outlier_probability: float = 0.0
    outlier_offset_m: float = 0.10
    seed: int = 0
    needs_depth: bool = False

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def estimate(self, frame: CameraFrame) -> Sequence[ObjectPose]:
        true = self.truth()
        if true is None or self._rng.random() < self.dropout_probability:
            return []
        position = true.position + self._rng.normal(0, self.position_sigma_m, 3)
        if self._rng.random() < self.outlier_probability:
            position = position + self._rng.normal(0, 1, 3) * self.outlier_offset_m
        rotation = Rotation.from_rotvec(np.radians(self._rng.normal(0, self.rotation_sigma_deg, 3))) \
            * Rotation.from_quat(true.quaternion_xyzw)
        return [ObjectPose(stamp=frame.stamp, frame_id=true.frame_id, position=position,
                           quaternion_xyzw=rotation.as_quat(), confidence=0.9, source="mock")]
