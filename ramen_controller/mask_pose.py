"""Object pose from a segmentation mask and an aligned depth image, as a `PoseEstimator` backend.

The segmenter (a YOLO-seg inferencer, for example) only has to return boolean masks at the depth resolution; this module
turns each mask into a position, a long-axis orientation and extents in the camera optical frame (x right, y down,
z forward). `VisionService` then expresses the pose in the robot base frame with the frame's `T_base_cam`.

Only the visible surface of an object is seen, so `position` is the center of the visible points' bounding box along the
principal axes, not the center of the whole object. A grasp planner that needs the object center has to add the missing
depth itself (for a lying cylinder: about one radius away from the camera, along the camera-facing normal axis).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import ObjectPose
from .vision import CAMERA_FRAME, CameraFrame

DEPTH_RANGE_M = (0.15, 3.0)
MIN_POINTS = 200
OUTLIER_MADS = 3.0
OUTLIER_FLOOR_M = 0.01
EXTENT_PERCENTILES = (1.0, 99.0)


@dataclass(frozen=True)
class Segmentation:
    """One instance from a segmenter. `mask` is (H, W) bool at the depth image resolution."""
    mask: np.ndarray
    label: str = "object"
    confidence: float = 1.0


Segmenter = Callable[[np.ndarray], Sequence[Segmentation]]     # rgb (H, W, 3) uint8 -> instances


@dataclass(frozen=True)
class MaskGeometry:
    """Pose and size of the visible points of one mask, in the camera optical frame."""
    position: np.ndarray             # (3,) center of the visible points' bounding box along the principal axes
    quaternion_xyzw: np.ndarray      # (4,) columns are: long axis, middle axis, camera-facing shortest axis
    extents: np.ndarray              # (3,) sizes along those axes, long to short, in meters
    point_count: int


@dataclass(frozen=True)
class MaskDetection:
    label: str
    confidence: float
    geometry: MaskGeometry


def deproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points (H, W, 3) from a z-depth image in meters."""
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    return np.dstack([(u - K[0, 2]) * depth_m / K[0, 0], (v - K[1, 2]) * depth_m / K[1, 1], depth_m])


def mask_geometry(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray, *, depth_range=DEPTH_RANGE_M,
                  min_points: int = MIN_POINTS) -> MaskGeometry | None:
    """Principal-axis pose of the depth points under `mask`, or None when too few valid points remain."""
    mask, depth_m, K = np.asarray(mask, dtype=bool), np.asarray(depth_m, dtype=float), np.asarray(K, dtype=float)
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask {mask.shape} does not match depth {depth_m.shape}")
    z = depth_m[mask]
    valid = np.isfinite(z) & (z >= depth_range[0]) & (z <= depth_range[1])
    if valid.sum() < min_points:
        return None
    points = deproject(depth_m, K)[mask][valid]
    z = points[:, 2]
    median = np.median(z)
    spread = OUTLIER_MADS * 1.4826 * np.median(np.abs(z - median)) + OUTLIER_FLOOR_M     # background bleeding across the mask edge
    points = points[np.abs(z - median) <= spread]
    if len(points) < min_points:
        return None

    centroid = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - centroid).T))
    long_axis, middle_axis, short_axis = (eigenvectors[:, index] for index in np.argsort(eigenvalues)[::-1])
    pivot = long_axis[0] if abs(long_axis[0]) > 1e-6 else long_axis[1]
    long_axis = long_axis if pivot >= 0 else -long_axis                    # fixed sign: long axis points to the image right (or down)
    short_axis = short_axis if short_axis[2] <= 0 else -short_axis          # shortest axis faces the camera
    middle_axis = np.cross(short_axis, long_axis)
    rotation = np.column_stack([long_axis, middle_axis, short_axis])

    local = (points - centroid) @ rotation
    low, high = np.percentile(local, EXTENT_PERCENTILES, axis=0)
    center = centroid + rotation @ ((low + high) / 2)
    return MaskGeometry(position=center, quaternion_xyzw=Rotation.from_matrix(rotation).as_quat(),
                        extents=high - low, point_count=len(points))


class MaskDepthPoseEstimator:
    """`PoseEstimator` backed by a segmenter and the camera's aligned depth. Poses are in CAMERA_FRAME."""
    needs_depth = True

    def __init__(self, segmenter: Segmenter, *, labels: Sequence[str] | None = None, min_confidence: float = 0.0,
                 depth_range=DEPTH_RANGE_M, min_points: int = MIN_POINTS, source: str = "mask-depth"):
        self.segmenter, self.labels, self.min_confidence = segmenter, None if labels is None else set(labels), min_confidence
        self.depth_range, self.min_points, self.source = depth_range, min_points, source

    def detect(self, frame: CameraFrame) -> list[MaskDetection]:
        if frame.depth is None:
            raise ValueError("MaskDepthPoseEstimator needs a depth image")
        detections = []
        for item in self.segmenter(frame.rgb):
            if item.confidence < self.min_confidence or (self.labels is not None and item.label not in self.labels):
                continue
            geometry = mask_geometry(item.mask, frame.depth, frame.K, depth_range=self.depth_range, min_points=self.min_points)
            if geometry is not None:
                detections.append(MaskDetection(item.label, float(item.confidence), geometry))
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)

    def estimate(self, frame: CameraFrame) -> Sequence[ObjectPose]:
        return [ObjectPose(stamp=frame.stamp, frame_id=CAMERA_FRAME, position=detection.geometry.position,
                           quaternion_xyzw=detection.geometry.quaternion_xyzw, confidence=detection.confidence,
                           source=f"{self.source}:{detection.label}")
                for detection in self.detect(frame)]
