"""Robust averaging of repeated object-pose detections (the OBSERVE stage)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import ObjectPose


class PoseAveragingError(ValueError):
    pass


@dataclass(frozen=True)
class PoseAveragingConfig:
    min_samples: int = 5
    min_confidence: float = 0.6
    max_position_deviation_m: float = 0.03   # samples farther than this from the median are outliers
    max_rotation_spread_deg: float = 15.0    # accepted samples must agree on rotation within this angle


def average_poses(poses: Sequence[ObjectPose], config: PoseAveragingConfig = PoseAveragingConfig()) -> ObjectPose:
    """Median-gated position mean and chordal rotation mean. Raises PoseAveragingError with the reason."""
    confident = [pose for pose in poses if pose.confidence >= config.min_confidence]
    if len(confident) < config.min_samples:
        raise PoseAveragingError(f"{len(confident)} confident samples, need {config.min_samples}")
    frames = {pose.frame_id for pose in confident}
    if len(frames) != 1:
        raise PoseAveragingError(f"samples are in mixed frames: {sorted(frames)}")
    positions = np.stack([pose.position for pose in confident])
    keep = np.linalg.norm(positions - np.median(positions, axis=0), axis=1) <= config.max_position_deviation_m
    accepted = [pose for pose, flag in zip(confident, keep) if flag]
    if len(accepted) < config.min_samples:
        raise PoseAveragingError(f"{len(accepted)} samples remain after outlier rejection, need {config.min_samples}")
    rotations = Rotation.from_quat([pose.quaternion_xyzw for pose in accepted])
    mean = rotations.mean()
    spread = np.degrees((mean.inv() * rotations).magnitude().max())
    if spread > config.max_rotation_spread_deg:
        raise PoseAveragingError(f"rotation spread {spread:.1f} deg exceeds {config.max_rotation_spread_deg} deg")
    return ObjectPose(stamp=max(pose.stamp for pose in accepted), frame_id=accepted[0].frame_id,
                      position=np.mean([pose.position for pose in accepted], axis=0),
                      quaternion_xyzw=mean.as_quat(),
                      confidence=float(np.mean([pose.confidence for pose in accepted])),
                      source=f"mean[{len(accepted)}/{len(poses)}]")
