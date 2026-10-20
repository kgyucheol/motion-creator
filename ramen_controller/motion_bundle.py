"""Read MotionCreator bundles (`motions/<name>_<id>/`) as upper-body motion segments.

Only `metadata.json` and `motion.csv` are used; MotionCreator stays an independent authoring tool.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import ARMS_SLICE, JOINT_NAMES, WAIST_SLICE, MotionKind, MotionSegment

BUNDLE_FORMAT = "motioncreator.g1.v1"
QPOS_COLUMNS = 7 + len(JOINT_NAMES)   # root xyz + wxyz, then the 29 hinge joints


class MotionBundleError(ValueError):
    pass


def load_bundle(folder) -> tuple[float, np.ndarray]:
    """Return (fps, qpos[T, 36]) after checking the bundle matches the contract used here."""
    folder = Path(folder)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MotionBundleError(f"{folder} has no metadata.json") from error
    if metadata.get("format") != BUNDLE_FORMAT:
        raise MotionBundleError(f"unsupported bundle format {metadata.get('format')!r}, expected {BUNDLE_FORMAT!r}")
    if tuple(metadata.get("joint_names", ())) != JOINT_NAMES:
        raise MotionBundleError("bundle joint_names differ from the G1 29-DoF order")
    fps = float(metadata.get("fps", 0))
    if fps <= 0:
        raise MotionBundleError("bundle fps missing or invalid")
    csv_path = folder / "motion.csv"
    if not csv_path.is_file():
        raise MotionBundleError(f"{folder} has no motion.csv")
    qpos = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if qpos.shape[1] != QPOS_COLUMNS:
        raise MotionBundleError(f"motion.csv must have {QPOS_COLUMNS} columns, got {qpos.shape[1]}")
    if "samples" in metadata and int(metadata["samples"]) != len(qpos):
        raise MotionBundleError(f"metadata samples={metadata['samples']} but motion.csv has {len(qpos)} rows")
    return fps, qpos


def load_segment(folder, kind: MotionKind, start_s: float = 0.0, end_s: float | None = None) -> MotionSegment:
    """Cut [start_s, end_s] out of a bundle as a segment of `kind` (whole bundle by default)."""
    fps, qpos = load_bundle(folder)
    first = int(round(start_s * fps))
    last = len(qpos) if end_s is None else int(round(end_s * fps)) + 1
    if not 0 <= first < min(last, len(qpos) + 1) or last - first < 1:
        raise MotionBundleError(f"empty slice [{start_s}, {end_s}] for a {(len(qpos) - 1) / fps:.2f}s bundle")
    joints = qpos[first:last, 7:]
    return MotionSegment(kind=kind, fps=fps, arm_targets=joints[:, ARMS_SLICE],
                         waist_ypr=joints[:, WAIST_SLICE], name=Path(folder).name)


def root_pose_at(folder, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(position, quaternion_wxyz) of the authored root (pelvis) at `time_s`; the frame authored scene poses refer to."""
    fps, qpos = load_bundle(folder)
    row = qpos[min(int(round(time_s * fps)), len(qpos) - 1)]
    return row[:3].copy(), row[3:7].copy()
