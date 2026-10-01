import json

import numpy as np
import pytest

from ramen_controller.contracts import JOINT_NAMES, MotionKind, ObjectPose
from ramen_controller.motion_bundle import BUNDLE_FORMAT, MotionBundleError, load_segment
from ramen_controller.pose_average import PoseAveragingConfig, PoseAveragingError, average_poses


def write_bundle(folder, frames=31, fps=30, **overrides):
    qpos = np.zeros((frames, 36))
    qpos[:, 3] = 1.0
    qpos[:, 7 + 15:7 + 29] = np.linspace(0, 1, frames)[:, None]       # arms ramp 0 -> 1
    qpos[:, 7 + 12:7 + 15] = [0.1, 0.0, 0.2]                         # waist yaw, roll, pitch
    metadata = {"format": BUNDLE_FORMAT, "joint_names": list(JOINT_NAMES), "fps": fps, "samples": frames}
    metadata.update(overrides)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(json.dumps(metadata))
    np.savetxt(folder / "motion.csv", qpos, delimiter=",")
    return folder


def test_segment_maps_arms_and_waist(tmp_path):
    segment = load_segment(write_bundle(tmp_path / "b"), MotionKind.GRASP)
    assert segment.arm_targets.shape == (31, 14) and segment.fps == 30
    assert np.isclose(segment.duration, 1.0)
    assert np.allclose(segment.arm_targets[-1], 1.0)
    assert np.allclose(segment.waist_ypr[0], [0.1, 0.0, 0.2])


def test_segment_slice_by_time(tmp_path):
    segment = load_segment(write_bundle(tmp_path / "b"), MotionKind.PLACE, start_s=0.5, end_s=1.0)
    assert np.isclose(segment.arm_targets[0, 0], 0.5) and np.isclose(segment.arm_targets[-1, 0], 1.0)


@pytest.mark.parametrize("override", [{"format": "other"}, {"fps": 0}, {"samples": 99},
                                      {"joint_names": list(reversed(JOINT_NAMES))}])
def test_bundle_contract_violations_are_rejected(tmp_path, override):
    with pytest.raises(MotionBundleError):
        load_segment(write_bundle(tmp_path / "b", **override), MotionKind.READY)


def test_segment_sampling_interpolates_and_clamps(tmp_path):
    segment = load_segment(write_bundle(tmp_path / "b"), MotionKind.READY)
    assert np.isclose(segment.sample(0.5)[0][0], 0.5)
    assert np.isclose(segment.sample(99.0)[0][0], 1.0)


def pose(x, stamp=0.0, quat=(0, 0, 0, 1), confidence=0.9, frame="base"):
    return ObjectPose(stamp=stamp, frame_id=frame, position=[x, 0, 1], quaternion_xyzw=quat,
                      confidence=confidence)


def test_average_rejects_position_outlier():
    poses = [pose(0.50 + 0.001 * i) for i in range(6)] + [pose(0.80)]
    result = average_poses(poses)
    assert abs(result.position[0] - 0.5025) < 1e-3 and result.source == "mean[6/7]"


def test_average_uses_rotation_space_not_euler_components():
    near_pi = [[0, 0, np.sin(a / 2), np.cos(a / 2)] for a in (np.pi - 0.02, -np.pi + 0.02) * 3]
    result = average_poses([pose(0.5, quat=q) for q in near_pi])
    assert abs(abs(result.quaternion_xyzw[2]) - 1.0) < 1e-3     # ~180 deg about z, not ~0


@pytest.mark.parametrize("poses,message", [
    ([pose(0.5)] * 3, "need 5"),
    ([pose(0.5, confidence=0.2)] * 8, "0 confident"),
    ([pose(0.5)] * 5 + [pose(0.5, frame="camera")], "mixed frames"),
    ([pose(0.5, quat=(0, 0, np.sin(a / 2), np.cos(a / 2))) for a in np.linspace(0, 1.5, 8)], "rotation spread"),
])
def test_average_failures_explain_why(poses, message):
    with pytest.raises(PoseAveragingError, match=message):
        average_poses(poses, PoseAveragingConfig())
