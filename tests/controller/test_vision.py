import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ramen_controller.contracts import ObjectPose
from ramen_controller.pose_average import average_poses
from ramen_controller.vision import (BASE_FRAME, CAMERA_FRAME, CameraFrame, MockPoseEstimator, VisionService,
                                     to_base_frame)

K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])


def frame(stamp, T=None, depth=False):
    return CameraFrame(stamp=stamp, rgb=np.zeros((48, 64, 3), np.uint8), K=K,
                       T_base_cam=np.eye(4) if T is None else T,
                       depth=np.ones((48, 64)) if depth else None)


class Camera:
    def __init__(self):
        self.current = None

    def latest(self):
        return self.current


def true_pose(frame_id=BASE_FRAME):
    return ObjectPose(stamp=0, frame_id=frame_id, position=[0.5, 0.1, 0.9], quaternion_xyzw=[0, 0, 0, 1])


def test_camera_to_base_transform():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T[:3, 3] = [1, 0, 0.5]
    pose = ObjectPose(stamp=2, frame_id=CAMERA_FRAME, position=[1, 0, 0], quaternion_xyzw=[0, 0, 0, 1], confidence=.7)
    out = to_base_frame(pose, T)
    assert out.frame_id == BASE_FRAME and np.allclose(out.position, [1, 1, 0.5])
    assert np.isclose(Rotation.from_quat(out.quaternion_xyzw).as_euler("xyz", degrees=True)[2], 90)
    assert out.confidence == .7 and out.stamp == 2
    assert to_base_frame(out, T) is out                      # already in base: untouched


def test_unknown_frame_is_rejected():
    with pytest.raises(ValueError, match="frame"):
        to_base_frame(true_pose("world"), np.eye(4))


def test_service_runs_each_frame_once_and_filters_by_since():
    camera, truth = Camera(), true_pose()
    service = VisionService(camera, MockPoseEstimator(truth=lambda: truth, position_sigma_m=0))
    assert service.detections(-1) == []                       # no frame yet
    camera.current = frame(1.0)
    assert len(service.detections(0.5)) == 1
    assert service.detections(0.5) == []                      # same frame: not processed twice
    camera.current = frame(2.0)
    assert service.detections(5.0) == [] and service.frames_processed == 2   # older than `since`: dropped


def test_service_converts_camera_frame_backend_output():
    class CameraFrameBackend:
        needs_depth = True
        def estimate(self, f):
            return [ObjectPose(stamp=f.stamp, frame_id=CAMERA_FRAME, position=[0, 0, 1], quaternion_xyzw=[0, 0, 0, 1])]
    T = np.eye(4); T[:3, 3] = [0.2, 0, 1.0]
    camera = Camera(); camera.current = frame(1.0, T, depth=True)
    (pose,) = VisionService(camera, CameraFrameBackend()).detections(0)
    assert pose.frame_id == BASE_FRAME and np.allclose(pose.position, [0.2, 0, 2.0])


def test_rgb_only_camera_with_depth_backend_fails_loudly():
    class DepthBackend:
        needs_depth = True
        def estimate(self, f):
            return []
    camera = Camera(); camera.current = frame(1.0, depth=False)
    with pytest.raises(ValueError, match="needs depth"):
        VisionService(camera, DepthBackend()).detections(0)


def test_backend_exception_degrades_to_no_detection():
    class Broken:
        needs_depth = False
        def estimate(self, f):
            raise RuntimeError("cuda out of memory")
    camera = Camera(); camera.current = frame(1.0)
    service = VisionService(camera, Broken())
    assert service.detections(0) == []
    assert service.estimator_errors == 1 and "cuda out of memory" in service.last_error


def test_mock_imperfections_are_survivable_by_the_averaging_stage():
    camera, truth = Camera(), true_pose()
    service = VisionService(camera, MockPoseEstimator(truth=lambda: truth, dropout_probability=0.3,
                                                     outlier_probability=0.15, seed=3))
    samples = []
    for i in range(60):
        camera.current = frame(i * 0.03)
        samples += service.detections(-1)
    assert 25 < len(samples) < 55                             # dropouts really happen
    average = average_poses(samples)
    assert np.linalg.norm(average.position - truth.position) < 0.01


def test_mock_is_seeded_and_not_visible_returns_nothing():
    visible = [True]
    a = MockPoseEstimator(truth=lambda: true_pose() if visible[0] else None, seed=5)
    b = MockPoseEstimator(truth=lambda: true_pose(), seed=5)
    assert np.allclose(a.estimate(frame(0))[0].position, b.estimate(frame(0))[0].position)
    visible[0] = False
    assert a.estimate(frame(1)) == []


def test_rotation_composes_as_base_from_camera_then_object():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    object_in_camera = Rotation.from_euler("x", 90, degrees=True)
    pose = ObjectPose(stamp=0, frame_id=CAMERA_FRAME, position=[0, 0, 0], quaternion_xyzw=object_in_camera.as_quat())
    expected = Rotation.from_matrix(T[:3, :3]) * object_in_camera
    got = Rotation.from_quat(to_base_frame(pose, T).quaternion_xyzw)
    assert (expected.inv() * got).magnitude() < 1e-9
    assert (( object_in_camera * Rotation.from_matrix(T[:3, :3])).inv() * got).magnitude() > 0.5   # not the swapped order
