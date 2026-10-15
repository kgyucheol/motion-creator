import numpy as np
from scipy.spatial.transform import Rotation

from ramen_controller.mask_pose import MaskDepthPoseEstimator, Segmentation, mask_geometry
from ramen_controller.vision import CAMERA_FRAME, BASE_FRAME, CameraFrame, to_base_frame

K = np.array([[608., 0, 428.], [0, 608., 256.], [0, 0, 1]])
SHAPE = (480, 848)


def cylinder_depth(center, length, radius, axis_angle_deg=0.0, background=0.0):
    """Z-depth of a cylinder lying across the view (axis in the image plane, rotated by `axis_angle_deg`), seen from the camera."""
    u, v = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(SHAPE)], axis=-1)
    angle = np.radians(axis_angle_deg)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    center = np.asarray(center, dtype=float)
    # perpendicular part of the ray offset: |t*d - c - ((t*d - c).axis) axis|^2 = r^2
    d_perp = rays - (rays @ axis)[..., None] * axis
    c_perp = center - (center @ axis) * axis
    a = (d_perp ** 2).sum(-1)
    b = -2 * (d_perp * c_perp).sum(-1)
    c = c_perp @ c_perp - radius ** 2
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.where(disc > 0, disc, np.nan))) / (2 * a)
    along = (t[..., None] * rays - center) @ axis
    hit = np.isfinite(t) & (np.abs(along) <= length / 2)
    depth = np.where(hit, t, background)             # rays have z = 1, so t is the z-depth
    return depth, hit


def test_a_lying_cylinder_gives_its_axis_center_and_size():
    depth, hit = cylinder_depth(center=(0.05, -0.02, 0.5), length=0.20, radius=0.04)
    geometry = mask_geometry(hit, depth, K)
    assert geometry is not None
    long_axis = Rotation.from_quat(geometry.quaternion_xyzw).as_matrix()[:, 0]
    assert abs(long_axis @ [1, 0, 0]) > np.cos(np.radians(2))
    assert abs(geometry.extents[0] - 0.20) < 0.01
    assert 0.06 < geometry.extents[1] < 0.09 and geometry.extents[2] < geometry.extents[1]
    assert np.allclose(geometry.position[:2], [0.05, -0.02], atol=0.005)
    assert 0.46 <= geometry.position[2] <= 0.50          # the visible surface only: between the top line and the silhouette


def test_the_axis_of_a_rotated_cylinder_is_recovered_with_a_fixed_sign():
    depth, hit = cylinder_depth(center=(0.0, 0.0, 0.55), length=0.18, radius=0.035, axis_angle_deg=30)
    rotation = Rotation.from_quat(mask_geometry(hit, depth, K).quaternion_xyzw).as_matrix()
    expected = np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0])
    assert rotation[:, 0] @ expected > np.cos(np.radians(3))        # same direction, not just parallel
    assert rotation[2, 2] <= 0                                      # the shortest axis faces the camera
    assert abs(np.linalg.det(rotation) - 1) < 1e-6


def test_far_background_inside_the_mask_does_not_move_the_pose():
    depth, hit = cylinder_depth(center=(0.0, 0.0, 0.5), length=0.20, radius=0.04)
    clean = mask_geometry(hit, depth, K)
    mask = hit.copy()
    mask[200:260, 600:640] = True                  # a patch of far wall swept into the mask
    depth = depth.copy()
    depth[200:260, 600:640] = 1.8
    dirty = mask_geometry(mask, depth, K)
    assert np.allclose(dirty.position, clean.position, atol=0.005)
    assert abs(dirty.extents[0] - clean.extents[0]) < 0.005


def test_too_few_or_out_of_range_depth_points_give_no_pose():
    depth, hit = cylinder_depth(center=(0.0, 0.0, 0.5), length=0.20, radius=0.04)
    assert mask_geometry(np.zeros(SHAPE, dtype=bool), depth, K) is None
    assert mask_geometry(hit, np.where(hit, 0.05, 0.0), K) is None          # nearer than the sensor's usable range
    assert mask_geometry(hit, np.where(hit, 0.0, 0.0), K) is None           # no depth at all


def frame(depth, T_base_cam=None):
    return CameraFrame(stamp=1.0, rgb=np.zeros(SHAPE + (3,), np.uint8), K=K, depth=depth,
                       T_base_cam=np.eye(4) if T_base_cam is None else T_base_cam)


def test_the_estimator_filters_by_label_and_confidence_and_orders_by_confidence():
    depth, hit = cylinder_depth(center=(0.0, 0.0, 0.5), length=0.20, radius=0.04)
    other, other_hit = cylinder_depth(center=(0.2, 0.1, 0.6), length=0.2, radius=0.04)
    both = np.where(hit, depth, other)
    items = [Segmentation(other_hit & ~hit, "cup", 0.6), Segmentation(hit, "cup", 0.9), Segmentation(hit, "box", 0.95),
             Segmentation(hit, "cup", 0.2)]
    estimator = MaskDepthPoseEstimator(lambda rgb: items, labels=["cup"], min_confidence=0.5)
    poses = estimator.estimate(frame(both))
    assert [round(pose.confidence, 2) for pose in poses] == [0.9, 0.6]
    assert all(pose.frame_id == CAMERA_FRAME and pose.source == "mask-depth:cup" for pose in poses)


def test_camera_frame_poses_convert_to_the_base_frame_with_the_capture_transform():
    depth, hit = cylinder_depth(center=(0.0, 0.0, 0.5), length=0.20, radius=0.04)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    T[:3, 3] = [0.1, 0.0, 0.4]
    pose = MaskDepthPoseEstimator(lambda rgb: [Segmentation(hit, "cup", 0.9)]).estimate(frame(depth, T))[0]
    base = to_base_frame(pose, T)
    assert base.frame_id == BASE_FRAME
    assert np.allclose(base.position, T[:3, :3] @ pose.position + T[:3, 3])


def test_a_mask_with_the_wrong_resolution_is_rejected():
    depth, hit = cylinder_depth(center=(0.0, 0.0, 0.5), length=0.20, radius=0.04)
    try:
        mask_geometry(hit[::2, ::2], depth, K)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("expected ValueError")
