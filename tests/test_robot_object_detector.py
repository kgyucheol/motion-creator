import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location('robot_object_detector', Path(__file__).resolve().parents[1] / 'scripts/robot_object_detector.py')
detector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detector)

FX, FY, CX, CY = detector.INTRINSICS
WIDTH, HEIGHT = detector.COLOR_SIZE


def cylinder_cloud(center, length, radius, axis_angle_deg=0.0, stride=4):
    """Camera-frame points of the camera-facing side of a cylinder lying across the view, sampled like the robot's cloud."""
    u, v = np.meshgrid(np.arange(0, WIDTH, stride), np.arange(0, HEIGHT, stride))
    rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones(u.shape)], axis=-1)
    angle = np.radians(axis_angle_deg)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    center = np.asarray(center, dtype=float)
    d_perp = rays - (rays @ axis)[..., None] * axis
    c_perp = center - (center @ axis) * axis
    a = (d_perp ** 2).sum(-1)
    b = -2 * (d_perp * c_perp).sum(-1)
    c = c_perp @ c_perp - radius ** 2
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.where(disc > 0, disc, np.nan))) / (2 * a)
    along = (t[..., None] * rays - center) @ axis
    hit = np.isfinite(t) & (np.abs(along) <= length / 2)
    return (t[..., None] * rays)[hit], axis


def distance_to_line(point, origin, direction):
    offset = np.asarray(point) - origin
    return np.linalg.norm(offset - (offset @ direction) * direction)


def test_ridge_ends_and_normal_give_the_axis_when_moved_one_radius_away_from_the_camera():
    center, length, radius = np.array([0.02, -0.03, 0.55]), 0.30, 0.065
    points, axis = cylinder_cloud(center, length, radius, axis_angle_deg=10)
    geometry = detector.row_geometry(points)
    assert geometry is not None and geometry['clipped'] == [False, False]
    assert abs(np.dot(geometry['axis'], axis)) > np.cos(np.radians(2))
    assert 0.12 < geometry['width'] < 0.135                            # tangent points at the silhouette are sparse, so slightly under 2r and abs(geometry['length'] - length) < 0.02
    normal = np.array(geometry['normal'])
    assert normal[2] < -0.9                                           # faces the camera
    for end, sign in ((geometry['start'], -1), (geometry['end'], 1)):
        on_axis = np.asarray(end) - radius * normal
        assert distance_to_line(on_axis, center, axis) < 0.006
        assert abs((on_axis - center) @ axis - sign * length / 2) < 0.02
    assert (np.array(geometry['end']) - geometry['start']) @ axis > 0  # start -> end follows the axis toward the image right


def test_a_row_that_runs_out_of_the_image_is_marked_clipped_at_that_end():
    points, _ = cylinder_cloud([0.0, 0.0, 0.5], length=1.5, radius=0.065)
    assert detector.row_geometry(points)['clipped'] == [True, True]
    left_edge = np.array([-0.05, 0.0, 0.5])                          # 0.3 m row left of the image center, both ends inside the view
    points, _ = cylinder_cloud(left_edge, length=0.30, radius=0.065)
    clipped = detector.row_geometry(points)['clipped']
    assert clipped == [False, False]


def test_too_few_points_or_points_outside_the_depth_range_give_no_geometry():
    points, _ = cylinder_cloud([0.0, 0.0, 0.5], length=0.2, radius=0.04)
    assert detector.row_geometry(points[:20]) is None
    assert detector.row_geometry(points * [1, 1, 0.1]) is None       # nearer than the usable range


def test_cloud_message_round_trip_and_projection():
    import struct
    xyz_mm = np.array([[0, 0, 500], [100, -50, 600]], dtype='<i2')
    message = struct.pack('<4sIHH', b'PC01', 2, 4, 0) + xyz_mm.tobytes() + bytes(6)
    points = detector.parse_cloud(message)
    assert np.allclose(points, [[0, 0, 0.5], [0.1, -0.05, 0.6]])
    u, v = detector.project(points)
    assert np.allclose([u[0], v[0]], [CX, CY]) and np.isclose(u[1], CX + FX * 0.1 / 0.6)


def test_letterbox_matches_the_ultralytics_padding_for_the_color_stream():
    scale, new_w, new_h, left, top = detector.letterbox_params(WIDTH, HEIGHT)
    assert (new_w, new_h, left, top) == (864, 489, 0, 187)
    assert abs(scale - 864 / 848) < 1e-12


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert detector.nms(boxes, np.array([0.6, 0.9, 0.5]), 0.5) == [1, 2]
