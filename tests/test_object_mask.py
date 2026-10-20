import numpy as np

from motioncreator.object_mask import segment_objects, write_png

FX = FY = 420.
WIDTH, HEIGHT = 212, 240
CX, CY = WIDTH / 2, HEIGHT / 2


def scene_depth(objects):
    """Z-depth of a floor seen from a tilted camera with horizontal cylinders (x, y, radius, length) lying on it."""
    eye = np.array([0., -.7, .55])
    forward = np.array([0., .7, -.55]) / np.linalg.norm([.7, .55])
    right = np.array([1., 0., 0.])
    down = np.cross(forward, right)
    u, v = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    rays = ((u - CX) / FX)[..., None] * right + ((v - CY) / FY)[..., None] * down + forward
    depth = np.zeros((HEIGHT, WIDTH))
    for t in np.linspace(.25, 2.5, 900)[::-1]:
        point = eye + rays * t
        surface = np.zeros(point.shape[:2])
        for x, y, radius, length in objects:
            across = np.abs(point[..., 0] - x)
            inside = (np.abs(point[..., 1] - y) < length / 2) & (across < radius)
            surface = np.where(inside, np.maximum(surface, radius + np.sqrt(np.maximum(radius ** 2 - across ** 2, 0))), surface)
        depth = np.where(point[..., 2] <= surface, t, depth)
    return depth


def test_floor_plane_is_removed_and_horizontal_cylinder_is_found():
    depth = scene_depth([(0., .02, .07, .3)])
    rng = np.random.default_rng(1)
    depth = np.where(depth > 0, depth + rng.normal(0, .002, depth.shape), 0)
    plane, objects = segment_objects(depth, (FX, FY, CX, CY), min_pixels=100)
    assert plane['plane_fraction'] > .5
    assert len(objects) == 1
    found = objects[0]
    assert abs(found['extents'][0] - .3) < .03
    assert .05 < found['max_height'] < .17
    # the cylinder axis is world Y, i.e. (0, -.618, .786) in this camera's frame
    assert abs(found['axis'] @ np.array([0., -.618, .786])) > .95


def test_two_objects_and_extent_filter():
    depth = scene_depth([(-.17, 0., .06, .22), (.15, .04, .07, .1)])
    _, objects = segment_objects(depth, (FX, FY, CX, CY), min_pixels=80)
    assert len(objects) == 2
    _, long_only = segment_objects(depth, (FX, FY, CX, CY), min_pixels=80, extent_range=(.18, 1.))
    assert len(long_only) == 1 and long_only[0]['extents'][0] > .18


def test_png_writer_roundtrip_header(tmp_path):
    path = tmp_path / 'mask.png'
    write_png(path, np.zeros((4, 6), bool))
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
