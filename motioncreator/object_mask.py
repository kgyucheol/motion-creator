"""Object masks from a depth image by removing the dominant plane (table or floor)."""
import zlib
import struct
import numpy as np
from scipy import ndimage


def deproject(depth_m, fx, fy, cx, cy):
    """Camera-frame points (H, W, 3) from a z-depth image in meters."""
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    return np.dstack([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def fit_plane(points, iterations=300, threshold=.01, seed=0):
    """RANSAC plane n.p + d = 0 with n pointing at the camera (d > 0); refined on the inliers."""
    if len(points) < 3:
        raise ValueError('평면을 추정하기에 depth 점이 부족합니다.')
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        inliers = np.abs(points @ normal - a @ normal) < threshold
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None:
        raise ValueError('평면을 찾지 못했습니다.')
    inlier_points = points[best]
    centroid = inlier_points.mean(axis=0)
    normal = np.linalg.svd(inlier_points - centroid, full_matrices=False)[2][-1]
    d = -float(normal @ centroid)
    if d < 0:
        normal, d = -normal, -d
    return normal, d, best_count / len(points)


def segment_objects(depth_m, intrinsics, min_depth=.2, max_depth=2.5, min_height=.012, max_height=.5,
                    min_pixels=250, plane_threshold=.01, extent_range=None):
    """Return the plane and one dict per object standing on it.

    `extent_range` (min_m, max_m) optionally keeps objects whose longest extent is inside it."""
    fx, fy, cx, cy = intrinsics[:4]
    points = deproject(depth_m, fx, fy, cx, cy)
    valid = (depth_m > min_depth) & (depth_m < max_depth)
    normal, d, plane_fraction = fit_plane(points[valid][::7], threshold=plane_threshold)
    height = points @ normal + d
    candidate = valid & (height > min_height) & (height < max_height)
    candidate = ndimage.binary_opening(candidate, structure=np.ones((3, 3), bool))
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), bool))
    objects = []
    for index in range(1, count + 1):
        mask = labels == index
        if mask.sum() < min_pixels:
            continue
        cloud = points[mask]
        centroid = cloud.mean(axis=0)
        axes = np.linalg.svd(cloud - centroid, full_matrices=False)[2]
        low, high = np.percentile((cloud - centroid) @ axes.T, [2, 98], axis=0)
        extents = np.sort(high - low)[::-1]
        if extent_range and not extent_range[0] <= extents[0] <= extent_range[1]:
            continue
        objects.append({'mask': mask, 'pixels': int(mask.sum()), 'centroid': centroid, 'extents': extents,
                        'max_height': float(height[mask].max()), 'axis': axes[0]})
    objects.sort(key=lambda item: -item['pixels'])
    return {'normal': normal, 'd': d, 'plane_fraction': plane_fraction}, objects


def overlay(color_bgr, objects):
    """Tint each object mask over a BGR image; returns an RGB uint8 image."""
    palette = np.array([[255, 80, 80], [80, 255, 120], [90, 150, 255], [255, 220, 60], [220, 90, 255]], float)
    image = color_bgr[..., ::-1].astype(float)
    for number, item in enumerate(objects):
        mask = item['mask']
        image[mask] = image[mask] * .45 + palette[number % len(palette)] * .55
        edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
        image[edge] = palette[number % len(palette)]
    return image.astype(np.uint8)


def write_png(path, rgb):
    """Minimal PNG writer for (H, W, 3) uint8 or (H, W) bool/uint8 images."""
    image = np.asarray(rgb)
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    channels = 1 if image.ndim == 2 else 3
    rows = b''.join(b'\x00' + image[row].tobytes() for row in range(image.shape[0]))

    def chunk(kind, data):
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)

    header = struct.pack('>IIBBBBB', image.shape[1], image.shape[0], 8, 0 if channels == 1 else 2, 0, 0, 0)
    with open(path, 'wb') as handle:
        handle.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
