"""Extract a tool's collision-mesh section in its TCP frame for the 2-D cylinder lab."""
from __future__ import annotations

from threading import Lock

import mujoco
import numpy as np

_SECTION_CACHE: dict[tuple[str, str], dict] = {}
_SECTION_CACHE_LOCK = Lock()


def _mesh_segments(vertices: np.ndarray, faces: np.ndarray, plane_y: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Intersect mesh triangles with a Y-constant plane and return unique line segments."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    epsilon = 1e-8
    for face in faces:
        triangle = vertices[face]
        distances = triangle[:, 1] - plane_y
        points: list[np.ndarray] = []
        for index in range(3):
            a, b = triangle[index], triangle[(index + 1) % 3]
            da, db = distances[index], distances[(index + 1) % 3]
            if abs(da) <= epsilon:
                points.append(a[[0, 2]])
            if da < -epsilon and db > epsilon or da > epsilon and db < -epsilon:
                points.append((a + (b - a) * (da / (da - db)))[[0, 2]])
        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - other) < 1e-8 for other in unique):
                unique.append(point)
        if len(unique) < 2:
            continue
        if len(unique) > 2:
            pair = max(((a, b) for i, a in enumerate(unique) for b in unique[i + 1:]),
                       key=lambda ab: np.linalg.norm(ab[0] - ab[1]))
            unique = list(pair)
        if np.linalg.norm(unique[0] - unique[1]) > 1e-8:
            segments.append((unique[0], unique[1]))
    return segments


def _contours(segments: list[tuple[np.ndarray, np.ndarray]], tolerance: float = 2e-6) -> list[np.ndarray]:
    """Join triangle intersections into closed section contours."""
    nodes: dict[tuple[int, int], np.ndarray] = {}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def key(point):
        return tuple(np.rint(point / tolerance).astype(np.int64))

    for a, b in segments:
        ka, kb = key(a), key(b)
        if ka == kb:
            continue
        nodes[ka], nodes[kb] = a, b
        edge = (ka, kb) if ka < kb else (kb, ka)
        if edge in edges:
            continue
        edges.add(edge)
        adjacency.setdefault(ka, set()).add(kb)
        adjacency.setdefault(kb, set()).add(ka)

    unused = set(edges)
    contours: list[np.ndarray] = []
    while unused:
        first = min(unused)
        start, current = first
        path = [start, current]
        unused.remove(first)
        previous = start
        for _ in range(len(edges) + 1):
            if current == start:
                break
            candidates = [neighbor for neighbor in adjacency.get(current, ()) if neighbor != previous
                          and ((current, neighbor) if current < neighbor else (neighbor, current)) in unused]
            if not candidates:
                break
            if len(candidates) > 1:
                incoming = nodes[current] - nodes[previous]
                candidates.sort(key=lambda neighbor: float(np.dot(
                    incoming / max(np.linalg.norm(incoming), 1e-12),
                    (nodes[neighbor] - nodes[current]) / max(np.linalg.norm(nodes[neighbor] - nodes[current]), 1e-12))), reverse=True)
            following = candidates[0]
            edge = (current, following) if current < following else (following, current)
            unused.remove(edge)
            previous, current = current, following
            path.append(current)
        if len(path) >= 4 and path[-1] == path[0]:
            contour = np.asarray([nodes[index] for index in path[:-1]], dtype=float)
            if abs(_signed_area(contour)) > 1e-9:
                contours.append(_simplify(contour))
    return contours


def _signed_area(points: np.ndarray) -> float:
    return float(.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                              - np.roll(points[:, 0], -1) * points[:, 1]))


def _simplify(points: np.ndarray, tolerance: float = .00025) -> np.ndarray:
    """Remove near-collinear mesh detail while keeping the collision silhouette in metres."""
    kept = list(points)
    changed = True
    while changed and len(kept) > 3:
        changed = False
        for index in range(len(kept)):
            a, b, c = kept[index - 1], kept[index], kept[(index + 1) % len(kept)]
            segment = c - a
            length = np.linalg.norm(segment)
            distance = abs(np.cross(segment, b - a)) / max(length, 1e-12)
            if distance <= tolerance and np.dot(b - a, c - b) >= -1e-10:
                kept.pop(index)
                changed = True
                break
    return np.asarray(kept, dtype=float)


def _triangulate(points: np.ndarray) -> list[list[list[float]]]:
    """Ear-clip a simple contour into convex triangles accepted by Planck.js."""
    polygon = points.copy()
    if _signed_area(polygon) < 0:
        polygon = polygon[::-1].copy()
    indices = list(range(len(polygon)))
    triangles: list[list[list[float]]] = []
    guard = len(indices) ** 2

    def cross(a, b, c):
        ab, ac = b - a, c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def in_triangle(point, a, b, c):
        return cross(a, b, point) >= -1e-10 and cross(b, c, point) >= -1e-10 and cross(c, a, point) >= -1e-10

    while len(indices) > 3 and guard:
        guard -= 1
        ear = None
        for pos, current in enumerate(indices):
            previous, following = indices[pos - 1], indices[(pos + 1) % len(indices)]
            a, b, c = polygon[previous], polygon[current], polygon[following]
            if cross(a, b, c) <= 1e-10:
                continue
            if any(in_triangle(polygon[other], a, b, c) for other in indices
                   if other not in (previous, current, following)):
                continue
            ear = pos
            triangles.append([a.tolist(), b.tolist(), c.tolist()])
            break
        if ear is None:
            return []
        indices.pop(ear)
    if len(indices) == 3:
        triangles.append([polygon[index].tolist() for index in indices])
    return triangles


def _extract_section(robot, side: str) -> dict:
    if robot.model_id != 'g1-tools' or robot.tools is None:
        raise ValueError('실제 툴 단면을 가져오려면 G1 그리퍼 모델에서 장착 툴을 선택하세요.')
    model = robot.model
    data = robot.data(robot.home)
    tcp_position, tcp_rotation = robot.point(data, f'{side}_hand')
    wrist = model.body(f'{side}_wrist_yaw_link').id
    body_id = robot.ids[f'{side}_hand']
    bodies = []
    while body_id != wrist and body_id > 0:
        bodies.append(body_id)
        body_id = int(model.body_parentid[body_id])
    if body_id != wrist:
        raise ValueError(f'{side} 툴의 장착 링크를 찾지 못했습니다.')

    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for body in bodies:
        for geom_id in range(model.ngeom):
            if (model.geom_bodyid[geom_id] != body or model.geom_contype[geom_id] == 0
                    or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = model.mesh_vert[va:va + vn].copy()
            faces = model.mesh_face[fa:fa + fn].copy()
            geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            world = data.geom_xpos[geom_id] + vertices @ geom_rotation.T
            in_tcp = (world - tcp_position) @ tcp_rotation
            meshes.append((in_tcp, faces))

    if not meshes:
        raise ValueError(f'{side} 툴에 단면을 만들 충돌 메시가 없습니다.')
    lateral = np.concatenate([vertices[:, 1] for vertices, _ in meshes])
    plane_y = float((lateral.min() + lateral.max()) * .5)
    contours: list[np.ndarray] = []
    for vertices, faces in meshes:
        for outline in _contours(_mesh_segments(vertices, faces, plane_y)):
            if len(outline) >= 3:
                contours.append(outline)

    outlines, fixtures, area = [], [], 0.
    for contour in contours:
        triangles = _triangulate(contour)
        if not triangles:
            continue
        outlines.append(contour.tolist())
        fixtures.extend(triangles)
        for triangle in triangles:
            a, b, c = np.asarray(triangle)
            area += abs(float(np.cross(b - a, c - a))) * .5
    if not fixtures:
        raise ValueError(f'{side} 툴의 중심 단면을 충돌 메시에서 만들지 못했습니다.')
    return {
        'side': side,
        'tool_id': robot.tools[0 if side == 'left' else 1],
        'section_y_m': plane_y,
        'outlines': outlines,
        'fixtures': fixtures,
        'area_m2': area,
    }


def tool_section(robot, side: str) -> dict:
    if side not in ('left', 'right'):
        raise ValueError('손은 left 또는 right여야 합니다.')
    key = (robot.cache_key, side)
    with _SECTION_CACHE_LOCK:
        cached = _SECTION_CACHE.get(key)
    if cached is not None:
        return cached
    profile = _extract_section(robot, side)
    with _SECTION_CACHE_LOCK:
        if len(_SECTION_CACHE) >= 32:
            _SECTION_CACHE.pop(next(iter(_SECTION_CACHE)))
        return _SECTION_CACHE.setdefault(key, profile)
