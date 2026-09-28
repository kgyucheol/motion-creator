"""Shared MuJoCo collision proxies for authored scene objects."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def numbers(values) -> str:
    return " ".join(str(float(value)) for value in values)


def _quaternion_product_xyzw(first, second) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(first, dtype=float)
    x2, y2, z2, w2 = np.asarray(second, dtype=float)
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _quaternion_matrix_xyzw(value) -> np.ndarray:
    x, y, z, w = np.asarray(value, dtype=float)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def grounded_position(item: dict) -> np.ndarray:
    position = np.asarray(item["position"], dtype=float).copy()
    if item.get("placement", {}).get("ground_lock") is False:
        return position
    shape = item.get("collision_shape", item["shape"])
    size = np.asarray(item.get("collision_size", item["size"]), dtype=float)
    quaternion = _quaternion_product_xyzw(
        item["quaternion_xyzw"], item.get("collision_quaternion_xyzw", [0., 0., 0., 1.]))
    if shape == "convex_hull":
        vertices = np.asarray(item["collision_hull_vertices"], dtype=float) * np.asarray(item["size"], dtype=float)
        extent = float(np.max(np.abs(vertices @ _quaternion_matrix_xyzw(quaternion).T[:, 2])))
    elif shape == "sphere":
        extent = size[0] / 2
    else:
        x, y, z, w = quaternion
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)
        extent = (size[0] / 2 * np.hypot(r20, r21) + size[2] / 2 * abs(r22)
                  if shape == "cylinder" else
                  abs(r20) * size[0] / 2 + abs(r21) * size[1] / 2 + abs(r22) * size[2] / 2)
    position[2] = max(position[2], extent)
    return position


def append_collision_geoms(body: ET.Element, prefix: str, index: int, item: dict,
                           root: ET.Element | None = None) -> list[str]:
    """Append a primitive or five-panel open-box proxy and return geom names."""
    shape = item.get("collision_shape", item["shape"])
    size = np.asarray(item.get("collision_size", item["size"]), dtype=float)
    friction = numbers([item["friction"], .005, .0001])
    common = {"friction": friction, "contype": "0", "conaffinity": "0"}
    base = f"{prefix}_object_geom_{index}"
    if shape == "convex_hull":
        if root is None:
            raise ValueError("Convex-hull collision requires the MuJoCo root element")
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        mesh_name = f"{prefix}_object_hull_{index}"
        vertices = np.asarray(item["collision_hull_vertices"], dtype=float)
        faces = np.asarray(item["collision_hull_faces"], dtype=int)
        ET.SubElement(asset, "mesh", name=mesh_name, vertex=numbers(vertices.ravel()),
                      face=" ".join(str(int(value)) for value in faces.ravel()),
                      scale=numbers(item["size"]))
        ET.SubElement(body, "geom", name=base, type="mesh", mesh=mesh_name,
                      mass=str(float(item["mass_kg"])), **common)
        return [base]
    if shape != "open_box":
        mj_size = (size / 2 if shape == "box" else [size[0] / 2]
                   if shape == "sphere" else [size[0] / 2, size[2] / 2])
        attributes = {"name": base, "type": shape, "size": numbers(mj_size),
                      "mass": str(float(item["mass_kg"])), **common}
        collision_quaternion = item.get("collision_quaternion_xyzw")
        if collision_quaternion is not None:
            attributes["quat"] = numbers(np.asarray(collision_quaternion, dtype=float)[[3, 0, 1, 2]])
        ET.SubElement(body, "geom", **attributes)
        return [base]

    thickness = float(np.clip(item.get("wall_thickness_m", .02), .001, min(size) / 3))
    height = size[2] - thickness
    parts = [
        ("bottom", [size[0], size[1], thickness], [0, 0, -size[2] / 2 + thickness / 2]),
        ("left", [thickness, size[1] - 2 * thickness, height], [-(size[0] - thickness) / 2, 0, thickness / 2]),
        ("right", [thickness, size[1] - 2 * thickness, height], [(size[0] - thickness) / 2, 0, thickness / 2]),
        ("front", [size[0], thickness, height], [0, -(size[1] - thickness) / 2, thickness / 2]),
        ("back", [size[0], thickness, height], [0, (size[1] - thickness) / 2, thickness / 2]),
    ]
    volumes = [float(np.prod(dimensions)) for _, dimensions, _ in parts]
    total_volume = sum(volumes)
    names = []
    for (part, dimensions, position), volume in zip(parts, volumes):
        name = f"{base}_{part}"
        ET.SubElement(body, "geom", name=name, type="box", size=numbers(np.asarray(dimensions) / 2),
                      pos=numbers(position), mass=str(float(item["mass_kg"]) * volume / total_volume), **common)
        names.append(name)
    return names
