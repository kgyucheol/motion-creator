"""Shared MuJoCo collision proxies for authored scene objects."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def numbers(values) -> str:
    return " ".join(str(float(value)) for value in values)


def grounded_position(item: dict) -> np.ndarray:
    position = np.asarray(item["position"], dtype=float).copy()
    if item.get("placement", {}).get("ground_lock") is False:
        return position
    size = np.asarray(item["size"], dtype=float)
    if item["shape"] == "sphere":
        extent = size[0] / 2
    else:
        x, y, z, w = np.asarray(item["quaternion_xyzw"], dtype=float)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)
        extent = (size[0] / 2 * np.hypot(r20, r21) + size[2] / 2 * abs(r22)
                  if item["shape"] == "cylinder" else
                  abs(r20) * size[0] / 2 + abs(r21) * size[1] / 2 + abs(r22) * size[2] / 2)
    position[2] = max(position[2], extent)
    return position


def append_collision_geoms(body: ET.Element, prefix: str, index: int, item: dict) -> list[str]:
    """Append a primitive or five-panel open-box proxy and return geom names."""
    size = np.asarray(item["size"], dtype=float)
    friction = numbers([item["friction"], .005, .0001])
    common = {"friction": friction, "contype": "0", "conaffinity": "0"}
    base = f"{prefix}_object_geom_{index}"
    if item["shape"] != "open_box":
        mj_size = (size / 2 if item["shape"] == "box" else [size[0] / 2]
                   if item["shape"] == "sphere" else [size[0] / 2, size[2] / 2])
        ET.SubElement(body, "geom", name=base, type=item["shape"], size=numbers(mj_size),
                      mass=str(float(item["mass_kg"])), **common)
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
