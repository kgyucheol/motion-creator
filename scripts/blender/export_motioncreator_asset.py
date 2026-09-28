#!/usr/bin/env python3
"""Export visible Blender mesh objects as one Z-up GLB plus import metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def main() -> None:
    args = parse_args()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("Blender file contains no visible mesh objects")
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    lower = [min(point[axis] for point in corners) for axis in range(3)]
    upper = [max(point[axis] for point in corners) for axis in range(3)]
    properties = {}
    for obj in meshes:
        for key in obj.keys():
            if key == "_RNA_UI":
                continue
            value = obj[key]
            if isinstance(value, (str, int, float, bool)):
                properties[key] = value
            elif hasattr(value, "__len__"):
                properties[key] = [float(item) for item in value]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(args.output.resolve()),
        export_format="GLB",
        export_yup=False,
        export_apply=True,
        use_visible=True,
    )
    report = {
        "objects": [obj.name for obj in meshes],
        "bounds_min": lower,
        "bounds_max": upper,
        "dimensions": [upper[index] - lower[index] for index in range(3)],
        "properties": properties,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Blender 5 can keep background processes alive after a scripted glTF export.
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
