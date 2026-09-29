#!/usr/bin/env python3
"""Join a cup stack into one textured mesh while preserving its arrangement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--blend-output", type=Path, required=True)
    parser.add_argument("--glb-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def mesh_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = (obj.matrix_world @ vertex.co for vertex in obj.data.vertices)
    first = next(points)
    lower = first.copy()
    upper = first.copy()
    for point in points:
        for axis in range(3):
            lower[axis] = min(lower[axis], point[axis])
            upper[axis] = max(upper[axis], point[axis])
    return lower, upper


def main() -> None:
    args = parse_args()
    for path in (args.blend_output, args.glb_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(args.source.resolve()))
    objects = list(bpy.context.scene.objects)
    if len(objects) != 30 or any(obj.type != "MESH" for obj in objects):
        raise RuntimeError("Source must contain exactly 30 cup mesh objects")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    active = bpy.data.objects.get("Mupama_Cup_01")
    if active is None:
        raise RuntimeError("First cup was not found")
    bpy.context.view_layer.objects.active = active
    bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = "Mupama_Cups_30_One_Object"
    merged.data.name = "Mupama_Cups_30_Merged_Mesh"

    # The active cup carries a random Z rotation. Bake it into the joined mesh
    # so downstream scenes get an identity transform at the stack base.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    merged["asset_role"] = "motion_validation_prop"
    merged["cup_count"] = 30
    merged["source_stack"] = str(args.source.resolve())

    lower, upper = mesh_bounds(merged)
    dimensions = upper - lower
    if abs(dimensions.z - 0.70) > 1e-6:
        raise RuntimeError(f"Unexpected merged stack height: {dimensions.z} m")
    if len(bpy.context.scene.objects) != 1:
        raise RuntimeError("Merged scene must contain exactly one object")

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output.resolve()))
    bpy.ops.export_scene.gltf(
        filepath=str(args.glb_output.resolve()),
        export_format="GLB",
        use_selection=False,
        export_yup=True,
    )

    report = {
        "source": str(args.source.resolve()),
        "blend": str(args.blend_output.resolve()),
        "glb": str(args.glb_output.resolve()),
        "object": merged.name,
        "scene_object_count": len(bpy.context.scene.objects),
        "cup_count": 30,
        "vertices": len(merged.data.vertices),
        "polygons": len(merged.data.polygons),
        "uv_layers": len(merged.data.uv_layers),
        "materials": [material.name for material in merged.data.materials if material],
        "dimensions_m": [round(value, 9) for value in dimensions],
        "bounds_min_m": [round(value, 9) for value in lower],
        "bounds_max_m": [round(value, 9) for value in upper],
        "location": [round(value, 9) for value in merged.location],
        "rotation_euler": [round(value, 9) for value in merged.rotation_euler],
        "scale": [round(value, 9) for value in merged.scale],
        "packed_image_count": sum(bool(image.packed_file) for image in bpy.data.images),
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
