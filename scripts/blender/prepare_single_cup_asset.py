#!/usr/bin/env python3
"""Prepare a single-cup asset using a measured top diameter."""

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
    parser.add_argument("--source-top-diameter", type=float, default=0.14)
    parser.add_argument("--target-top-diameter", type=float, default=0.138)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = Vector(min(corner[axis] for corner in corners) for axis in range(3))
    upper = Vector(max(corner[axis] for corner in corners) for axis in range(3))
    return lower, upper


def main() -> None:
    args = parse_args()
    for path in (args.blend_output, args.glb_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("The source GLB did not contain a mesh")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    cup = bpy.context.view_layer.objects.active
    diameter_label = f"{args.target_top_diameter * 100:g}".replace(".", "_")
    cup.name = f"Mupama_Cup_Single_{diameter_label}cm_Top"
    cup.data.name = "Mupama_Cup_Single_Mesh"

    source_lower, source_upper = bounds(cup)
    source_dimensions = source_upper - source_lower
    cup.location += Vector(
        (
            -(source_lower.x + source_upper.x) / 2,
            -(source_lower.y + source_upper.y) / 2,
            -source_lower.z,
        )
    )
    uniform_scale = args.target_top_diameter / args.source_top_diameter
    # glTF import may preserve a source-unit conversion in the object's scale
    # (this asset imports its millimeter mesh with scale 0.001). Multiply that
    # existing transform instead of replacing it.
    cup.scale = tuple(component * uniform_scale for component in cup.scale)
    bpy.context.view_layer.objects.active = cup
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    cup["asset_role"] = "motion_validation_prop"
    cup["source_top_diameter_m"] = args.source_top_diameter
    cup["target_top_diameter_m"] = args.target_top_diameter
    cup["applied_uniform_scale"] = uniform_scale

    final_lower, final_upper = bounds(cup)
    final_dimensions = final_upper - final_lower
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output.resolve()))
    bpy.ops.export_scene.gltf(
        filepath=str(args.glb_output.resolve()),
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_yup=True,
    )

    report = {
        "source": str(args.source.resolve()),
        "blend": str(args.blend_output.resolve()),
        "glb": str(args.glb_output.resolve()),
        "object": cup.name,
        "source_dimensions_m": [round(value, 6) for value in source_dimensions],
        "source_top_diameter_m": args.source_top_diameter,
        "target_top_diameter_m": args.target_top_diameter,
        "applied_uniform_scale": round(uniform_scale, 9),
        "final_dimensions_m": [round(value, 6) for value in final_dimensions],
        "bounds_min": [round(value, 6) for value in final_lower],
        "bounds_max": [round(value, 6) for value in final_upper],
        "texture_packed": all(
            image.packed_file is not None
            for image in bpy.data.images
            if image.source == "FILE"
        ),
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
