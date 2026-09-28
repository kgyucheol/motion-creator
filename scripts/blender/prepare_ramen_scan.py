#!/usr/bin/env python3
"""Import the ramen-container scan for motion-validation scene assembly.

Run with Blender, for example:
    blender --background --python scripts/blender/prepare_ramen_scan.py -- \
        --source assets/ramen_scan/source/3DModel.obj \
        --output assets/ramen_scan/ramen_container_scan.blend \
        --target-diameter 0.14 \
        --report assets/ramen_scan/ramen_container_scan_report.json
"""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-diameter", type=float, default=0.14)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_scan(source: Path) -> list[bpy.types.Object]:
    suffix = source.suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source), forward_axis="NEGATIVE_Z", up_axis="Y")
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(source))
    else:
        raise ValueError(f"Unsupported scan format: {suffix}")
    return [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]


def join_meshes(meshes: list[bpy.types.Object]) -> bpy.types.Object:
    if not meshes:
        raise RuntimeError("The imported file did not contain a mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    scan = bpy.context.view_layer.objects.active
    scan.name = "Ramen_Container_Scan"
    scan.data.name = "Ramen_Container_Scan_Mesh"
    return scan


def world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector(min(point[axis] for point in corners) for axis in range(3)),
        Vector(max(point[axis] for point in corners) for axis in range(3)),
    )


def put_pivot_at_ground_center(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    lower, upper = world_bounds(obj)
    obj.location += Vector((-(lower.x + upper.x) / 2, -(lower.y + upper.y) / 2, -lower.z))
    bpy.context.view_layer.update()

    # Bake the centering offset into the mesh. This leaves the object origin at
    # the ground-contact center and gives downstream motion-validation scenes a
    # predictable transform: location/rotation 0, scale 1.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    return world_bounds(obj)


def clean_normals(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def scale_uniformly_to_diameter(obj: bpy.types.Object, target_diameter: float) -> float:
    lower, upper = world_bounds(obj)
    current = upper - lower
    mean_horizontal_diameter = (current.x + current.y) / 2
    factor = target_diameter / mean_horizontal_diameter
    obj.scale = (factor, factor, factor)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return factor


def main() -> None:
    args = parse_args()
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    meshes = import_scan(args.source.resolve())
    scan = join_meshes(meshes)
    lower, upper = put_pivot_at_ground_center(scan)
    source_dimensions = upper - lower
    applied_uniform_scale = scale_uniformly_to_diameter(scan, args.target_diameter)
    lower, upper = world_bounds(scan)
    dimensions = upper - lower
    clean_normals(scan)
    scan["asset_role"] = "motion_validation_prop"
    scan["source_scan"] = str(args.source.resolve())
    scan["scaling_priority"] = "0.14m_x_0.14m_cross_section"

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"

    source_vertices = len(scan.data.vertices)
    source_edges = len(scan.data.edges)
    source_polygons = len(scan.data.polygons)
    uv_layers = len(scan.data.uv_layers)
    material_names = [material.name for material in scan.data.materials if material]

    # Keep the source texture self-contained in the working file.
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report = {
        "source": str(args.source.resolve()),
        "blend": str(args.output.resolve()),
        "object": scan.name,
        "scene_objects": [obj.name for obj in bpy.context.scene.objects],
        "location": [round(value, 6) for value in scan.location],
        "rotation_euler": [round(value, 6) for value in scan.rotation_euler],
        "scale": [round(value, 6) for value in scan.scale],
        "vertices": source_vertices,
        "edges": source_edges,
        "polygons": source_polygons,
        "uv_layers": uv_layers,
        "materials": material_names,
        "source_dimensions": [round(value, 6) for value in source_dimensions],
        "target_mean_horizontal_diameter": round(args.target_diameter, 6),
        "scaling_priority": "x_y_cross_section",
        "applied_uniform_scale": round(applied_uniform_scale, 6),
        "bounds_min": [round(value, 6) for value in lower],
        "bounds_max": [round(value, 6) for value in upper],
        "dimensions": [round(value, 6) for value in dimensions],
        "texture_packed": all(image.packed_file is not None for image in bpy.data.images if image.source == "FILE"),
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
