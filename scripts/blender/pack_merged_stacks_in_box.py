#!/usr/bin/env python3
"""Place twelve merged 30-cup stacks in the measured open-top box."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


INNER_WIDTH = 0.72
INNER_LENGTH = 0.55
INNER_HEIGHT = 0.40
WALL = 0.02
ROWS = 4
LAYERS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--box", type=Path, required=True)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--blend-output", type=Path, required=True)
    parser.add_argument("--glb-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def import_one_mesh(path: Path) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    added = set(bpy.context.scene.objects) - before
    meshes = [obj for obj in added if obj.type == "MESH"]
    if len(added) != 1 or len(meshes) != 1:
        raise RuntimeError(f"Expected exactly one mesh object in {path}")
    return meshes[0]


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = Vector(min(corner[axis] for corner in corners) for axis in range(3))
    upper = Vector(max(corner[axis] for corner in corners) for axis in range(3))
    return lower, upper


def centers(span: float, diameter: float, count: int) -> list[float]:
    first = -(span - diameter) / 2
    pitch = (span - diameter) / (count - 1)
    return [first + index * pitch for index in range(count)]


def main() -> None:
    args = parse_args()
    for path in (args.blend_output, args.glb_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.cursor.location = (0.0, 0.0, 0.0)

    box = import_one_mesh(args.box)
    box.name = "Cardboard_Box_72x55x40cm"
    box_collection = bpy.data.collections.new("Box")
    scene.collection.children.link(box_collection)
    move_to_collection(box, box_collection)
    box_lower, box_upper = bounds(box)
    box_dimensions = box_upper - box_lower
    expected_outer = Vector((INNER_WIDTH + 2 * WALL, INNER_LENGTH + 2 * WALL, INNER_HEIGHT + WALL))
    if any(abs(box_dimensions[i] - expected_outer[i]) > 1e-5 for i in range(3)):
        raise RuntimeError(f"Unexpected box size: {tuple(box_dimensions)}")

    template = import_one_mesh(args.stack)
    stack_dimensions = Vector(template.dimensions)
    row_diameter = stack_dimensions.y
    layer_diameter = stack_dimensions.x
    stack_length = stack_dimensions.z
    if stack_length > INNER_WIDTH + 1e-6:
        raise RuntimeError("The merged stack is longer than the box interior")

    stack_collection = bpy.data.collections.new("Merged_30_Cup_Stacks")
    scene.collection.children.link(stack_collection)
    move_to_collection(template, stack_collection)

    y_centers = centers(INNER_LENGTH, row_diameter, ROWS)
    z_min_center = WALL + layer_diameter / 2
    z_pitch = (INNER_HEIGHT - layer_diameter) / (LAYERS - 1)
    z_centers = [z_min_center + layer * z_pitch for layer in range(LAYERS)]

    stacks: list[bpy.types.Object] = []
    for layer, z_center in enumerate(z_centers, start=1):
        for row, y_center in enumerate(y_centers, start=1):
            stack = template if not stacks else template.copy()
            if stacks:
                stack.data = template.data
                stack_collection.objects.link(stack)
            stack.name = f"Mupama_Stack_L{layer:02d}_R{row:02d}"
            stack.rotation_mode = "XYZ"
            stack.rotation_euler = (0.0, math.pi / 2, 0.0)
            stack.location = (-stack_length / 2, y_center, z_center)
            stack["cup_count"] = 30
            stack["packing_layer"] = layer
            stack["packing_row"] = row
            stacks.append(stack)

    bpy.context.view_layer.update()
    interior_lower = Vector((-INNER_WIDTH / 2, -INNER_LENGTH / 2, WALL))
    interior_upper = Vector((INNER_WIDTH / 2, INNER_LENGTH / 2, WALL + INNER_HEIGHT))
    stack_bounds = [bounds(stack) for stack in stacks]
    if any(
        lower[axis] < interior_lower[axis] - 1e-5
        or upper[axis] > interior_upper[axis] + 1e-5
        for lower, upper in stack_bounds
        for axis in range(3)
    ):
        raise RuntimeError("A stack extends beyond the box interior")

    scene["asset_role"] = "motion_validation_packing"
    scene["stack_count"] = len(stacks)
    scene["cups_per_stack"] = 30
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output.resolve()))
    bpy.ops.export_scene.gltf(
        filepath=str(args.glb_output.resolve()),
        export_format="GLB",
        use_selection=False,
        export_yup=True,
    )

    report = {
        "box_source": str(args.box.resolve()),
        "merged_stack_source": str(args.stack.resolve()),
        "blend": str(args.blend_output.resolve()),
        "glb": str(args.glb_output.resolve()),
        "box_internal_dimensions_m": [INNER_WIDTH, INNER_LENGTH, INNER_HEIGHT],
        "box_outer_dimensions_m": [round(value, 6) for value in box_dimensions],
        "stack_dimensions_m": [round(value, 6) for value in stack_dimensions],
        "rows": ROWS,
        "layers": LAYERS,
        "stack_count": len(stacks),
        "total_cup_count": len(stacks) * 30,
        "scene_objects": [obj.name for obj in scene.objects],
        "row_overlap_m": round(max(0.0, ROWS * row_diameter - INNER_LENGTH), 6),
        "layer_overlap_m": round(max(0.0, LAYERS * layer_diameter - INNER_HEIGHT), 6),
        "stack_bounds_m": [
            {
                "min": [round(value, 6) for value in lower],
                "max": [round(value, 6) for value in upper],
            }
            for lower, upper in stack_bounds
        ],
        "packed_image_count": sum(bool(image.packed_file) for image in bpy.data.images),
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
