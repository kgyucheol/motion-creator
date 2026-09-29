#!/usr/bin/env python3
"""Place twelve merged 30-cup stacks in the measured open-top box."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


INNER_WIDTH = 0.72
INNER_LENGTH = 0.55
INNER_HEIGHT = 0.40
WALL = 0.02
ROWS = 4
LAYERS = 3
ROLL_SEED = 20260929
FINAL_Z_ROTATION_DEGREES = 270


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


def mesh_bounds(obj: bpy.types.Object, local_vertices: np.ndarray) -> tuple[Vector, Vector]:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    world_vertices = local_vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return Vector(world_vertices.min(axis=0)), Vector(world_vertices.max(axis=0))


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

    rng = random.Random(ROLL_SEED)
    stacks: list[bpy.types.Object] = []
    roll_angles_degrees: list[float] = []
    for layer, z_center in enumerate(z_centers, start=1):
        for row, y_center in enumerate(y_centers, start=1):
            stack = template if not stacks else template.copy()
            if stacks:
                stack.data = template.data
                stack_collection.objects.link(stack)
            stack.name = f"Mupama_Stack_L{layer:02d}_R{row:02d}"
            roll = rng.uniform(0.0, 2.0 * math.pi)
            stack.matrix_world = (
                Matrix.Translation((-stack_length / 2, y_center, z_center))
                @ Matrix.Rotation(roll, 4, "X")
                @ Matrix.Rotation(math.pi / 2, 4, "Y")
            )
            stack["cup_count"] = 30
            stack["packing_layer"] = layer
            stack["packing_row"] = row
            stack["x_axis_roll_degrees"] = math.degrees(roll)
            stacks.append(stack)
            roll_angles_degrees.append(math.degrees(roll))

    bpy.context.view_layer.update()
    interior_lower = Vector((-INNER_WIDTH / 2, -INNER_LENGTH / 2, WALL))
    interior_upper = Vector((INNER_WIDTH / 2, INNER_LENGTH / 2, WALL + INNER_HEIGHT))
    vertex_data = np.empty(len(template.data.vertices) * 3, dtype=np.float32)
    template.data.vertices.foreach_get("co", vertex_data)
    local_vertices = vertex_data.reshape(-1, 3)
    stack_bounds = [mesh_bounds(stack, local_vertices) for stack in stacks]
    if any(
        lower[axis] < interior_lower[axis] - 1e-5
        or upper[axis] > interior_upper[axis] + 1e-5
        for lower, upper in stack_bounds
        for axis in range(3)
    ):
        raise RuntimeError(f"A stack extends beyond the box interior: {stack_bounds}")

    final_rotation = Matrix.Rotation(math.radians(FINAL_Z_ROTATION_DEGREES), 4, "Z")
    for obj in (box, *stacks):
        obj.matrix_world = final_rotation @ obj.matrix_world
    bpy.context.view_layer.update()

    scene["asset_role"] = "motion_validation_packing"
    scene["stack_count"] = len(stacks)
    scene["cups_per_stack"] = 30
    scene["final_z_rotation_degrees"] = FINAL_Z_ROTATION_DEGREES
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
        "x_axis_roll_seed": ROLL_SEED,
        "x_axis_roll_degrees": [round(value, 3) for value in roll_angles_degrees],
        "final_z_rotation_degrees": FINAL_Z_ROTATION_DEGREES,
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
