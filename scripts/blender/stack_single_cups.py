#!/usr/bin/env python3
"""Stack copies of a single cup to a specified overall height."""

from __future__ import annotations

import argparse
import json
import math
import random
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
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--height", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    # Rotating the corners of an axis-aligned bound box exaggerates the width
    # of a round cup. Measure the transformed mesh vertices instead.
    points = (obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices)
    first = next(points)
    lower = first.copy()
    upper = first.copy()
    for point in points:
        for axis in range(3):
            lower[axis] = min(lower[axis], point[axis])
            upper[axis] = max(upper[axis], point[axis])
    return lower, upper


def random_yaw_angles(count: int, seed: int) -> list[float]:
    """Give adjacent labels visibly different, reproducible Z rotations."""
    rng = random.Random(seed)
    angles: list[float] = []
    for _ in range(count):
        while True:
            angle = rng.uniform(0.0, 360.0)
            if not angles or abs((angle - angles[-1] + 180.0) % 360.0 - 180.0) >= 25.0:
                angles.append(angle)
                break
    return angles


def main() -> None:
    args = parse_args()
    if args.count < 2:
        raise ValueError("Count must be at least 2")
    for path in (args.blend_output, args.glb_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(args.source.resolve()))
    source_objects = list(bpy.context.scene.objects)
    if len(source_objects) != 1 or source_objects[0].type != "MESH":
        raise RuntimeError("Source blend must contain exactly one mesh object")
    cup = source_objects[0]
    source_lower, source_upper = bounds([cup])
    cup_height = source_upper.z - source_lower.z
    if args.height <= cup_height:
        raise ValueError("Requested stack height must exceed one cup height")

    pitch = (args.height - cup_height) / (args.count - 1)
    yaw_angles = random_yaw_angles(args.count, args.seed)
    cup.name = "Mupama_Cup_01"
    # GLB imports commonly use quaternion rotation mode. Euler angles have no
    # effect in that mode, so switch modes before setting each cup's yaw.
    cup.rotation_mode = "XYZ"
    cup.rotation_euler.z = math.radians(yaw_angles[0])
    cups = [cup]
    for index in range(1, args.count):
        instance = cup.copy()
        instance.data = cup.data
        instance.name = f"Mupama_Cup_{index + 1:02d}"
        instance.location = cup.location + Vector((0.0, 0.0, index * pitch))
        instance.rotation_euler.z = math.radians(yaw_angles[index])
        bpy.context.scene.collection.objects.link(instance)
        cups.append(instance)

    bpy.context.view_layer.update()
    stack_lower, stack_upper = bounds(cups)
    stack_dimensions = stack_upper - stack_lower
    if abs(stack_dimensions.z - args.height) > 1e-6:
        raise RuntimeError(f"Stack height mismatch: {stack_dimensions.z} m")

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene["asset_role"] = "motion_validation_prop"
    scene["cup_count"] = args.count
    scene["target_height_m"] = args.height
    scene["stack_pitch_m"] = pitch
    scene["rotation_seed"] = args.seed
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
        "cup_count": len(cups),
        "single_cup_height_m": round(cup_height, 9),
        "center_pitch_m": round(pitch, 9),
        "rotation_seed": args.seed,
        "yaw_degrees_by_cup": [round(angle, 3) for angle in yaw_angles],
        "stack_dimensions_m": [round(value, 9) for value in stack_dimensions],
        "bounds_min_m": [round(value, 9) for value in stack_lower],
        "bounds_max_m": [round(value, 9) for value in stack_upper],
        "objects": [obj.name for obj in cups],
        "packed_image_count": sum(bool(image.packed_file) for image in bpy.data.images),
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
