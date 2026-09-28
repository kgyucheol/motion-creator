#!/usr/bin/env python3
"""Build the measured box and a 4-by-3 ramen-bundle packing scene."""

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
WALL_THICKNESS = 0.02
ROW_COUNT = 4
LAYER_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--box-output", type=Path, required=True)
    parser.add_argument("--packing-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.cursor.location = (0.0, 0.0, 0.0)


def create_cardboard_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Cardboard")
    material.use_nodes = True
    material.diffuse_color = (0.36, 0.19, 0.075, 1.0)
    material.roughness = 0.82
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.36, 0.19, 0.075, 1.0)
        principled.inputs["Roughness"].default_value = 0.82
    return material


def add_slab(
    name: str,
    dimensions: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    slab = bpy.context.object
    slab.name = name
    slab.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    slab.data.materials.append(material)
    return slab


def build_box() -> bpy.types.Object:
    outer_width = INNER_WIDTH + 2 * WALL_THICKNESS
    outer_length = INNER_LENGTH + 2 * WALL_THICKNESS
    wall_center_z = WALL_THICKNESS + INNER_HEIGHT / 2
    material = create_cardboard_material()

    parts = [
        add_slab(
            "Box_Bottom",
            (outer_width, outer_length, WALL_THICKNESS),
            (0.0, 0.0, WALL_THICKNESS / 2),
            material,
        ),
        add_slab(
            "Box_Left",
            (WALL_THICKNESS, INNER_LENGTH, INNER_HEIGHT),
            (-(INNER_WIDTH + WALL_THICKNESS) / 2, 0.0, wall_center_z),
            material,
        ),
        add_slab(
            "Box_Right",
            (WALL_THICKNESS, INNER_LENGTH, INNER_HEIGHT),
            ((INNER_WIDTH + WALL_THICKNESS) / 2, 0.0, wall_center_z),
            material,
        ),
        add_slab(
            "Box_Front",
            (outer_width, WALL_THICKNESS, INNER_HEIGHT),
            (0.0, -(INNER_LENGTH + WALL_THICKNESS) / 2, wall_center_z),
            material,
        ),
        add_slab(
            "Box_Back",
            (outer_width, WALL_THICKNESS, INNER_HEIGHT),
            (0.0, (INNER_LENGTH + WALL_THICKNESS) / 2, wall_center_z),
            material,
        ),
    ]

    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    box = bpy.context.object
    box.name = "Cardboard_Box_72x55x40cm"
    box.data.name = "Cardboard_Box_Mesh"
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
    box["asset_role"] = "motion_validation_container"
    box["internal_dimensions_m"] = [INNER_WIDTH, INNER_LENGTH, INNER_HEIGHT]
    box["wall_thickness_m"] = WALL_THICKNESS
    return box


def append_object(blend_path: Path, object_name: str) -> bpy.types.Object:
    with bpy.data.libraries.load(str(blend_path.resolve()), link=False) as (data_from, data_to):
        if object_name not in data_from.objects:
            raise RuntimeError(f"{object_name!r} not found in {blend_path}")
        data_to.objects = [object_name]
    obj = data_to.objects[0]
    bpy.context.scene.collection.objects.link(obj)
    return obj


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)


def evenly_spaced_centers(total: float, diameter: float, count: int) -> list[float]:
    if count == 1:
        return [0.0]
    first = -(total - diameter) / 2
    spacing = (total - diameter) / (count - 1)
    return [first + index * spacing for index in range(count)]


def build_packing_scene(
    bundle_path: Path, box_path: Path
) -> tuple[list[bpy.types.Object], Vector]:
    reset_scene()
    box_collection = bpy.data.collections.new("Box")
    bundle_collection = bpy.data.collections.new("Ramen_Bundles")
    bpy.context.scene.collection.children.link(box_collection)
    bpy.context.scene.collection.children.link(bundle_collection)

    box = append_object(box_path, "Cardboard_Box_72x55x40cm")
    move_to_collection(box, box_collection)
    template = append_object(bundle_path, "Ramen_Container_Scan")
    move_to_collection(template, bundle_collection)

    # Source axes are X/Y cross-section and Z bundle length. After rotating the
    # bundle +90 degrees around Y, those map to box Z/Y/X respectively.
    source_dimensions = Vector(template.dimensions)
    layer_diameter = source_dimensions.x
    row_diameter = source_dimensions.y
    bundle_length = source_dimensions.z

    y_centers = evenly_spaced_centers(INNER_LENGTH, row_diameter, ROW_COUNT)
    layer_bottom = WALL_THICKNESS + layer_diameter / 2
    layer_span = INNER_HEIGHT - layer_diameter
    z_centers = [
        layer_bottom + layer * layer_span / (LAYER_COUNT - 1)
        for layer in range(LAYER_COUNT)
    ]

    bundles: list[bpy.types.Object] = []
    index = 0
    for layer, z_center in enumerate(z_centers, start=1):
        for row, y_center in enumerate(y_centers, start=1):
            bundle = template if index == 0 else template.copy()
            if index > 0:
                bundle.data = template.data
                bundle_collection.objects.link(bundle)
            bundle.name = f"Ramen_Bundle_L{layer:02d}_R{row:02d}"
            bundle.rotation_euler = (0.0, math.radians(90.0), 0.0)
            # The source pivot is at one end of its long axis. After rotating
            # +90 degrees around Y, offset by half its length to center it in X.
            bundle.location = (-bundle_length / 2, y_center, z_center)
            bundle["packing_layer"] = layer
            bundle["packing_row"] = row
            bundles.append(bundle)
            index += 1

    bpy.context.view_layer.update()
    return [box, *bundles], source_dimensions


def object_bounds(obj: bpy.types.Object) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = [min(corner[axis] for corner in corners) for axis in range(3)]
    upper = [max(corner[axis] for corner in corners) for axis in range(3)]
    return lower, upper


def main() -> None:
    args = parse_args()
    for path in (args.box_output, args.packing_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    reset_scene()
    box = build_box()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.box_output.resolve()))
    box_lower, box_upper = object_bounds(box)

    objects, bundle_dimensions = build_packing_scene(
        args.bundle.resolve(), args.box_output.resolve()
    )
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.packing_output.resolve()))

    bundle_bounds = [object_bounds(obj) for obj in objects[1:]]
    report = {
        "box_blend": str(args.box_output.resolve()),
        "packing_blend": str(args.packing_output.resolve()),
        "box_internal_dimensions_m": [INNER_WIDTH, INNER_LENGTH, INNER_HEIGHT],
        "box_wall_thickness_m": WALL_THICKNESS,
        "box_outer_bounds": {
            "min": [round(value, 6) for value in box_lower],
            "max": [round(value, 6) for value in box_upper],
        },
        "bundle_dimensions_m": [round(value, 6) for value in bundle_dimensions],
        "packing": {"rows": ROW_COUNT, "layers": LAYER_COUNT, "count": ROW_COUNT * LAYER_COUNT},
        "scene_objects": [obj.name for obj in bpy.context.scene.objects],
        "bundle_bounds": [
            {
                "min": [round(value, 6) for value in lower],
                "max": [round(value, 6) for value in upper],
            }
            for lower, upper in bundle_bounds
        ],
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
