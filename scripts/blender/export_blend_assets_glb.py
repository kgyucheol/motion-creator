#!/usr/bin/env python3
"""Export one or more Blender asset files to self-contained GLB files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        action="append",
        nargs=2,
        required=True,
        metavar=("INPUT_BLEND", "OUTPUT_GLB"),
        help="Blend input and GLB output pair; repeat for multiple assets",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def main() -> None:
    args = parse_args()
    exports = []

    for input_name, output_name in args.asset:
        input_path = Path(input_name).resolve()
        output_path = Path(output_name).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bpy.ops.wm.open_mainfile(filepath=str(input_path))
        objects = list(bpy.context.scene.objects)
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            use_selection=False,
            export_apply=True,
            export_yup=True,
        )
        exports.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "objects": [obj.name for obj in objects],
                "mesh_count": sum(obj.type == "MESH" for obj in objects),
                "output_bytes": output_path.stat().st_size,
            }
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"exports": exports}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exports": exports}, indent=2))


if __name__ == "__main__":
    main()
