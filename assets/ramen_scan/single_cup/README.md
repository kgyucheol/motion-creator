# Single ramen cup asset

The complete single-cup model package copied from
`/home/kim/Documents/KakaoTalk Downloads/mupama_cup_photo_model` is stored
locally in `source/`.

## Primary files

- `source/mupama_cup.glb`: self-contained textured model in meter units
- `source/mupama_cup.obj`, `source/mupama_cup.mtl`, and
  `source/mupama_wrap_photo.png`: editable OBJ asset in millimeter units
- `source/mupama_cup.stl`: untextured geometry for 3D printing in millimeters
- `source/mupama_preview.png`: source preview
- `source/build_model.py`, `source/photo_texture.py`, and
  `source/make_preview.py`: model and texture regeneration scripts
- `source/source_photos/`: four photographs used for the wrap texture
- `source/photo_texture_sources.json`: photo projection metadata

The source package remains ignored by Git because it contains binary models,
textures, and original photographs. This inventory file is tracked.

## Folder layout

- `source/`: original model package and source photographs
- `individual/`: scaled single cup, one mesh object
- `stack_30_objects/`: 30 independently editable cups, with varied label angles
- `stack_1_object/`: the same 30-cup arrangement joined into one mesh object

Each derived folder contains a BLEND file, a GLB file, and a JSON build report.

## Measured working asset

`individual/mupama_cup_13_8cm_top.blend` and its GLB are derived working assets.
The source model's 140 mm top diameter is uniformly scaled to the
measured 138 mm diameter. The same `138 / 140` scale is applied to X, Y, and Z,
preserving the cup's proportions.

```bash
blender --background --python scripts/blender/prepare_single_cup_asset.py -- \
  --source assets/ramen_scan/single_cup/source/mupama_cup.glb \
  --blend-output assets/ramen_scan/single_cup/individual/mupama_cup_13_8cm_top.blend \
  --glb-output assets/ramen_scan/single_cup/individual/mupama_cup_13_8cm_top.glb \
  --source-top-diameter 0.14 \
  --target-top-diameter 0.138 \
  --report assets/ramen_scan/single_cup/individual/mupama_cup_13_8cm_top_report.json
```

## 30-cup stack

`stack_30_objects/mupama_cups_30_stack_70cm.blend` and its GLB contain
30 copies of the measured single cup, all facing upward. The first cup starts
at Z=0. The center-to-center vertical pitch is calculated as
`(0.70 m - one cup height) / 29`, making the total stack height 70 cm.
The copies share one mesh in the Blender file. No collision simulation is used.
Each cup has its own Z-axis rotation so the printed labels face different
directions. A fixed random seed makes the arrangement reproducible; adjacent
cups differ by at least 25 degrees.

```bash
blender --background --python scripts/blender/stack_single_cups.py -- \
  --source assets/ramen_scan/single_cup/individual/mupama_cup_13_8cm_top.blend \
  --blend-output assets/ramen_scan/single_cup/stack_30_objects/mupama_cups_30_stack_70cm.blend \
  --glb-output assets/ramen_scan/single_cup/stack_30_objects/mupama_cups_30_stack_70cm.glb \
  --report assets/ramen_scan/single_cup/stack_30_objects/mupama_cups_30_stack_70cm_report.json \
  --count 30 --height 0.70 --seed 20260929
```

## 30-cup stack as one object

`stack_1_object/mupama_cups_30_stack_70cm_merged.blend` and its GLB preserve
the exact cup positions, textures, and varied label directions from the editable
30-object version. All geometry is joined into one mesh with its pivot at the
bottom center, identity object transform, and the same 70 cm overall height.

```bash
blender --background --python scripts/blender/merge_cup_stack.py -- \
  --source assets/ramen_scan/single_cup/stack_30_objects/mupama_cups_30_stack_70cm.blend \
  --blend-output assets/ramen_scan/single_cup/stack_1_object/mupama_cups_30_stack_70cm_merged.blend \
  --glb-output assets/ramen_scan/single_cup/stack_1_object/mupama_cups_30_stack_70cm_merged.glb \
  --report assets/ramen_scan/single_cup/stack_1_object/mupama_cups_30_stack_70cm_merged_report.json
```
