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

## Measured working asset

`mupama_cup_13_8cm_top.blend` and `mupama_cup_13_8cm_top.glb` are derived working
assets. The source model's 140 mm top diameter is uniformly scaled to the
measured 138 mm diameter. The same `138 / 140` scale is applied to X, Y, and Z,
preserving the cup's proportions.

```bash
blender --background --python scripts/blender/prepare_single_cup_asset.py -- \
  --source assets/ramen_scan/single_cup/source/mupama_cup.glb \
  --blend-output assets/ramen_scan/single_cup/mupama_cup_13_8cm_top.blend \
  --glb-output assets/ramen_scan/single_cup/mupama_cup_13_8cm_top.glb \
  --source-top-diameter 0.14 \
  --target-top-diameter 0.138 \
  --report assets/ramen_scan/single_cup/mupama_cup_13_8cm_top_report.json
```

## 30-cup stack

`mupama_cups_30_stack_70cm.blend` and `mupama_cups_30_stack_70cm.glb` contain
30 copies of the measured single cup, all facing upward. The first cup starts
at Z=0. The center-to-center vertical pitch is calculated as
`(0.70 m - one cup height) / 29`, making the total stack height 70 cm.
The copies share one mesh in the Blender file. No collision simulation is used.

```bash
blender --background --python scripts/blender/stack_single_cups.py -- \
  --source assets/ramen_scan/single_cup/mupama_cup_13_8cm_top.blend \
  --blend-output assets/ramen_scan/single_cup/mupama_cups_30_stack_70cm.blend \
  --glb-output assets/ramen_scan/single_cup/mupama_cups_30_stack_70cm.glb \
  --report assets/ramen_scan/single_cup/mupama_cups_30_stack_70cm_report.json \
  --count 30 --height 0.70
```
