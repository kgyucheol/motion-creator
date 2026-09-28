# Ramen container scan

`ramen_container_scan.blend` is the local editable Blender working file prepared from:

`assets/ramen_scan/source/3DModel.obj`

The scan is a vertical stack of ramen containers rather than one isolated cup.
The working file keeps the original scan geometry and UVs and packs the 2K
source texture. A single uniform XYZ scale is chosen so the mean of the scanned
X/Y diameters is approximately `0.14 m`; the scan's proportions are not forced
to an exact `0.14 x 0.14 x 0.65 m` box. Its pivot is the ground-contact center at
the world origin. The
scene contains only `Ramen_Container_Scan`; there are no cameras, lights, ground
planes, or helper objects, so it can be appended directly into motion-validation
scenes.

`cardboard_box_72x55x40cm.blend` contains one simple open-top cardboard box. Its
internal dimensions are `0.72 x 0.55 x 0.40 m`, its wall/bottom thickness is
`0.02 m`, and its external dimensions are `0.76 x 0.59 x 0.42 m`.

`ramen_box_packing_4x3.blend` combines the box with 12 linked copies of the ramen
bundle. The bundles lie along the box's 72 cm axis and are arranged four per
layer over three layers. After uniform scaling, the bundle bounds are about
`0.133 x 0.147 x 0.587 m`. Four row-axis diameters total about 58.8 cm, so their
centers are evenly distributed within the 55 cm internal length with slight
overlap. Three layer-axis diameters total about 39.9 cm and fit the 40 cm internal
height. Collision response is intentionally not configured in this first layout.

Each working file also has a self-contained GLB export for direct use by the
motion-validation project:

- `ramen_container_scan.glb`
- `cardboard_box_72x55x40cm.glb`
- `ramen_box_packing_4x3.glb`

The raw scan folder and generated `.blend`/report files are intentionally ignored
by Git. Only this note and the reproducible preparation script are tracked.

## Regenerate

```bash
blender --background --python scripts/blender/prepare_ramen_scan.py -- \
  --source assets/ramen_scan/source/3DModel.obj \
  --output assets/ramen_scan/ramen_container_scan.blend \
  --target-diameter 0.14 \
  --report assets/ramen_scan/ramen_container_scan_report.json
```

```bash
blender --background --python scripts/blender/build_ramen_packaging_assets.py -- \
  --bundle assets/ramen_scan/ramen_container_scan.blend \
  --box-output assets/ramen_scan/cardboard_box_72x55x40cm.blend \
  --packing-output assets/ramen_scan/ramen_box_packing_4x3.blend \
  --report assets/ramen_scan/ramen_packaging_assets_report.json
```

```bash
blender --background --python scripts/blender/export_blend_assets_glb.py -- \
  --asset assets/ramen_scan/ramen_container_scan.blend assets/ramen_scan/ramen_container_scan.glb \
  --asset assets/ramen_scan/cardboard_box_72x55x40cm.blend assets/ramen_scan/cardboard_box_72x55x40cm.glb \
  --asset assets/ramen_scan/ramen_box_packing_4x3.blend assets/ramen_scan/ramen_box_packing_4x3.glb \
  --report assets/ramen_scan/glb_export_report.json
```

The raw scan, generated Blender files, reports, and previews are local assets and
remain ignored by Git. The preparation scripts and this documentation are tracked.
