# Ramen container scan

`ramen_container_scan.blend` is the local editable Blender working file prepared from:

`assets/ramen_scan/source/3DModel.obj`

The scan is a vertical stack of ramen containers rather than one isolated cup.
The working file keeps the original scan geometry and UVs and packs the 2K
source texture. Its pivot is the ground-contact center at the world origin. The
scene contains only `Ramen_Container_Scan`; there are no cameras, lights, ground
planes, or helper objects, so it can be appended directly into motion-validation
scenes.

The raw scan folder and generated `.blend`/report files are intentionally ignored
by Git. Only this note and the reproducible preparation script are tracked.

## Regenerate

```bash
blender --background --python scripts/blender/prepare_ramen_scan.py -- \
  --source assets/ramen_scan/source/3DModel.obj \
  --output assets/ramen_scan/ramen_container_scan.blend \
  --report assets/ramen_scan/ramen_container_scan_report.json
```

The source dimensions reported by Blender are approximately 0.207 x 0.229 x
0.914 Blender units. The scene uses Blender's default metric interpretation, so
confirm one real-world measurement before making dimension-critical changes.
