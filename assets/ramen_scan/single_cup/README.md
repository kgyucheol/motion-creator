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
