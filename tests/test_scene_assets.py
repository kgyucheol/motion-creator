import json
import struct

import numpy as np
import pytest
import trimesh
import mujoco

from motioncreator import scene_assets
from motioncreator.motion import new_project, validate_project
from motioncreator.robot import Robot
from motioncreator.scene_geometry import append_collision_geoms
from motioncreator.policy_preview import build_model
import xml.etree.ElementTree as ET


def test_glb_import_is_content_addressed_and_reports_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_assets, "ASSET_ROOT", tmp_path)
    # A standard glTF is +Y-up: X width, Y height, Z depth.
    mesh = trimesh.creation.box(extents=[.72, .4, .55])
    content = trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh))

    imported = scene_assets.import_scene_asset(content, "tray.glb")
    assert imported["asset_id"] == scene_assets.import_scene_asset(content, "tray.glb")["asset_id"]
    assert np.allclose(imported["dimensions"], [.72, .55, .4])
    assert np.allclose(imported["axis_transform_xyzw"], [2 ** -.5, 0, 0, 2 ** -.5])
    assert imported["suggestion"]["shape"] == "box"
    assert imported["suggestion"]["collision_shape"] == "convex_hull"
    assert len(imported["suggestion"]["collision_hull_vertices"]) == 8
    assert scene_assets.asset_path(imported["asset_id"]).read_bytes() == content
    assert json.loads((tmp_path / imported["asset_id"] / "metadata.json").read_text())["source_name"] == "tray.glb"

    robot = Robot()
    project = new_project(robot)
    project["scene_objects"] = [{
        "id": "legacy-glb", "name": "Legacy GLB", "shape": "box", "position": [0, 0, .2],
        "quaternion_xyzw": [0, 0, 0, 1], "size": [.72, .4, .55], "mass_kg": 1.,
        "friction": .7, "color": "#ffffff", "opacity": 1., "visible": True,
        "asset_id": imported["asset_id"], "asset_bounds_min": [-.36, -.2, -.275],
        "asset_bounds_max": [.36, .2, .275],
    }]
    migrated = validate_project(robot, project)["scene_objects"][0]
    assert np.allclose(migrated["size"], [.72, .55, .4])
    assert migrated["asset_axis_transform_xyzw"] == imported["axis_transform_xyzw"]


@pytest.mark.parametrize("filename, content", [
    ("bad.glb", b"not glb"),
    ("bad.blend", b"not blend"),
    ("bad.obj", b"anything"),
    ("../bad.glb", b"glTFanything"),
])
def test_asset_import_rejects_invalid_input(tmp_path, monkeypatch, filename, content):
    monkeypatch.setattr(scene_assets, "ASSET_ROOT", tmp_path)
    with pytest.raises(ValueError):
        scene_assets.import_scene_asset(content, filename)


def test_joined_cardboard_glb_remains_one_semantic_object(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_assets, "ASSET_ROOT", tmp_path)
    panels = []
    for extents, translation in [
        ([.76, .02, .59], [0, .01, 0]),
        ([.02, .4, .55], [-.37, .2, 0]), ([.02, .4, .55], [.37, .2, 0]),
        ([.76, .4, .02], [0, .2, -.285]), ([.76, .4, .02], [0, .2, .285]),
    ]:
        panel = trimesh.creation.box(extents=extents)
        panel.apply_translation(translation)
        panels.append(panel)
    joined = trimesh.util.concatenate(panels)
    content = trimesh.exchange.gltf.export_glb(trimesh.Scene(joined))

    imported = scene_assets.import_scene_asset(content, "cardboard_box_72x55x40cm.glb")
    assert "parts" not in imported
    assert imported["suggestion"]["shape"] == "open_box"
    assert imported["suggestion"]["fixed"]


def test_packing_glb_uses_thirteen_authored_nodes_and_preserves_texture(tmp_path, monkeypatch):
    source = scene_assets.ROOT / "assets/ramen_scan/ramen_box_packing_4x3.glb"
    monkeypatch.setattr(scene_assets, "ASSET_ROOT", tmp_path)

    imported = scene_assets.import_scene_asset(source.read_bytes(), source.name)

    assert len(imported["parts"]) == 13
    assert sum("Cardboard" in part["node_name"] for part in imported["parts"]) == 1
    assert sum("Ramen_Bundle" in part["node_name"] for part in imported["parts"]) == 12
    ramen = next(part for part in imported["parts"] if "Ramen_Bundle" in part["node_name"])
    assert ramen["suggestion"]["shape"] == "box"
    assert ramen["suggestion"]["collision_shape"] == "cylinder"
    assert ramen["suggestion"]["collision_size"] == pytest.approx([.1329772547, .1329772547, .5866761208])
    assert ramen["suggestion"]["collision_quaternion_xyzw"] == pytest.approx([0., 2 ** -.5, 0., 2 ** -.5])
    assert all(part["url"] == imported["url"] for part in imported["parts"])
    assert not (tmp_path / imported["asset_id"] / "parts").exists()
    # The original GLB is served intact; its embedded JPEG and material texture
    # reference are therefore not discarded by a per-node re-export.
    stored = (tmp_path / imported["asset_id"] / "model.glb").read_bytes()
    assert stored == source.read_bytes()
    json_length, _ = struct.unpack_from("<II", stored, 12)
    gltf = json.loads(stored[20:20 + json_length])
    assert gltf["images"]
    assert any("baseColorTexture" in material.get("pbrMetallicRoughness", {})
               for material in gltf["materials"])


def test_open_box_proxy_has_floor_and_four_walls():
    body = ET.Element("body")
    names = append_collision_geoms(body, "preview", 0, {
        "shape": "open_box", "size": [.76, .59, .42], "wall_thickness_m": .02,
        "mass_kg": 2., "friction": .7,
    })
    assert len(names) == 5
    assert {name.rsplit("_", 1)[-1] for name in names} == {"bottom", "left", "right", "front", "back"}
    assert sum(float(geom.get("mass")) for geom in body.findall("geom")) == pytest.approx(2.)


def test_separate_inscribed_cylinder_proxy_uses_its_local_axis():
    body = ET.Element("body")
    append_collision_geoms(body, "preview", 0, {
        "shape": "box", "size": [.586, .147, .133],
        "collision_shape": "cylinder", "collision_size": [.133, .133, .586],
        "collision_quaternion_xyzw": [0., 2 ** -.5, 0., 2 ** -.5],
        "mass_kg": .5, "friction": .7,
    })
    geom = body.find("geom")
    assert geom.get("type") == "cylinder"
    assert np.fromstring(geom.get("size"), sep=" ") == pytest.approx([.0665, .293])
    assert np.fromstring(geom.get("quat"), sep=" ") == pytest.approx([2 ** -.5, 0., 2 ** -.5, 0.])


def test_convex_hull_proxy_compiles_as_a_mujoco_mesh():
    robot = Robot()
    project = new_project(robot)
    project["scene_objects"] = [{
        "id": "hull", "name": "Hull", "shape": "box", "position": [1.5, 0., .2],
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.4, .3, .2], "mass_kg": 1.,
        "friction": .7, "color": "#ffffff", "opacity": 1., "visible": True,
        "collision_shape": "convex_hull",
        "collision_hull_vertices": [[-.5, -.5, -.5], [.5, -.5, -.5], [0., .5, -.5], [0., 0., .5]],
        "collision_hull_faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    }]
    model = build_model(robot, validate_project(robot, project))
    geom = model.geom("preview_object_geom_0")
    assert geom.type == mujoco.mjtGeom.mjGEOM_MESH
    mesh_id = int(np.asarray(geom.dataid).reshape(-1)[0])
    assert model.mesh(mesh_id).vertnum == 4


def test_project_validates_assets_and_rigid_scene_groups():
    robot = Robot()
    project = new_project(robot)
    project["scene_objects"] = [
        {"id": "box", "name": "Box", "shape": "open_box", "position": [0, 0, .21],
         "quaternion_xyzw": [0, 0, 0, 1], "size": [.76, .59, .42], "wall_thickness_m": .02,
         "mass_kg": 2., "friction": .7, "color": "#9b6b3c", "opacity": 1., "visible": True,
         "asset_id": "0123456789abcdef01234567", "asset_bounds_min": [-.38, -.295, 0],
         "asset_bounds_max": [.38, .295, .42]},
        {"id": "ramen", "name": "Ramen", "shape": "cylinder", "position": [0, 0, .3],
         "quaternion_xyzw": [0, 0, 0, 1], "size": [.14, .14, .6], "mass_kg": 1.,
         "friction": .7, "color": "#e8d8bd", "opacity": 1., "visible": True},
    ]
    project["scene_groups"] = [{"id": "packing", "name": "Packing", "member_ids": ["box", "ramen"],
                                "position": [0, 0, .3], "quaternion_xyzw": [0, 0, 0, 1]}]
    assert validate_project(robot, project)["scene_groups"][0]["member_ids"] == ["box", "ramen"]


def test_dynamic_ramen_proxy_settles_inside_fixed_open_box():
    robot = Robot()
    project = new_project(robot)
    placement = {"prevent_overlap": False, "surface_snap": False, "ground_lock": False}
    project["scene_objects"] = [
        {"id": "box", "name": "Box", "shape": "open_box", "position": [1.5, 0, .21],
         "quaternion_xyzw": [0, 0, 0, 1], "size": [.76, .59, .42], "wall_thickness_m": .02,
         "mass_kg": 2., "friction": .7, "color": "#9b6b3c", "opacity": 1., "visible": True,
         "fixed": True, "placement": placement},
        {"id": "ramen", "name": "Ramen", "shape": "cylinder", "position": [1.5, 0, .32],
         "quaternion_xyzw": [0, 0, 0, 1], "size": [.147, .147, .587], "mass_kg": 1.,
         "friction": .7, "color": "#e8d8bd", "opacity": 1., "visible": True, "placement": placement},
    ]
    model = build_model(robot, validate_project(robot, project))
    data = mujoco.MjData(model)
    data.qpos[:robot.model.nq] = robot.home
    for _ in range(1000):
        mujoco.mj_step(model, data)
    position = data.xpos[model.body("preview_object_1").id]
    assert np.allclose(position[:2], [1.5, 0], atol=.01)
    assert position[2] > .25
