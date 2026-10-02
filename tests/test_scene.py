import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from ramen_controller.contracts import JOINT_NAMES                     # noqa: E402
from ramen_controller.scene import CAMERA_NAME, CameraSpec, build_scene  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "g1_tools"
pytestmark = pytest.mark.skipif(not (ASSETS / "g1_tools.xml").exists(), reason="robot assets not exported")


def environment(tmp_path, objects):
    path = tmp_path / "environment.json"
    path.write_text(json.dumps({"scene_objects": objects}))
    return path


def item(name, shape="box", size=(0.2, 0.2, 0.2), position=(0.6, 0, 0.1), fixed=False, **extra):
    return {"id": name, "name": name, "shape": shape, "size": list(size), "position": list(position),
            "quaternion_xyzw": [0, 0, 0, 1], "mass_kg": 1.0, "friction": 0.7, "fixed": fixed, **extra}


def load(path):
    return mujoco.MjModel.from_xml_path(str(path))


def test_robot_only_scene_compiles_with_camera_on_the_head(tmp_path):
    scene = build_scene(ASSETS, environment(tmp_path, []), tmp_path / "out")
    model = load(scene)
    assert model.nu == 29 and model.nq == 36 and model.ncam == 1
    camera = model.camera(CAMERA_NAME)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, camera.bodyid[0]) == "d435_link"
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(29)] == list(JOINT_NAMES)


def test_dynamic_objects_get_free_joints_and_fixed_ones_do_not(tmp_path):
    objects = [item("cube"), item("table", fixed=True, size=(1, 1, 0.3), position=(0, 1, 0.15)),
               item("can", shape="cylinder", size=(0.1, 0.1, 0.2)), item("ball", shape="sphere", size=(0.1, 0.1, 0.1))]
    scene = build_scene(ASSETS, environment(tmp_path, objects), tmp_path / "out")
    model = load(scene)
    assert model.nq == 36 + 7 * 3 and model.nv == 35 + 6 * 3
    sidecar = json.loads((tmp_path / "out" / "scene_objects.json").read_text())
    assert [o["fixed"] for o in sidecar["objects"]] == [False, True, False, False]


def test_open_box_becomes_five_panels_with_conserved_mass(tmp_path):
    scene = build_scene(ASSETS, environment(tmp_path, [item("carton", shape="open_box", size=(0.6, 0.8, 0.4), fixed=True)]),
                        tmp_path / "out")
    model = load(scene)
    geoms = [g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("obj0_geom")]
    assert len(geoms) == 5
    assert np.isclose(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj0")], 1.0)


def test_convex_hull_object_and_scale(tmp_path):
    cube = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    hull = item("hull", shape="box", size=(0.2, 0.3, 0.4), collision_shape="convex_hull",
                collision_hull_vertices=cube.tolist(), collision_hull_faces=faces)
    model = load(build_scene(ASSETS, environment(tmp_path, [hull]), tmp_path / "out"))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj0_geom")
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH


def test_dropped_object_settles_on_the_floor(tmp_path):
    model = load(build_scene(ASSETS, environment(tmp_path, [item("cube", position=(0.8, 0, 0.5))]), tmp_path / "out"))
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    z = data.qpos[36 + 2]
    assert 0.08 < z < 0.12                                    # half of the 0.2 m cube, resting on the floor


def test_camera_intrinsics_and_unknown_target_or_mount(tmp_path):
    K = CameraSpec().intrinsics()
    assert np.isclose(K[0, 0], K[1, 1]) and K[0, 2] == 320 and K[1, 2] == 240
    assert np.isclose(2 * np.degrees(np.arctan(240 / K[1, 1])), 42.0)
    with pytest.raises(ValueError, match="nothing"):
        build_scene(ASSETS, environment(tmp_path, [item("cube")]), tmp_path / "a", target_name="nothing")
    with pytest.raises(ValueError, match="no body"):
        build_scene(ASSETS, environment(tmp_path, []), tmp_path / "b", camera=CameraSpec(body="missing_link"))


def test_target_flag_lands_in_the_sidecar(tmp_path):
    build_scene(ASSETS, environment(tmp_path, [item("a"), item("b", position=(0.9, 0, 0.1))]), tmp_path / "out", target_name="b")
    objects = json.loads((tmp_path / "out" / "scene_objects.json").read_text())["objects"]
    assert [o["is_target"] for o in objects] == [False, True]


def test_robot_starts_at_the_requested_pose_and_yaw(tmp_path):
    model = load(build_scene(ASSETS, environment(tmp_path, []), tmp_path / "out", robot_start_xy_yaw=(-1.2, 0.3, np.pi / 2)))
    data = mujoco.MjData(model)
    assert np.allclose(data.qpos[:2], [-1.2, 0.3]) and np.isclose(data.qpos[2], 0.79)
    assert np.allclose(data.qpos[3:7], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    sidecar = json.loads((tmp_path / "out" / "scene_objects.json").read_text())
    assert sidecar["robot_start_xy_yaw"] == [-1.2, 0.3, np.pi / 2]
    default = load(build_scene(ASSETS, environment(tmp_path, []), tmp_path / "default"))
    assert np.allclose(mujoco.MjData(default).qpos[:2], [-1.0, 0.0])


def test_sidecar_lists_the_tool_bodies_for_grip_detection(tmp_path):
    build_scene(ASSETS, environment(tmp_path, []), tmp_path / "out")
    sidecar = json.loads((tmp_path / "out" / "scene_objects.json").read_text())
    assert set(sidecar["tool_bodies"]) == {"left", "right"}
    model = load(tmp_path / "out" / "scene.xml")
    for link in sidecar["tool_bodies"].values():
        assert model.body(link).id > 0                                  # the named bodies exist in the compiled model


def test_joints_and_floor_carry_the_stock_dynamics_the_policy_was_trained_on(tmp_path):
    model = load(build_scene(ASSETS, environment(tmp_path, []), tmp_path / "out"))
    for name in JOINT_NAMES:
        joint = model.joint(name)
        assert np.isclose(joint.damping[0], 0.05), name
        assert np.isclose(joint.frictionloss[0], 0.1 if "wrist" in name else 0.2), name
    assert np.allclose(model.geom_friction[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")], [1.0, 0.005, 0.0001])
