import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

mujoco = pytest.importorskip("mujoco")
from ramen_controller.sim_feed import TruthExtractor, pack_message, unpack_message  # noqa: E402

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="pelvis" pos="-1 0 0.8" quat="0.7071068 0 0 0.7071068"><freejoint/><geom type="box" size=".05 .05 .05"/>
      <body name="head_mount" pos="0.1 0 0.5"><camera name="head" xyaxes="0 -1 0 0 0 1" fovy="42"/></body>
      <body name="left_tool" pos="0.5 0.105 0"><geom name="lt" type="box" size=".1 .01 .1"/></body>
      <body name="right_tool" pos="0.5 -0.105 0"><geom name="rt" type="box" size=".1 .01 .1"/></body>
    </body>
    <body name="obj0" pos="-1 0.5 0.8"><freejoint/><geom name="obj0_geom" type="box" size=".1 .1 .1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


def sidecar(tmp_path, target=True):
    path = tmp_path / "scene_objects.json"
    path.write_text(json.dumps({"camera": {"name": "head", "width": 640, "height": 480, "fovy_deg": 42,
                                           "K": [[615, 0, 320], [0, 615, 240], [0, 0, 1]], "body": "head_mount"},
                                "objects": [{"index": 0, "body": "obj0", "id": "a", "name": "can", "fixed": False,
                                             "geoms": ["obj0_geom"], "is_target": target}]}))
    return path


def setup(tmp_path, target=True):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    extractor = TruthExtractor(model, sidecar(tmp_path, target), {"left": "left_tool", "right": "right_tool"})
    return model, data, extractor


def test_object_pose_is_expressed_in_the_base_frame(tmp_path):
    _, data, ex = setup(tmp_path)
    # robot at world (-1,0,.8) yawed +90deg: base +x = world +y, base +y = world -x. Object is at world (-1,.5,.8).
    position, quat = ex.object_pose_in_base(data, "can")
    assert np.allclose(position, [0.5, 0.0, 0.0], atol=1e-9)             # 0.5 m straight ahead of the pelvis
    assert np.isclose(Rotation.from_quat(quat).as_euler("xyz", degrees=True)[2], -90, atol=1e-6)   # object yaw 0 = base yaw -90


def test_camera_pose_uses_the_opencv_optical_convention(tmp_path):
    _, data, ex = setup(tmp_path)
    T = ex.camera_pose_in_base(data)
    assert np.allclose(T[:3, 3], [0.1, 0, 0.5], atol=1e-9)
    # optical z (forward) = base +x, optical x (right) = base -y, optical y (down) = base -z
    assert np.allclose(T[:3, 2], [1, 0, 0], atol=1e-9)
    assert np.allclose(T[:3, 0], [0, -1, 0], atol=1e-9)
    assert np.allclose(T[:3, 1], [0, 0, -1], atol=1e-9)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def place(model, data, world_xyz):
    data.qpos[7:10] = world_xyz                                           # obj0 free joint follows the pelvis one
    mujoco.mj_forward(model, data)


def test_in_view_follows_geometry(tmp_path):
    model, data, ex = setup(tmp_path)
    assert not ex.in_view(data, "can")                                    # 0.5 m below a camera 0.4 m away: outside 42 deg fov
    place(model, data, [-1.0, 1.0, 1.0])                                  # base (1.0, 0, 0.2): ahead and slightly below
    assert ex.in_view(data, "can")
    place(model, data, [-1.0, -2.0, 0.8])                                 # behind the robot
    assert not ex.in_view(data, "can")
    place(model, data, [-4.0, 1.0, 1.0])                                  # base (1.0, 3.0, 0.2): far to the side
    assert not ex.in_view(data, "can")


def test_partly_visible_target_counts_only_within_the_margin(tmp_path):
    model, data, ex = setup(tmp_path)
    place(model, data, [-1.0, 1.0, 1.7])                                  # centre projects 33 px above the image top
    assert not ex.in_view(data, "can")
    ex.visible_margin_px = 20
    assert not ex.in_view(data, "can")
    ex.visible_margin_px = 60
    assert ex.in_view(data, "can")
    ex.visible_margin_px = 0
    place(model, data, [-1.0, -2.0, 0.8])                                 # behind the camera is never visible, whatever the margin
    ex.visible_margin_px = 1000
    assert not ex.in_view(data, "can")


def test_held_needs_both_tools_squeezing_the_target(tmp_path):
    model, data, ex = setup(tmp_path)
    truth = ex.truth(data)
    assert truth["grip_forces_n"]["left"] > 1 and truth["grip_forces_n"]["right"] > 1 and truth["held"]


def test_one_tool_alone_is_not_a_grip(tmp_path):
    model, data, ex = setup(tmp_path)
    model.body_pos[model.body("left_tool").id] = [0.5, 0.4, 0]            # pull the left plate away
    mujoco.mj_forward(model, data)
    truth = ex.truth(data)
    assert truth["grip_forces_n"]["left"] == 0 and truth["grip_forces_n"]["right"] > 1 and not truth["held"]


def test_contact_with_unrelated_geometry_is_ignored(tmp_path):
    model, data, ex = setup(tmp_path)
    model.body_pos[model.body("left_tool").id] = [0.5, 0.4, 0]
    model.body_pos[model.body("right_tool").id] = [0.5, -0.4, 0]
    mujoco.mj_forward(model, data)
    assert ex.truth(data)["grip_forces_n"] == {"left": 0.0, "right": 0.0}


def test_truth_message_roundtrip_and_depth_quantisation(tmp_path):
    pytest.importorskip("msgpack"); pytest.importorskip("msgpack_numpy")
    _, data, ex = setup(tmp_path)
    rgb = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    depth = np.full((48, 64), 0.8768, np.float32); depth[0, 0] = np.nan; depth[0, 1] = np.inf; depth[1, 0] = 0.8764
    payload = pack_message(12.5, rgb, depth, ex.camera["K"], ex.camera_pose_in_base(data), ex.truth(data))
    message = unpack_message(payload)
    assert message["stamp"] == 12.5 and np.array_equal(message["rgb"], rgb)
    assert message["depth_mm"].dtype == np.uint16 and message["depth_mm"][5, 5] == 877 and message["depth_mm"][1, 0] == 876 and message["depth_mm"][0, :2].tolist() == [0, 0]
    assert message["truth"]["target"] == "can" and np.allclose(message["truth"]["objects"]["can"]["position"], [0.5, 0, 0], atol=1e-9)
    assert message["T_base_cam"].shape == (4, 4)
