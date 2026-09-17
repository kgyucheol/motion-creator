import json

import mujoco
import numpy as np
import pytest

from motioncreator import decoupled_api
from motioncreator.decoupled_wbc import (
    ASSET_ROOT,
    CommandState,
    DecoupledSimulation,
    authored_waist_to_torso_rpy,
    build_environment_model,
    build_observation,
    save_recording_bundle,
    verify_assets,
)
from motioncreator.motion import new_project
from motioncreator.robot import Robot


def parameters():
    return json.loads((ASSET_ROOT.parents[2] / "integrations/decoupled-wbc-parameters.json").read_text())


def test_keyboard_commands_are_clamped_and_reset():
    values = parameters()
    command = CommandState()
    for _ in range(20): command.apply_key("w", values)
    for _ in range(20): command.apply_key("q", values)
    for _ in range(20): command.apply_key("2", values)
    np.testing.assert_allclose(command.nav, [.5, 0., 1.])
    assert command.height == pytest.approx(.2)
    assert command.apply_key("x", values) is False
    command.apply_key("z", values)
    np.testing.assert_allclose(command.nav, 0.)
    assert command.height == pytest.approx(.74)


def test_policy_observation_has_exact_training_layout():
    values = parameters()
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "g1_gear_wbc.xml")) if verify_assets()["available"] else Robot().model
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    command = CommandState(nav=np.array([.1, -.2, .3], dtype=np.float32), height=.7)
    observation = build_observation(data, np.arange(15, dtype=np.float32), command,
                                    np.array([.1, .2, .3]), values)
    assert observation.shape == (86,)
    np.testing.assert_allclose(observation[:7], [.2, -.4, .15, .7, .1, .2, .3])
    np.testing.assert_allclose(observation[71:], np.arange(15))


def test_authored_waist_is_converted_to_policy_rpy_and_limited():
    limits = np.array([.52, .52, 2.618])
    np.testing.assert_allclose(authored_waist_to_torso_rpy([.2, 0., 0.], limits), [0., 0., .2], atol=1e-6)
    converted = authored_waist_to_torso_rpy([3., 1., 1.], limits)
    assert np.all(np.abs(converted) <= limits + 1e-7)


def test_recording_bundle_is_reimportable(tmp_path):
    objects = [{"id": "box", "name": "Box", "shape": "box", "position": [.4, 0., .1],
                "quaternion_xyzw": [0., 0., 0., 1.], "size": [.2, .2, .2], "mass_kg": 1.,
                "friction": .7, "color": "#ffffff", "opacity": 1., "visible": True}]
    frames = [{"time": i / 25, "qpos": np.r_[0., 0., .74, 1., 0., 0., 0., np.zeros(29)],
               "nav": np.array([.1, 0., 0.]), "height": .74,
               "torso_rpy": np.zeros(3), "upper_time": i / 25,
               "object_states": {"box": {"position": [.4, 0., .1 + i / 100],
                                            "quaternion_xyzw": [0., 0., 0., 1.]}}} for i in range(3)]
    result = save_recording_bundle(frames, "wave", tmp_path, scene_objects=objects)
    folder = tmp_path / result["folder"]
    with np.load(folder / "motion.npz") as motion:
        assert motion["qpos"].shape == (3, 36)
        assert motion["navigate_command"].shape == (3, 3)
        assert motion["object_position"].shape == (3, 1, 3)
        assert motion["object_quaternion_xyzw"].shape == (3, 1, 4)
        assert motion["object_ids"].tolist() == ["box"]
        assert motion["fps"] == 25
    assert (folder / "commands.csv").read_text().startswith("time,vx,vy,yaw_rate")
    metadata = json.loads((folder / "metadata.json").read_text())
    environment = json.loads((folder / "environment.json").read_text())
    assert metadata["source_project"] == "wave" and metadata["samples"] == 3
    assert metadata["object_count"] == 1 and environment["scene_objects"] == objects


@pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")
def test_environment_uses_a_free_body_for_grasp_target_and_fixed_support():
    robot = Robot()
    project = new_project(robot)
    project["scene_objects"] = [{
        "id": "crate", "name": "Crate", "shape": "box", "position": [.4, 0., .1],
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.2, .3, .2], "mass_kg": .5,
        "friction": .9, "color": "#aa7744", "opacity": 1., "visible": True,
    }, {
        "id": "support", "name": "Support", "shape": "box", "position": [.4, 0., .025],
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.4, .5, .05], "mass_kg": 1.,
        "friction": .9, "color": "#777777", "opacity": 1., "visible": True,
    }]
    project["keyframes"][0]["grasp"] = {"object_id": "crate"}
    model = build_environment_model(project)
    assert model.nq == 43 and model.nv == 41 and model.nu == 29
    assert model.joint("wbc_object_joint_0").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("wbc_object_0").id > 0
    assert model.body("wbc_object_1").jntnum == 0
    assert model.geom("left_wbc_grip").type == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("right_wbc_grip").type == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("left_wbc_grip").contype == 1
    assert model.geom("right_wbc_grip").contype == 1
    assert model.geom("wbc_object_geom_0").contype == 0
    assert model.npair == 7


def test_http_command_fallback_dispatches_without_websocket(monkeypatch):
    class FakeSession:
        def __init__(self): self.calls = []
        def play(self): self.calls.append(("play", ""))
        def stop(self): self.calls.append(("stop", ""))
        def reset(self): self.calls.append(("reset", ""))
        def apply_key(self, key): self.calls.append(("key", key))
        def snapshot(self): return {"calls": self.calls}

    session = FakeSession()
    monkeypatch.setattr(decoupled_api.sessions, "get", lambda identifier: session)
    for action, key in (("play", ""), ("key", "w"), ("stop", ""), ("reset", "")):
        result = decoupled_api.session_command("session", decoupled_api.CommandInput(action=action, key=key))
        assert result["calls"][-1] == (action, key)


@pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")
def test_policy_drives_lower_body_while_authored_arm_tracks():
    robot = Robot()
    project = new_project(robot)
    project["keyframes"][0]["pins"] = []
    target = json.loads(json.dumps(project["keyframes"][0]))
    target.update(name="arm target", duration=1., pins=[])
    target["qpos"][22] = .35
    project["keyframes"].append(target)
    simulation = DecoupledSimulation(robot, project, autostart=False)
    try:
        simulation.play()
        for index in range(700):
            if index == 450: simulation.apply_key("w")
            simulation.step()
        snapshot = simulation.snapshot()
        assert snapshot["phase"] == "playing" and snapshot["policy"] == "walk"
        assert snapshot["upper_time"] == pytest.approx(1., abs=.01)
        assert snapshot["recording_frames"] == 87
        assert simulation.data.qpos[22] == pytest.approx(.35, abs=.08)
        assert simulation.data.qpos[0] > .01
    finally:
        simulation.close()
