import copy
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
    ghost_tracking_command,
    prepare_grasp_control,
    save_recording_bundle,
    verify_assets,
)
from motioncreator.grasp import object_signature
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
    for _ in range(20): command.apply_key("3", values)
    for _ in range(20): command.apply_key("8", values)
    np.testing.assert_allclose(command.nav, [.5, 0., 1.])
    assert command.height == pytest.approx(.2)
    np.testing.assert_allclose(command.torso_offset_rpy, [.52, 0., -2.], atol=1e-6)
    assert command.apply_key("x", values) is False
    command.apply_key("z", values)
    np.testing.assert_allclose(command.nav, 0.)
    assert command.height == pytest.approx(.74)
    np.testing.assert_allclose(command.torso_offset_rpy, 0.)


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


def test_ghost_root_is_converted_to_bounded_policy_inputs():
    values = parameters()
    origin = np.array([0., 0., .74, 1., 0., 0., 0.])
    reference = origin.copy(); reference[:3] = [.1, 0., .69]
    following = reference.copy(); following[0] += .02
    current = reference.copy(); current[0] -= .04
    nav, height, tracking = ghost_tracking_command(
        current, reference, following, origin, origin, .02, values)
    assert nav[0] == pytest.approx(.5)  # feed-forward + feedback is safely clamped
    assert nav[1] == pytest.approx(0.) and nav[2] == pytest.approx(0.)
    assert height == pytest.approx(.69)
    assert tracking["root_error_m"] == pytest.approx(.04)


def test_grasp_control_ignores_stale_initial_event_and_never_refits_authored_pose():
    robot = Robot()
    project = new_project(robot)
    item = {"id": "crate", "name": "Crate", "shape": "box", "position": [.4, 0., .2],
            "quaternion_xyzw": [0., 0., 0., 1.], "size": [.2, .3, .4], "mass_kg": .5,
            "friction": .9, "color": "#aa7744", "opacity": 1., "visible": True}
    project["scene_objects"] = [item]
    event = {"format": "motioncreator.two-hand-grasp.v1", "object_id": "crate",
             "left_surface_uv": [0., 0.], "right_surface_uv": [0., 0.],
             "hand_gap_m": .29, "closure_seconds": .4,
             "target_force_n": 8., "max_force_n": 60.}
    project["keyframes"][0]["grasp"] = copy.deepcopy(event)
    approach = copy.deepcopy(project["keyframes"][0])
    approach.update(name="approach", duration=1.)
    approach.pop("grasp")
    contact = copy.deepcopy(approach)
    contact.update(name="contact", duration=1., grasp=copy.deepcopy(event))
    project["keyframes"].extend([approach, contact])

    control = prepare_grasp_control(robot, project)
    assert control["start_time"] == pytest.approx(2.)
    assert "closure_ready" not in control and "arm_offset" not in control

    closure = np.asarray(contact["qpos"], dtype=float)
    closure[22] += .1
    contact["grasp"].update(object_signature=object_signature(item), closure_qpos=closure.tolist())
    control = prepare_grasp_control(robot, project)
    assert "closure_ready" not in control and "arm_offset" not in control


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
def test_environment_applies_physics_and_robot_collision_to_every_object():
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
    assert model.nq == 50 and model.nv == 47 and model.nu == 29
    assert model.joint("wbc_object_joint_0").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("wbc_object_joint_1").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("wbc_object_0").id > 0
    assert model.body("wbc_object_1").jntnum == 1
    assert model.geom("left_wbc_grip").type == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("right_wbc_grip").type == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("left_wbc_grip").contype == 0
    assert model.geom("right_wbc_grip").contype == 0
    for side in ("left", "right"):
        assert all(model.geom(f"{side}_physical_hand_{index}").id >= 0 for index in range(6))
    assert model.geom("wbc_object_geom_0").contype == 0
    assert model.geom("wbc_object_geom_0").conaffinity == 0
    assert model.geom("wbc_object_geom_1").contype == 0
    assert model.geom("wbc_object_geom_1").conaffinity == 0
    assert model.npair == 27


@pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")
def test_ungrasped_object_falls_and_collides_with_floor():
    robot = Robot()
    project = new_project(robot)
    project["scene_objects"] = [{
        "id": "falling", "name": "Falling box", "shape": "box", "position": [1.5, 0., 1.],
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.2, .2, .2], "mass_kg": .5,
        "friction": .8, "color": "#ffffff", "opacity": 1., "visible": True,
    }]
    model = build_environment_model(project)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("wbc_object_0").id
    initial_z = float(data.xpos[body, 2])
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert data.xpos[body, 2] < initial_z - .5
    assert data.xpos[body, 2] == pytest.approx(.1, abs=.01)


@pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")
def test_scene_object_does_not_contact_unlisted_robot_links():
    robot = Robot()
    project = new_project(robot)
    project["scene_objects"] = [{
        "id": "overlap", "name": "Overlap probe", "shape": "box", "position": [0., 0., .75],
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.3, .3, .3], "mass_kg": .5,
        "friction": .8, "color": "#ffffff", "opacity": 1., "visible": True,
    }]
    model = build_environment_model(project)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_geom = model.geom("wbc_object_geom_0").id
    object_body = model.body("wbc_object_0").id
    assert not any(object_geom in (contact.geom1, contact.geom2)
                   and model.geom_bodyid[contact.geom2 if contact.geom1 == object_geom else contact.geom1]
                   not in (0, object_body)
                   for contact in data.contact)


def test_http_command_fallback_dispatches_without_websocket(monkeypatch):
    class FakeSession:
        def __init__(self): self.calls = []
        def play(self): self.calls.append(("play", ""))
        def stop(self): self.calls.append(("stop", ""))
        def reset(self): self.calls.append(("reset", ""))
        def set_control_mode(self, mode): self.calls.append(("mode", mode))
        def apply_key(self, key): self.calls.append(("key", key))
        def snapshot(self): return {"calls": self.calls}

    session = FakeSession()
    monkeypatch.setattr(decoupled_api.sessions, "get", lambda identifier: session)
    for action, key in (("play", ""), ("key", "w"), ("stop", ""), ("reset", "")):
        result = decoupled_api.session_command("session", decoupled_api.CommandInput(action=action, key=key))
        assert result["calls"][-1] == (action, key)
    result = decoupled_api.session_command(
        "session", decoupled_api.CommandInput(action="mode", mode="manual"))
    assert result["calls"][-1] == ("mode", "manual")


@pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")
def test_auto_mode_drives_policy_toward_moving_ghost_root():
    robot = Robot()
    project = new_project(robot)
    project["keyframes"][0]["pins"] = []
    target = copy.deepcopy(project["keyframes"][0])
    target.update(name="forward", duration=1., pins=[])
    target["qpos"][0] += .2
    project["keyframes"].append(target)
    simulation = DecoupledSimulation(robot, project, autostart=False)
    try:
        simulation.play()
        peak_forward_command = 0.
        for _ in range(700):
            simulation.step()
            peak_forward_command = max(peak_forward_command, float(simulation.command.nav[0]))
        snapshot = simulation.snapshot()
        assert snapshot["control_mode"] == "auto"
        assert peak_forward_command > .1
        assert snapshot["state"]["qpos"][0] > .05
        assert snapshot["tracking"]["root_error_m"] < .15
    finally:
        simulation.close()


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
        simulation.set_control_mode("manual")
        simulation.play()
        for index in range(700):
            if index == 450: simulation.apply_key("w")
            simulation.step()
        snapshot = simulation.snapshot()
        assert snapshot["phase"] == "playing" and snapshot["policy"] == "walk"
        assert snapshot["upper_time"] == pytest.approx(1., abs=.01)
        assert snapshot["recording_frames"] == 87
        np.testing.assert_allclose(snapshot["reference_state"]["qpos"], simulation._reference_at(1.))
        assert set(snapshot["tracking"]) == {"lower_rmse_deg", "upper_rmse_deg", "upper_max_deg",
                                              "root_error_m", "yaw_error_deg"}
        assert all(np.isfinite(value) for value in snapshot["tracking"].values())
        assert simulation.data.qpos[22] == pytest.approx(.35, abs=.08)
        assert simulation.data.qpos[0] > .01
    finally:
        simulation.close()
