"""The task FSM driving the editor's decoupled-WBC simulation (needs the installed policy assets)."""
import json

import numpy as np
import pytest

from motioncreator import controller_api
from motioncreator.controller_session import ControllerSession, _validated_stages, load_stages
from motioncreator.decoupled_wbc import DecoupledSimulation, verify_assets
from motioncreator.motion import new_project
from motioncreator.robot import Robot
from ramen_controller.contracts import JOINT_NAMES

needs_assets = pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")

STAGES = {"target_object": "Crate", "authored_root_at_s": 0.0,
          "ready": {"start_s": 0.0, "end_s": 0.0}, "grasp": {"start_s": 0.0, "end_s": 0.5},
          "transfer": {"start_s": 0.5, "end_s": 1.0}, "place": {"start_s": 1.0, "end_s": 1.5},
          "return": {"start_s": 1.5, "end_s": 2.0}, "grasp_tolerance_m": 0.08}


def project_with_crate():
    project = new_project(Robot())
    project["keyframes"][0]["pins"] = []
    project["scene_objects"] = [{
        "id": "crate", "name": "Crate", "shape": "box", "position": [.45, 0., .45],
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.1, .2, .1], "mass_kg": .3,
        "friction": .9, "color": "#aa7744", "opacity": 1., "visible": True,
    }, {
        "id": "table", "name": "Table", "shape": "box", "position": [.45, 0., .2], "fixed": True,
        "quaternion_xyzw": [0., 0., 0., 1.], "size": [.4, .6, .4], "mass_kg": 5.,
        "friction": .9, "color": "#777777", "opacity": 1., "visible": True,
    }]
    return project


def write_small_bundle(folder, root_z):
    """A 2 s bundle whose arms drift gently; its authored pelvis sits where the simulated robot stands."""
    frames, fps = 61, 30
    qpos = np.zeros((frames, 36))
    qpos[:, 2], qpos[:, 3] = root_z, 1.0
    qpos[:, 7 + 15:7 + 29] = np.linspace(0, 0.15, frames)[:, None]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(json.dumps({"format": "motioncreator.g1.v1", "joint_names": list(JOINT_NAMES),
                                                      "fps": fps, "samples": frames}))
    np.savetxt(folder / "motion.csv", qpos, delimiter=",")
    return folder


def make_session(tmp_path, **options):
    simulation = DecoupledSimulation(Robot(), project_with_crate(), autostart=False)
    bundle = write_small_bundle(tmp_path / "bundle", float(simulation.data.qpos[2]))
    return ControllerSession(simulation, bundle, STAGES, options, autostart=False)


def run(session, seconds, until=None):
    simulation = session.simulation
    end = simulation.play_time + seconds
    while simulation.play_time < end:
        simulation.step()
        session.poll()
        if until and until(session):
            return True
    return until is None


def transitions(session):
    return [e.state for e in session.fsm.events if e.kind.value == "transition"]


@needs_assets
def test_a_full_cycle_runs_through_the_simulation_with_a_proxy_grasp(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        assert session.simulation.control_mode == "external"
        session.start()
        assert run(session, 40, until=lambda s: s.fsm.cycles >= 1), transitions(session)
        assert session.simulation.phase != "fallen"
        assert transitions(session)[:7] == ["MANUAL_POSITION", "OBSERVE", "PLAN_GRASP", "EXECUTE_GRASP", "TRANSFER", "PLACE", "RETURN"]
        snapshot = session.snapshot()
        assert snapshot["task"]["cycles"] >= 1 and snapshot["task"]["options"]["proxy_grasp"] is True
        assert snapshot["control_mode"] == "external" and "state" in snapshot and "object_states" in snapshot
        json.dumps(snapshot)                                           # the whole snapshot must be JSON-serialisable
    finally:
        session.close()


@needs_assets
def test_an_unconfirmed_grasp_waits_for_the_operator_then_restarts(tmp_path):
    session = make_session(tmp_path)                                   # contact-based grasp judgement
    try:
        session.start()
        assert run(session, 40, until=lambda s: s.fsm.state.value == "RECOVERY")
        assert session.fsm.recovery_reason == "grasp not confirmed"
        run(session, 3)
        assert session.fsm.state.value == "RECOVERY"                   # no automatic retry with the arms out of the ready pose
        session.operator_reset()
        assert run(session, 3, until=lambda s: s.fsm.state.value == "INITIALIZE")
    finally:
        session.close()


@needs_assets
def test_hiding_the_target_ends_in_recovery_and_showing_it_again_resumes(tmp_path):
    session = make_session(tmp_path, vision_visible=False)
    try:
        session.start()
        assert run(session, 40, until=lambda s: s.fsm.state.value == "RECOVERY")
        assert "observation failed" in session.fsm.recovery_reason
        session.set_options({"vision_visible": True})
        assert run(session, 20, until=lambda s: "PLAN_GRASP" in transitions(s))
    finally:
        session.close()


@needs_assets
def test_a_fall_stops_the_simulation_and_is_reported_to_the_fsm_once(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        session.start()
        run(session, 6)
        assert session.fsm.state.value != "RECOVERY"
        session.simulation.data.qpos[2] = 0.1                          # the pelvis drops below the fall threshold
        session.simulation.step()
        assert session.simulation.phase == "fallen" and not session.simulation.running
        assert session.poll() is True                                  # one tick to deliver the fall
        assert session.fsm.state.value == "RECOVERY" and session.fsm.recovery_reason == "robot fell"
        assert session.snapshot()["task"]["state"] == "RECOVERY"
        assert session.poll() is False                                 # and not again until the robot is reset
        session.reset()
        assert session.fsm.state.value == "INITIALIZE"
    finally:
        session.close()


@needs_assets
def test_reset_rebuilds_the_controller_and_stops_the_robot(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        session.start()
        run(session, 12)
        assert session.runner.ticks > 0
        session.reset()
        assert not session.simulation.running and session.simulation.control_mode == "external"
        assert session.fsm.state.value == "INITIALIZE" and session.fsm.cycles == 0 and session.runner.ticks == 0
        session.set_options({"proxy_grasp": False, "unknown": True})
        assert session.options == {"proxy_grasp": False, "vision_visible": True}
    finally:
        session.close()


@needs_assets
def test_unknown_target_object_is_rejected(tmp_path):
    simulation = DecoupledSimulation(Robot(), project_with_crate(), autostart=False)
    try:
        bundle = write_small_bundle(tmp_path / "bundle", 0.79)
        with pytest.raises(ValueError, match="Crate|not in the scene"):
            ControllerSession(simulation, bundle, {**STAGES, "target_object": "Nope"}, autostart=False)
    finally:
        simulation.close()


def test_stage_mapping_is_validated_and_a_bundle_file_overrides_the_default(tmp_path):
    assert _validated_stages(STAGES) is STAGES
    for broken, message in (({**STAGES, "grasp": {"start_s": 1.0}}, "start_s and end_s"),
                            ({**STAGES, "place": {"start_s": 2.0, "end_s": 1.0}}, "ends before"),
                            ({k: v for k, v in STAGES.items() if k != "target_object"}, "target_object")):
        with pytest.raises(ValueError, match=message):
            _validated_stages(broken)
    assert load_stages(None)["target_object"]                          # the repository default
    (tmp_path / "task_stages.json").write_text(json.dumps(STAGES))
    assert load_stages(tmp_path)["target_object"] == "Crate"


def test_bundle_folder_rejects_unsafe_or_incomplete_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(controller_api, "ROOT", tmp_path)
    (tmp_path / "motions" / "good").mkdir(parents=True)
    (tmp_path / "motions" / "good" / "metadata.json").write_text("{}")
    (tmp_path / "motions" / "good" / "motion.csv").write_text("")
    (tmp_path / "motions" / "legacy").mkdir()
    assert controller_api.bundle_folder("good/project.json") == tmp_path / "motions" / "good"
    for bad, message in (("/etc/passwd", "bundle must be"), ("../x/project.json", "bundle must be"),
                         ("project.json", "bundle must be"), ("legacy/project.json", "not a saved bundle")):
        with pytest.raises(ValueError, match=message):
            controller_api.bundle_folder(bad)
