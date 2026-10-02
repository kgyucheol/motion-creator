"""The task FSM driving the editor's decoupled-WBC simulation (needs the installed policy assets)."""
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")        # the editor simulation; the runtime-only environment skips this file

from motioncreator import controller_api
from motioncreator.controller_session import ControllerSession, _validated_stages, load_stages
from motioncreator.decoupled_wbc import DecoupledSimulation, verify_assets
from motioncreator.motion import new_project
from motioncreator.robot import Robot
from ramen_controller.contracts import JOINT_NAMES

needs_assets = pytest.mark.skipif(not verify_assets()["available"], reason="decoupled-WBC assets are not installed")

STAGES = {"target_object": "Crate", "authored_root_at_s": 0.0, "robot_start_xy_yaw": [-0.6, 0.0, 0.0],
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
    # the fixed planner is the deterministic one for these wiring tests; adaptive planning has its own tests
    return ControllerSession(simulation, bundle, STAGES, {"adaptive_grasp": False, **options}, autostart=False)


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
def test_robot_starts_away_from_the_authored_spot_and_walks_in(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        assert session.simulation.data.qpos[0] == pytest.approx(session.simulation.model.qpos0[0] - 0.6)
        session.start()
        assert run(session, 30, until=lambda s: s.fsm.state.value == "OBSERVE")          # arrived and confirmed its position
        assert abs(session.simulation.data.qpos[0]) < 0.12                                # walked to the authored origin
        session.reset()
        assert session.simulation.data.qpos[0] == pytest.approx(session.simulation.model.qpos0[0] - 0.6)
    finally:
        session.close()


@needs_assets
def test_start_pose_applies_position_and_yaw_on_every_reset():
    simulation = DecoupledSimulation(Robot(), project_with_crate(), autostart=False)
    try:
        origin = simulation.data.qpos[:2].copy()
        simulation.set_start_pose(-1.0, 0.25, np.pi / 2)
        np.testing.assert_allclose(simulation.data.qpos[:2], origin + (-1.0, 0.25))
        np.testing.assert_allclose(simulation.data.qpos[3:7], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-9)
        simulation.reset()
        np.testing.assert_allclose(simulation.data.qpos[:2], origin + (-1.0, 0.25))
    finally:
        simulation.close()


def test_start_pose_in_the_stage_mapping_must_be_x_y_yaw():
    with pytest.raises(ValueError, match="robot_start_xy_yaw"):
        _validated_stages({**STAGES, "robot_start_xy_yaw": [1.0, 2.0]})


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
        assert session.options == {"proxy_grasp": False, "vision_visible": True, "adaptive_grasp": False,
                                   "skip_positioning": False}
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


def test_robot_follows_the_projects_mounted_tools():
    from motioncreator.controller_session import ControllerSessions
    sessions = ControllerSessions()
    tooled = sessions._robot_for({"mounted_tools": ["scoop", "scoop"]})
    assert tooled.model_id == "g1-tools" and tooled.tools == ("scoop", "scoop")
    assert sessions._robot_for({"mounted_tools": ["scoop", "scoop"]}) is tooled          # built once per tool pair
    other = sessions._robot_for({"mounted_tools": ["scoop", "end_support"]})
    assert other is not tooled and other.tools == ("scoop", "end_support")
    assert sessions._robot_for({}).tools == sessions._robot_for({"mounted_tools": None}).tools


@needs_assets
def test_simulation_model_uses_the_given_robots_tools_not_the_servers_environment(monkeypatch):
    from motioncreator.decoupled_wbc import build_environment_model
    monkeypatch.delenv("MOTIONCREATOR_MODEL", raising=False)                              # the server default would be the stock hands
    project = project_with_crate()
    stock = build_environment_model(project)
    tooled = build_environment_model(project, Robot("g1-tools", ("scoop", "scoop")))
    names = lambda model: {model.body(i).name for i in range(model.nbody)}
    assert "right_scoop_link" not in names(stock) and "right_scoop_link" in names(tooled)
    assert tooled.nu == 29 and tooled.nq == stock.nq


def test_old_sessions_are_closed_when_too_many_accumulate():
    from motioncreator.controller_session import ControllerSessions

    class Fake:
        def __init__(self): self.closed = False
        def close(self): self.closed = True

    sessions = ControllerSessions()
    fakes = [Fake() for _ in range(4)]
    sessions.sessions = {f"s{i}": fake for i, fake in enumerate(fakes)}
    sessions._evict_oldest(1)
    assert [f.closed for f in fakes] == [True, True, True, False] and list(sessions.sessions) == ["s3"]
    sessions._evict_oldest(1)
    assert list(sessions.sessions) == ["s3"]                           # nothing more to evict


def test_create_keeps_at_most_max_sessions_by_closing_the_oldest(monkeypatch):
    from motioncreator import controller_session as module

    class FakeSimulation:
        def __init__(self, robot, project): self.closed = False
        def close(self): self.closed = True

    class FakeSession:
        def __init__(self, simulation, bundle, stages, options=None): self.simulation, self.closed = simulation, False
        def close(self): self.closed = True

    monkeypatch.setattr(module, "verify_assets", lambda: {"available": True})
    monkeypatch.setattr(module, "DecoupledSimulation", FakeSimulation)
    monkeypatch.setattr(module, "ControllerSession", FakeSession)
    sessions = module.ControllerSessions()
    monkeypatch.setattr(sessions, "_robot_for", lambda project: object())
    made = [sessions.create({}, None, {})[1] for _ in range(module.MAX_SESSIONS + 1)]
    assert len(sessions.sessions) == module.MAX_SESSIONS
    assert made[0].closed and not made[-1].closed



@needs_assets
def test_the_planner_option_swaps_the_grasp_planner_and_reports_timing(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        assert session.fsm.motion.grasp_planner.name == "fixed"
        session.set_options({"adaptive_grasp": True})
        assert session.fsm.motion.grasp_planner.name == "adaptive" and session.options["adaptive_grasp"] is True
        session.set_options({"adaptive_grasp": False})
        assert session.fsm.motion.grasp_planner.name == "fixed"
        assert set(session.snapshot()["task"]["planner"]) == {"name", "last_plan_s", "last_error"}
    finally:
        session.close()


@needs_assets
def test_the_fsm_hands_the_observed_pose_and_the_current_robot_state_to_the_planner(tmp_path):
    from ramen_controller.contracts import MotionKind, RobotState

    seen = []
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        real = session.planners["fixed"].inner

        class Recorder:
            def plan(self, pose, state=None):
                seen.append((pose, state))
                return real.plan(pose, state)

        session.planners["fixed"].inner = Recorder()
        session.start()
        assert run(session, 30, until=lambda s: "EXECUTE_GRASP" in transitions(s))
        pose, state = seen[0]
        assert pose.frame_id == "base" and isinstance(state, RobotState) and state.joint_positions.shape == (29,)
        assert state.root_position[2] == pytest.approx(session.simulation.data.qpos[2], abs=0.05)
        assert session.snapshot()["task"]["planner"]["last_plan_s"] is not None
    finally:
        session.close()


@needs_assets
def test_a_planner_that_cannot_reach_the_object_sends_the_task_to_recovery_with_its_reason(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True, adaptive_grasp=True, vision_visible=True)
    try:
        session.stages["max_object_shift_m"] = 0.0           # nothing counts as "where it was authored"
        session._build()
        session.start()
        assert run(session, 40, until=lambda s: s.fsm.state.value == "RECOVERY")
        assert "authored" in session.fsm.recovery_reason or "unreachable" in session.fsm.recovery_reason
        assert session.snapshot()["task"]["planner"]["last_error"]
    finally:
        session.close()


@needs_assets
def test_skipping_positioning_starts_at_the_work_position_and_never_walks(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True, skip_positioning=True)
    try:
        assert session.simulation.data.qpos[0] == pytest.approx(session.simulation.model.qpos0[0])   # not the -0.6 m walk-in start
        session.start()
        # (the fixed planner may then refuse: the standing robot settles ~9 cm back of its start, beyond its 8 cm tolerance)
        assert run(session, 20, until=lambda s: s.fsm.state.value == "PLAN_GRASP"), transitions(session)
        assert "MANUAL_POSITION" not in transitions(session)
        assert transitions(session)[:2] == ["OBSERVE", "PLAN_GRASP"]
        assert abs(session.simulation.data.qpos[0]) < 0.15                     # never walked anywhere
        assert session.snapshot()["task"]["options"]["skip_positioning"] is True
    finally:
        session.close()


@needs_assets
def test_toggling_skip_positioning_resets_to_the_matching_start_and_back(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        origin = session.simulation.model.qpos0[0]
        session.start()
        run(session, 3)
        session.set_options({"skip_positioning": True})
        assert not session.simulation.running and session.fsm.state.value == "INITIALIZE"
        assert session.simulation.data.qpos[0] == pytest.approx(origin)
        assert session.fsm.config.skip_manual_position is True
        session.set_options({"skip_positioning": True, "proxy_grasp": False})      # unchanged skip: no second reset
        assert session.simulation.data.qpos[0] == pytest.approx(origin)
        session.set_options({"skip_positioning": False})
        assert session.simulation.data.qpos[0] == pytest.approx(origin - 0.6)
        assert session.fsm.config.skip_manual_position is False
    finally:
        session.close()


@needs_assets
def test_step_state_runs_to_the_next_state_then_pauses(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True, skip_positioning=True)
    try:
        session.step_state()
        assert session.snapshot()["task"]["stepping"] is True
        assert run(session, 10, until=lambda s: not s.simulation.running)
        assert session.fsm.state.value == "OBSERVE" and session.snapshot()["task"]["stepping"] is False
        time_paused = session.simulation.play_time
        for _ in range(300):                                                    # a paused simulation does not advance
            session.simulation.step()
            session.poll()
        assert session.simulation.play_time == time_paused and session.fsm.state.value == "OBSERVE"
        session.step_state()
        assert run(session, 10, until=lambda s: not s.simulation.running)
        assert session.fsm.state.value == "PLAN_GRASP"
    finally:
        session.close()


@needs_assets
def test_tcp_trace_records_planned_and_actual_paths_and_clearances(tmp_path):
    session = make_session(tmp_path, proxy_grasp=True)
    try:
        empty = session.trace.report()
        assert empty["planned"]["left"] == [] and empty["actual"]["left"] == [] and empty["error_mm"] is None
        session.start()
        assert run(session, 40, until=lambda s: s.fsm.state.value == "EXECUTE_GRASP"), transitions(session)
        run(session, 1.5)
        report = session.trace.report()
        json.dumps(report)
        assert report["segment"] and report["active"] and report["t"] is not None and not report["blending"]
        assert len(report["planned"]["left"]) >= 5 and len(report["planned"]["right"]) >= 5
        assert len(report["actual"]["left"]) >= 3
        assert set(report["error_mm"]) == {"left", "right"} and all(0 <= v < 150 for v in report["error_mm"].values())
        assert np.linalg.norm(np.subtract(report["planned_now"]["left"], report["now"]["left"])) * 1000 == \
            pytest.approx(report["error_mm"]["left"], abs=60)               # the reported error is the planned-to-actual gap
        assert np.allclose(report["now"]["left"], report["actual"]["left"][-1], atol=0.05)
        for side in ("left", "right"):
            assert report["clearance"][side]["target_mm"] < 300 + 1e-6
            assert report["clearance"][side]["nearest"] is None or "name" in report["clearance"][side]["nearest"]
        first_path = report["path_id"]
        assert run(session, 20, until=lambda s: s.fsm.state.value == "TRANSFER"), transitions(session)
        run(session, 0.2)
        assert session.trace.report()["path_id"] > first_path                 # a new segment starts a new path
        assert session.trace_error == ""
        assert session.snapshot()["task"]                                    # the trace does not disturb the snapshot
    finally:
        session.close()


@needs_assets
def test_stage_mapping_configures_the_adaptive_planners_sampling_and_closed_loop(tmp_path):
    simulation = DecoupledSimulation(Robot(), project_with_crate(), autostart=False)
    bundle = write_small_bundle(tmp_path / "bundle", float(simulation.data.qpos[2]))
    plain = ControllerSession(simulation, bundle, STAGES, {"adaptive_grasp": True}, autostart=False)
    try:
        inner = plain.planners["adaptive"].inner
        assert inner.closed_loop is False and np.diff(inner.sample_times)[0] == pytest.approx(0.2)
    finally:
        plain.close()
    simulation = DecoupledSimulation(Robot(), project_with_crate(), autostart=False)
    tuned = ControllerSession(simulation, bundle, {**STAGES, "adaptive_sample_period_s": 0.05, "closed_loop_tcp": True},
                              {"adaptive_grasp": True}, autostart=False)
    try:
        inner = tuned.planners["adaptive"].inner
        assert inner.closed_loop is True and np.diff(inner.sample_times)[0] == pytest.approx(0.05)
    finally:
        tuned.close()
