import json

import numpy as np
import pytest

from ramen_controller.contracts import MotionKind, ObjectPose
from ramen_controller.fsm import GraspPlanError
from ramen_controller.motion_library import FixedGraspPlanner, MotionEntry, MotionLibrary, load_entries
from ramen_controller.operator import KeyboardOperator, MAX_LINEAR
from tests.test_motion_bundle_and_pose import write_bundle


def pose(x=0.5, frame="base"):
    return ObjectPose(stamp=0, frame_id=frame, position=[x, 0, 0.9], quaternion_xyzw=[0, 0, 0, 1])


def library(tmp_path, kinds=(MotionKind.READY, MotionKind.TRANSFER, MotionKind.PLACE, MotionKind.RETURN)):
    bundle = write_bundle(tmp_path / "b")
    entries = {k: MotionEntry(bundle, end_s=0.5) for k in kinds}
    from ramen_controller.motion_bundle import load_segment
    grasp = load_segment(bundle, MotionKind.GRASP)
    return MotionLibrary(entries, FixedGraspPlanner(grasp, [0.5, 0, 0.9], tolerance_m=0.03))


def test_library_loads_slices_and_caches(tmp_path):
    lib = library(tmp_path)
    segment = lib.segment(MotionKind.READY)
    assert segment.kind is MotionKind.READY and np.isclose(segment.duration, 0.5)
    assert lib.segment(MotionKind.READY) is segment


def test_library_requires_all_fixed_stages(tmp_path):
    with pytest.raises(ValueError, match="place"):
        library(tmp_path, kinds=(MotionKind.READY, MotionKind.TRANSFER, MotionKind.RETURN))


def test_fixed_planner_accepts_authored_pose_and_explains_refusals(tmp_path):
    lib = library(tmp_path)
    assert lib.plan_grasp(pose(0.51)).kind is MotionKind.GRASP
    with pytest.raises(GraspPlanError, match="150 mm.*limit 30 mm"):
        lib.plan_grasp(pose(0.65))
    with pytest.raises(GraspPlanError, match="base frame"):
        lib.plan_grasp(pose(0.5, frame="camera"))


def test_config_paths_are_relative_to_the_file(tmp_path):
    write_bundle(tmp_path / "bundles" / "ready")
    (tmp_path / "motions.json").write_text(json.dumps({"ready": {"bundle": "bundles/ready", "end_s": 0.4}}))
    entry = load_entries(tmp_path / "motions.json")[MotionKind.READY]
    assert entry.bundle == (tmp_path / "bundles" / "ready").resolve() and entry.end_s == 0.4
    (tmp_path / "bad.json").write_text(json.dumps({"dance": {"bundle": "x"}}))
    with pytest.raises(ValueError, match="dance"):
        load_entries(tmp_path / "bad.json")


def test_keyboard_accumulates_clips_and_stops():
    op = KeyboardOperator()
    for _ in range(8):
        op.handle_key("w")
    assert op.nav()[0] == MAX_LINEAR
    op.handle_key("s"); op.handle_key("a"); op.handle_key("e")
    assert np.allclose(op.nav(), [MAX_LINEAR - 0.1, 0.1, -0.2])
    op.handle_key("z")
    assert not op.nav().any()


def test_keyboard_confirm_and_reset_semantics():
    op = KeyboardOperator()
    op.handle_key("w"); op.handle_key("\n")
    assert op.position_confirmed() and op.nav()[0] == 0.1        # confirm does not stop motion by itself
    op.handle_key("r")
    assert op.reset_requested() and not op.reset_requested()     # one-shot
    assert not op.position_confirmed() and not op.nav().any()


def test_keyboard_has_no_negative_zero_residue():
    op = KeyboardOperator()
    op.handle_key("w"); op.handle_key("s")
    assert not np.signbit(op.nav()).any() and not op.nav().any()
