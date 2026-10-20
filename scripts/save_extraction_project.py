#!/usr/bin/env python
"""Turn a maneuver parameter file (from extraction_search.py) into an editor project + bundle, and re-check it with the
editor's own object-physics preview.

    python scripts/save_extraction_project.py build/search_093_a.jsonl.best.json --scale 0.93 --name "라면 꺼내기 후보 A"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_lab as L  # noqa: E402
from motioncreator.motion import new_project, save_bundle  # noqa: E402
from motioncreator.robot import FEET, Robot  # noqa: E402


def build_project(robot: Robot, params: dict, scale: float, name: str, pelvis_x: float, lateral: float) -> dict:
    scene = L.scale_stacks(L.sample_project(), scale)
    qposes, durations, stages, failures = L.build_keyframes(robot, params, scene, pelvis_x, lateral)
    if failures:
        raise SystemExit(f"IK failed at {failures[:5]}")
    scene = L.settled_scene(robot, scene, qposes[0])           # the layout the physics evaluates, at rest
    project = new_project(robot, name)
    project["scene_objects"] = scene["scene_objects"]
    project["scene_groups"] = scene.get("scene_groups", [])
    project["keyframes"] = [{"name": f"{stage} {index}", "duration": float(max(0.1, duration)), "qpos": q.tolist(),
                             "pins": list(FEET), "angle_pins": []}
                            for index, (stage, duration, q) in enumerate(zip(stages, durations, qposes))]
    project["current_qpos"] = qposes[0].tolist()
    project["extraction_maneuver"] = {"params": params, "stack_scale": scale, "pelvis_x": pelvis_x, "lateral_m": lateral}
    return project


ROOT = Path(__file__).resolve().parents[1]


def write_stages(robot: Robot, project: dict, folder: Path, scale: float, tolerance_m: float = 0.15) -> dict:
    """Cut the compiled bundle into the task stages by locating each keyframe in the dense motion."""
    from motioncreator.motion import compile_motion
    motion = compile_motion(robot, project, 30)
    times, qpos = np.asarray(motion["time"]), np.asarray(motion["qpos"])
    names = [k["name"] for k in project["keyframes"]]
    cursor, found = 0, []
    for key in project["keyframes"]:
        distance = np.linalg.norm(qpos[cursor:, 7:] - np.asarray(key["qpos"])[7:], axis=1)
        cursor += int(np.argmin(distance))
        found.append(float(times[cursor]))
    place = next(i for i, n in enumerate(names) if n.startswith("place ") and "hold" not in n)
    lift_end = max(i for i, n in enumerate(names) if n.startswith("lift_tilt"))
    carry_end = max(i for i, n in enumerate(names) if n.startswith("carry"))
    stages = {"target_object": "Mupama_Stack_L03_R04", "authored_root_at_s": 0.0,
              "robot_start_xy_yaw": [-0.5, 0.0, 0.0], "grasp_tolerance_m": tolerance_m, "adaptive_sample_period_s": 0.04,
              "max_object_shift_m": 0.25,
              "ready": {"start_s": 0.0, "end_s": 0.0},
              "grasp": {"start_s": 0.0, "end_s": round(found[lift_end], 3)},
              "transfer": {"start_s": round(found[lift_end], 3), "end_s": round(found[carry_end], 3)},
              "place": {"start_s": round(found[carry_end], 3), "end_s": round(found[carry_end] + L.PLACE_HOLD_S, 3)},
              "return": {"start_s": round(found[carry_end] + L.PLACE_HOLD_S, 3), "end_s": round(float(times[-1]), 3)}}
    (folder / "task_stages.json").write_text(json.dumps(stages, indent=2, ensure_ascii=False) + "\n")
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file")
    parser.add_argument("--scale", type=float, default=0.93)
    parser.add_argument("--name", default="라면 꺼내기 후보")
    parser.add_argument("--pelvis-x", type=float, default=-0.004)
    parser.add_argument("--lateral", type=float, default=0.25)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    params = json.loads(Path(args.params_file).read_text())["params"]
    robot = Robot("g1-tools", ("scoop", "scoop"))
    project = build_project(robot, params, args.scale, args.name, args.pelvis_x, args.lateral)
    print(f"keyframes {len(project['keyframes'])}, total {sum(k['duration'] for k in project['keyframes']):.1f}s")
    from motioncreator import policy_preview
    result = policy_preview.simulate(project, controller="objects")
    tid = L.target_of(project)["id"]
    first, last = result["object_states"][0], result["object_states"][-1]
    shift = np.array(last[tid]["position"]) - np.array(first[tid]["position"])
    print("editor object-physics preview:", result["summary"]["reason"], "| target shift", np.round(shift, 3).tolist())
    if not args.no_save:
        saved = save_bundle(robot, project, 30)
        folder = ROOT / "motions" / saved["files"][0].split("/")[0]
        write_stages(robot, project, folder, scale=args.scale)
        print("saved", folder.name)


if __name__ == "__main__":
    main()
