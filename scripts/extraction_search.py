#!/usr/bin/env python
"""Random + local search over the push-lift-tilt maneuver, scored by the physics rig (see extraction_lab.py).

    python scripts/extraction_search.py --scale 0.93 --n 120 --out build/search_093.jsonl [--init best.json --spread 0.3]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_lab as L  # noqa: E402

RANGES = {
    "dx0": (-0.20, -0.09), "dz0": (-0.09, 0.03), "th0_deg": (-20.0, 12.0), "phi_deg": (-30.0, 20.0),
    "pushes": (2.0, 6.99), "depth_m": (0.04, 0.17), "push_s": (0.25, 0.9), "wiggle_z_m": (0.0, 0.009),
    "wiggle_th_deg": (0.0, 9.0), "lift_m": (0.0, 0.09), "tilt_pull_m": (0.0, 0.07), "th1_deg": (8.0, 55.0),
    "carry_pull_m": (0.04, 0.22), "carry_lift_m": (0.05, 0.28),
}
_STATE = {}


def init(scales, pelvis_x: float, lateral: float):
    """`scales`: stack cross-section factors; a candidate is scored on all of them (robust to the packing size)."""
    from motioncreator.robot import Robot
    robot = Robot("g1-tools", ("scoop", "scoop"))
    rigs = {}
    for scale in ([scales] if isinstance(scales, float) else scales):
        project = L.scale_stacks(L.sample_project(), scale)
        rigs[scale] = (project, L.KinematicRig(robot, project), L.target_of(project)["id"])
    first = next(iter(rigs.values()))[0]
    _STATE.update(robot=robot, project=first, rigs=rigs, lateral=lateral, pelvis_x=pelvis_x)


def score(report: dict, ik_failures: int) -> float:
    if ik_failures:
        return -5.0 - ik_failures * 0.1
    held = min(report["end_target_n"].values()) > 2.0
    others = sum(report["others_moved_m"].values())
    value = 10.0 * float(np.clip(report["rise_m"], -0.05, 0.16)) + (1.0 if held and report["rise_m"] > 0.05 else 0.0)
    value += 2.0 * float(np.clip(report["toward_robot_m"], 0.0, 0.15))
    value -= 4.0 * others                                        # contact magnitudes are ignored for now (sim friction/size differ)
    return float(value)


def evaluate(params: dict) -> dict:
    s = _STATE
    if L.keyframe_count(params) > L.PREFERRED_KEYFRAMES:         # keep it editable by a person (the hard limit is 100)
        return {"params": params, "score": -4.0, "ik_failures": 0, "error": "too many keyframes"}
    try:
        qposes, durations, _, fails = L.build_keyframes(s["robot"], params, s["project"], s["pelvis_x"], s["lateral"], nfev=120)
        if fails:
            return {"params": params, "score": score({}, len(fails)), "ik_failures": len(fails)}
        end = len(qposes) - 3                                   # evaluate up to the end of the carry; hold/return are not scored
        pose, total = L.pose_function(qposes[:end + 1], durations[:end + 1])
        reports, scores = {}, []
        for scale, (project, rig, tid) in s["rigs"].items():
            report = L.extraction_report(project, rig.run(pose, total + 1.0, tid), tid)
            reports[scale] = report
            scores.append(score(report, 0))
        # robust: the worst size decides, the mean breaks ties
        return {"params": params, "score": float(min(scores) + 0.25 * np.mean(scores)), "scores": scores,
                "report": next(iter(reports.values())), "reports": {str(k): v for k, v in reports.items()}, "ik_failures": 0}
    except Exception as exc:                                  # a bad sample must not stop the search
        return {"params": params, "score": -9.0, "error": f"{type(exc).__name__}: {exc}"}


def sample(rng, init_params=None, spread=0.3):
    out = {}
    for key, (low, high) in RANGES.items():
        if init_params is None:
            out[key] = float(rng.uniform(low, high))
        else:
            out[key] = float(np.clip(init_params[key] + rng.normal(0, spread * (high - low)), low, high))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="0.91,0.93,0.95")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pelvis-x", type=float, default=-0.004)
    parser.add_argument("--lateral", type=float, default=0.25)
    parser.add_argument("--init", help="JSON file with the parameters to perturb")
    parser.add_argument("--spread", type=float, default=0.3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    seed_params = json.loads(Path(args.init).read_text())["params"] if args.init else None
    candidates = [sample(rng, seed_params, args.spread) for _ in range(args.n)]
    if seed_params:
        candidates[0] = {k: float(seed_params[k]) for k in RANGES}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    began, best = time.time(), None
    with mp.get_context("spawn").Pool(args.workers, initializer=init, initargs=([float(v) for v in args.scales.split(",")], args.pelvis_x, args.lateral)) as pool, \
            out.open("w") as handle:
        for index, result in enumerate(pool.imap_unordered(evaluate, candidates)):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if best is None or result["score"] > best["score"]:
                best = result
                rep = result.get("report", {})
                print(f"[{index + 1}/{args.n} {time.time() - began:.0f}s] score {result['score']:.2f} per-size {[round(v, 2) for v in result.get('scores', [])]} rise {rep.get('rise_m')} "
                      f"toward {rep.get('toward_robot_m')} end {rep.get('end_target_n')} max {rep.get('max_any_n')} "
                      f"others {rep.get('others_moved_m')}", flush=True)
    Path(str(out) + ".best.json").write_text(json.dumps(best, indent=1, ensure_ascii=False))
    print("best score", best["score"], "->", str(out) + ".best.json")


if __name__ == "__main__":
    main()
