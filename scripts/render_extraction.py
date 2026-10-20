#!/usr/bin/env python
"""Render an extraction maneuver (from extraction_search.py) to an mp4: robot prescribed, stacks physical.

    python scripts/render_extraction.py build/search_093_a.jsonl.best.json --scale 0.93 --out build/extraction_A.mp4
Left: side view along the stacks' axis (the section the path was planned in). Right: oblique view.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_lab as L  # noqa: E402
from motioncreator.robot import Robot  # noqa: E402

VIEWS = (dict(azimuth=90, elevation=-8, distance=1.7, lookat=(0.35, 0.0, 0.85)),
         dict(azimuth=140, elevation=-22, distance=2.0, lookat=(0.35, 0.0, 0.85)))


def render_video(params: dict, scale: float, out: str, pelvis_x: float = -0.004, lateral: float = 0.25, fps: int = 30,
                 slow: float = 1.0) -> int:
    robot = Robot("g1-tools", ("scoop", "scoop"))
    project = L.scale_stacks(L.sample_project(), scale)
    qposes, durations, _, failures = L.build_keyframes(robot, params, project, pelvis_x, lateral)
    if failures:
        raise RuntimeError(f"IK failed: {failures[:3]}")
    pose, total = L.pose_function(qposes, durations)
    rig = L.KinematicRig(robot, project)
    model = rig.model
    data = mujoco.MjData(model)
    data.qpos[:] = rig.sim.data.qpos
    data.qpos[:36] = pose(0.0)
    mujoco.mj_forward(model, data)
    for _ in range(int(1.0 / model.opt.timestep)):
        data.qpos[:36] = pose(0.0)
        data.qvel[:35] = 0
        mujoco.mj_step(model, data)
    width, height = 800, 600
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.55, 0.9
    renderer = mujoco.Renderer(model, height, width)
    cameras = []
    for view in VIEWS:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = view["azimuth"], view["elevation"], view["distance"]
        camera.lookat[:] = view["lookat"]
        cameras.append(camera)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * width}x{height}",
                               "-r", str(fps), "-i", "-", "-vf", f"setpts={slow}*PTS", "-pix_fmt", "yuv420p", "-c:v", "libx264", out],
                              stdin=subprocess.PIPE)
    dt, every = model.opt.timestep, int(round(1.0 / fps / model.opt.timestep))
    frames = 0
    for step in range(int((total + 1.0) / dt)):
        data.qpos[:36] = pose(step * dt)
        data.qvel[:35] = 0
        mujoco.mj_step(model, data)
        if step % every == 0:
            images = []
            for camera in cameras:
                renderer.update_scene(data, camera)
                images.append(renderer.render())
            ffmpeg.stdin.write(np.hstack(images).tobytes())
            frames += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file")
    parser.add_argument("--scale", type=float, default=0.93)
    parser.add_argument("--pelvis-x", type=float, default=-0.004)
    parser.add_argument("--lateral", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--slow", type=float, default=1.0, help="playback slowdown factor")
    args = parser.parse_args()
    params = json.loads(Path(args.params_file).read_text())["params"]
    frames = render_video(params, args.scale, args.out, args.pelvis_x, args.lateral, args.fps, args.slow)
    print(f"wrote {args.out}: {frames} frames, {frames / args.fps:.1f}s")


if __name__ == "__main__":
    main()
