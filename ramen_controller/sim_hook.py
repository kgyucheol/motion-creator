"""Run this project's scene and head camera inside decoupled_wbc's own simulator, without modifying it.

`install()` swaps `base_sim.DefaultEnv` for `RamenEnv`. The environment renders the head camera (RGB + depth)
in the simulator thread, where its GL context lives, and publishes it with the ground truth on a ZMQ PUB
socket (see sim_feed). Imports decoupled_wbc/rclpy, so it only loads in the sim environment.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import zmq

from decoupled_wbc.control.envs.g1.sim import base_sim

from .scene import CAMERA_NAME
from .sim_feed import DEFAULT_PORT, TruthExtractor, pack_message


class RamenEnv(base_sim.DefaultEnv):
    def __init__(self, config, env_name="default", camera_configs=None, onscreen=False, offscreen=False,
                 enable_image_publish=False):
        scene_dir = Path(config["ROBOT_SCENE"]).parent
        sidecar = scene_dir / "scene_objects.json"
        camera = json.loads(sidecar.read_text(encoding="utf-8"))["camera"]
        # offscreen rendering is always on: the camera feed is the point of this environment
        super().__init__(config, env_name, camera_configs={camera["name"]: {"height": camera["height"],
                                                                            "width": camera["width"]}},
                         onscreen=onscreen, offscreen=True, enable_image_publish=False)
        self._camera = camera
        self._extractor = TruthExtractor(self.mj_model, sidecar,
                                         tool_bodies=json.loads(sidecar.read_text(encoding="utf-8"))["tool_bodies"],
                                         camera_name=camera["name"])
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.bind(f"tcp://127.0.0.1:{int(os.environ.get('RAMEN_FEED_PORT', DEFAULT_PORT))}")
        self.frames_published = 0

    def update_render_caches(self):
        caches = super().update_render_caches()
        renderer = self.renderers[CAMERA_NAME]
        renderer.enable_depth_rendering()
        renderer.update_scene(self.mj_data, camera=CAMERA_NAME)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        payload = pack_message(time.monotonic(), caches[CAMERA_NAME + "_image"], depth, self._camera["K"],
                               self._extractor.camera_pose_in_base(self.mj_data), self._extractor.truth(self.mj_data))
        try:
            self._socket.send(payload, flags=zmq.NOBLOCK)
            self.frames_published += 1
        except zmq.Again:
            pass                                    # slow subscriber: drop, the feed is latest-wins
        return caches


def install() -> None:
    base_sim.DefaultEnv = RamenEnv
