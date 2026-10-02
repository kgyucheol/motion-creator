import json
import struct
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from ramen_controller.debug_gui import DebugServer, draw_pose, png_bytes, project
from ramen_controller.contracts import ObjectPose
from ramen_controller.vision import CameraFrame
from tests.test_runner import Clock, make

K = np.array([[100., 0, 64], [0, 100, 48], [0, 0, 1]])


def frame(T=None):
    # optical frame: z forward. T_base_cam maps camera -> base; base x forward means camera z = base x.
    base_from_cam = np.eye(4)
    base_from_cam[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])     # cam x=-base y, cam y=-base z, cam z=base x
    return CameraFrame(stamp=0, rgb=np.zeros((96, 128, 3), np.uint8), K=K, T_base_cam=base_from_cam if T is None else T)


def decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, size = 8, b"", None
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(kind + body) & 0xFFFFFFFF
        if kind == b"IHDR":
            size = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    w, h = size
    raw = zlib.decompress(idat)
    rows = np.frombuffer(raw, np.uint8).reshape(h, 1 + 3 * w)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(h, w, 3)


def test_png_roundtrip():
    image = np.random.default_rng(0).integers(0, 255, (7, 9, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(png_bytes(image)), image)


def test_projection_matches_the_pinhole_model_and_drops_points_behind():
    f = frame()
    ahead, behind = project(np.array([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), f)
    assert np.allclose(ahead, [64, 48])                      # straight ahead lands on the principal point
    assert np.isnan(behind).all()
    left = project(np.array([[2.0, 0.2, 0.0]]), f)[0]        # base +y is left: image x decreases
    assert np.isclose(left[0], 64 - 100 * 0.2 / 2.0) and np.isclose(left[1], 48)
    up = project(np.array([[2.0, 0.0, 0.2]]), f)[0]
    assert np.isclose(up[1], 48 - 100 * 0.2 / 2.0)


def test_draw_pose_colours_the_axes_where_they_project():
    f = frame()
    out = draw_pose(f.rgb, [2.0, 0.0, 0.0], [0, 0, 0, 1], f, axis_m=0.4)
    assert (out != 0).any() and not (f.rgb != 0).any()       # input untouched
    # the x axis points straight at the camera (one pixel, overdrawn by the others); y goes left, z goes up
    green = np.argwhere((out[..., 1] == 255) & (out[..., 0] == 40))
    assert len(green) > 5 and green[:, 1].min() < 64 - 10 and abs(green[:, 0].mean() - 48) < 2
    blue = np.argwhere((out[..., 2] == 255) & (out[..., 0] == 60))
    assert len(blue) > 5 and blue[:, 0].min() < 48 - 10 and abs(blue[:, 1].mean() - 64) < 2
    off = draw_pose(f.rgb, [-2.0, 0.0, 0.0], [0, 0, 0, 1], f)               # behind the camera: nothing drawn
    assert not (off != 0).any()


@pytest.fixture
def server():
    clock = Clock()
    fsm, runner, *_ = make(clock)
    runner.tick()
    fsm.target_pose = ObjectPose(stamp=0, frame_id="base", position=[2.0, 0, 0], quaternion_xyzw=[0, 0, 0, 1])

    class Source:
        def __init__(self):
            self.value = frame()

        def latest(self):
            return self.value

    source = Source()
    srv = DebugServer(runner, source, port=0).start()
    yield srv, source
    srv.stop()


def get(srv, path):
    return urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=5)


def test_state_endpoint_page_and_frame(server):
    srv, _ = server
    state = json.loads(get(srv, "/api/state").read())
    assert state["state"] == "INITIALIZE" and state["target"]["position"] == [2.0, 0.0, 0.0]
    assert b"Ramen controller" in get(srv, "/").read()
    image = decode_png(get(srv, "/api/frame.png").read())
    assert image.shape == (96, 128, 3) and (image != 0).any()              # target overlay was drawn


def test_unknown_path_and_missing_frame_are_404(server):
    srv, source = server
    with pytest.raises(urllib.error.HTTPError) as error:
        get(srv, "/nope")
    assert error.value.code == 404
    source.value = None
    with pytest.raises(urllib.error.HTTPError) as error:
        get(srv, "/api/frame.png")
    assert error.value.code == 404
