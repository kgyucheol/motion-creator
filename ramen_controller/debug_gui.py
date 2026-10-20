"""Debug web GUI (stdlib only): FSM state, events, robot summary and the head camera with the target overlaid.

The product front end is built by a teammate; this is a developer view. Serve with `DebugServer(runner,
frame_source).start()` and open http://<host>:<port>/. Binds to 127.0.0.1 by default (use an SSH tunnel).
"""
from __future__ import annotations

import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy.spatial.transform import Rotation

from .vision import CameraFrame


def png_bytes(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder for an (H, W, 3) uint8 array."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def project(points_base: np.ndarray, frame: CameraFrame) -> np.ndarray:
    """Pixel coordinates (N, 2) of base-frame points; NaN for points behind the camera."""
    T = np.linalg.inv(np.asarray(frame.T_base_cam, float))
    points = np.atleast_2d(points_base)
    camera = (T[:3, :3] @ points.T).T + T[:3, 3]
    pixels = np.full((len(points), 2), np.nan)
    front = camera[:, 2] > 0.05
    uv = (np.asarray(frame.K, float) @ camera[front].T).T
    pixels[front] = uv[:, :2] / uv[:, 2:3]
    return pixels


def _line(image: np.ndarray, a, b, color) -> None:
    if np.isnan(a).any() or np.isnan(b).any():
        return
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1))
    xs = np.linspace(a[0], b[0], n + 1).round().astype(int)
    ys = np.linspace(a[1], b[1], n + 1).round().astype(int)
    ok = (xs >= 0) & (xs < image.shape[1]) & (ys >= 0) & (ys < image.shape[0])
    image[ys[ok], xs[ok]] = color


def draw_pose(image: np.ndarray, position, quaternion_xyzw, frame: CameraFrame, axis_m: float = 0.08) -> np.ndarray:
    """Overlay a base-frame pose as RGB axes (x red, y green, z blue)."""
    out = np.array(image, copy=True)
    origin = np.asarray(position, float)
    rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    ends = np.vstack([origin, *(origin + rotation[:, i] * axis_m for i in range(3))])
    pixels = project(ends, frame)
    for i, color in enumerate(((255, 40, 40), (40, 255, 40), (60, 120, 255))):
        _line(out, pixels[0], pixels[i + 1], color)
    return out


PAGE = """<!doctype html><meta charset=utf-8><title>Ramen controller debug</title>
<style>
:root{color-scheme:dark;--bg:#14161a;--panel:#1c1f26;--fg:#e6e8ec;--mute:#8a909c;--ok:#4cc38a;--warn:#e5b567;--bad:#ef6a6a}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
.grid{display:grid;grid-template-columns:minmax(300px,1fr) minmax(300px,1fr);gap:16px}@media(max-width:720px){.grid{grid-template-columns:1fr}}
section{background:var(--panel);border-radius:8px;padding:12px 14px}h2{margin:0 0 8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute)}
#state{font-size:26px;font-weight:600}.RECOVERY{color:var(--bad)}.OBSERVE,.MANUAL_POSITION,.INITIALIZE{color:var(--warn)}.EXECUTE_GRASP,.TRANSFER,.PLACE,.RETURN,.PLAN_GRASP{color:var(--ok)}
table{border-collapse:collapse;width:100%}td{padding:2px 0}td:first-child{color:var(--mute);width:42%}img{width:100%;border-radius:6px;background:#000}
#events div{font-family:ui-monospace,monospace;font-size:12px;padding:2px 0;border-bottom:1px solid #262a33}.warning{color:var(--warn)}
</style>
<h1 style="font-size:16px;margin:0 0 12px">Ramen controller <span style="color:var(--mute);font-weight:400">debug</span></h1>
<div class=grid>
<section><h2>Task</h2><div id=state>-</div><table id=kv></table></section>
<section><h2>Head camera</h2><img id=cam alt="head camera"></section>
<section style="grid-column:1/-1"><h2>Events (newest first)</h2><div id=events></div></section></div>
<script>
const kv=(a)=>a.map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
async function tick(){try{const s=await (await fetch('/api/state')).json();
document.getElementById('state').textContent=s.state;document.getElementById('state').className=s.state;
const r=s.robot,c=s.command,t=s.target;
document.getElementById('kv').innerHTML=kv([['cycles',s.cycles],['retries',s.retries],['task time',s.task_time+' s'],['stalled',s.stalled],
['recovery',s.recovery_reason||'-'],['robot xyz',r?r.root_xyz.join(', '):'-'],['tilt / fallen',r?r.tilt_deg+'° / '+r.fallen:'-'],
['nav cmd',c?c.nav.join(', '):'-'],['waist ypr',c?c.waist_ypr.join(', '):'-'],['target (base)',t?t.position.join(', ')+' ['+t.source+']':'-']]);
document.getElementById('events').innerHTML=s.events.slice().reverse().map(e=>`<div class="${e.kind}">${e.t.toFixed(1)}s  ${e.state}  ${e.reason}</div>`).join('');
document.getElementById('cam').src='/api/frame.png?'+Date.now();}catch(e){}setTimeout(tick,400)}
tick();
</script>"""


class DebugServer:
    def __init__(self, runner, frame_source=None, host: str = "127.0.0.1", port: int = 8765):
        """frame_source: object with `.latest() -> CameraFrame | None` (e.g. SimFeed)."""
        self.runner, self.frame_source = runner, frame_source
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body: bytes, content_type: str):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._send(200, json.dumps(owner.runner.snapshot()).encode(), "application/json")
                elif path == "/api/frame.png":
                    image = owner.frame_image()
                    if image is None:
                        self._send(404, b"no frame yet", "text/plain")
                    else:
                        self._send(200, png_bytes(image), "image/png")
                else:
                    self._send(404, b"not found", "text/plain")

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def frame_image(self) -> np.ndarray | None:
        frame = self.frame_source.latest() if self.frame_source else None
        if frame is None:
            return None
        image = frame.rgb
        pose = self.runner.fsm.target_pose
        if pose is not None and pose.frame_id == "base":
            image = draw_pose(image, pose.position, pose.quaternion_xyzw, frame)
        return image

    def start(self) -> "DebugServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
