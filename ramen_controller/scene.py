"""Generate the sim2sim MJCF: tool-equipped G1 + MotionCreator scene objects + head camera.

Inputs are plain data: the exported robot model (assets/g1_tools, see scripts/export_robot_assets.py) and
the `environment.json` of a MotionCreator bundle. Nothing in upstream decoupled_wbc is modified; the
generated files live in a build directory and can be opened in any MuJoCo viewer.

Collision proxies follow MotionCreator's conventions (sizes are full extents): box, cylinder, sphere,
convex hull (mesh) and a five-panel open box. Objects use MuJoCo's default collision masks, so they
collide with the floor, each other and the whole robot.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAMERA_NAME = "head"


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole head camera rigidly mounted on a body. Defaults approximate a D435 colour stream (69 x 42 deg)."""
    body: str = "d435_link"
    width: int = 640
    height: int = 480
    fovy_deg: float = 42.0
    # d435_link follows the ROS convention (x forward, y left, z up); MuJoCo cameras look along -z with +y up.
    xyaxes: tuple[float, ...] = (0, -1, 0, 0, 0, 1)

    def intrinsics(self) -> np.ndarray:
        f = 0.5 * self.height / np.tan(np.radians(self.fovy_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])


def _numbers(values) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def _wxyz(quaternion_xyzw) -> str:
    q = np.asarray(quaternion_xyzw, dtype=float)
    return _numbers(q[[3, 0, 1, 2]])


def _add_collision_geoms(body: ET.Element, asset: ET.Element, index: int, item: dict) -> list[str]:
    shape = item.get("collision_shape", item["shape"])
    size = np.asarray(item.get("collision_size", item["size"]), dtype=float)
    friction = _numbers([item["friction"], .005, .0001])
    base = f"obj{index}_geom"
    mass = float(item["mass_kg"])
    if shape == "convex_hull":
        mesh = f"obj{index}_hull"
        ET.SubElement(asset, "mesh", name=mesh, scale=_numbers(item["size"]),
                      vertex=_numbers(np.asarray(item["collision_hull_vertices"])),
                      face=" ".join(str(int(v)) for v in np.asarray(item["collision_hull_faces"]).ravel()))
        ET.SubElement(body, "geom", name=base, type="mesh", mesh=mesh, mass=str(mass), friction=friction)
        return [base]
    if shape != "open_box":
        half = size / 2 if shape == "box" else [size[0] / 2] if shape == "sphere" else [size[0] / 2, size[2] / 2]
        attributes = {"name": base, "type": shape, "size": _numbers(half), "mass": str(mass), "friction": friction}
        if item.get("collision_quaternion_xyzw") is not None:
            attributes["quat"] = _wxyz(item["collision_quaternion_xyzw"])
        ET.SubElement(body, "geom", **attributes)
        return [base]
    t = float(np.clip(item.get("wall_thickness_m", .02), .001, min(size) / 3))
    h = size[2] - t
    panels = [("bottom", [size[0], size[1], t], [0, 0, -size[2] / 2 + t / 2]),
              ("left", [t, size[1] - 2 * t, h], [-(size[0] - t) / 2, 0, t / 2]),
              ("right", [t, size[1] - 2 * t, h], [(size[0] - t) / 2, 0, t / 2]),
              ("front", [size[0], t, h], [0, -(size[1] - t) / 2, t / 2]),
              ("back", [size[0], t, h], [0, (size[1] - t) / 2, t / 2])]
    volumes = [float(np.prod(d)) for _, d, _ in panels]
    names = []
    for (part, dims, pos), volume in zip(panels, volumes):
        name = f"{base}_{part}"
        ET.SubElement(body, "geom", name=name, type="box", size=_numbers(np.asarray(dims) / 2), pos=_numbers(pos),
                      mass=str(mass * volume / sum(volumes)), friction=friction)
        names.append(name)
    return names


def build_scene(assets_dir, environment_json, out_dir, camera: CameraSpec = CameraSpec(),
                target_name: str | None = None, robot_start_xy_yaw: tuple[float, float, float] = (-1.0, 0.0, 0.0)) -> Path:
    """Write out_dir/scene.xml and out_dir/scene_objects.json; return the scene path.

    Objects keep the world coordinates they were authored in (robot at the origin in its ready pose). The
    robot starts `robot_start_xy_yaw` away, because the default arms-down pose would intersect the tools
    with the box; the operator walks it to the authored spot (MANUAL_POSITION), as on the real robot.
    """
    assets_dir, out_dir = Path(assets_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(assets_dir / "g1_tools.xml").getroot()
    root.find("compiler").set("meshdir", str(assets_dir / "meshes"))

    mount = root.find(f".//body[@name='{camera.body}']")
    if mount is None:
        raise ValueError(f"robot model has no body {camera.body!r} to mount the camera on")
    ET.SubElement(mount, "camera", name=CAMERA_NAME, pos="0 0 0", xyaxes=_numbers(camera.xyaxes),
                  fovy=str(camera.fovy_deg))

    world = root.find("worldbody")
    pelvis = world.find("body[@name='pelvis']")
    x, y, yaw = robot_start_xy_yaw
    z = float(pelvis.get("pos", "0 0 .79").split()[2])
    pelvis.set("pos", _numbers([x, y, z]))
    pelvis.set("quat", _numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", pos="1 -2 2.5", dir="-.3 .6 -.7", diffuse=".4 .4 .4")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    environment = json.loads(Path(environment_json).read_text(encoding="utf-8"))
    sidecar = []
    for index, item in enumerate(environment["scene_objects"]):
        body = ET.SubElement(world, "body", name=f"obj{index}", pos=_numbers(item["position"]),
                             quat=_wxyz(item["quaternion_xyzw"]))
        if not item.get("fixed", False):
            ET.SubElement(body, "freejoint", name=f"obj{index}_joint")
        geoms = _add_collision_geoms(body, asset, index, item)
        sidecar.append({"index": index, "body": f"obj{index}", "id": item.get("id"), "name": item["name"],
                        "fixed": bool(item.get("fixed", False)), "geoms": geoms,
                        "is_target": item["name"] == target_name})
    if target_name and not any(entry["is_target"] for entry in sidecar):
        raise ValueError(f"no scene object named {target_name!r}")

    ET.indent(root)
    scene_path = out_dir / "scene.xml"
    scene_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    handles_path = assets_dir / "tcp_handles.json"
    handles = json.loads(handles_path.read_text(encoding="utf-8")) if handles_path.exists() else {}
    tool_bodies = {side: handles[f"{side}_hand"]["link"] for side in ("left", "right") if f"{side}_hand" in handles}
    (out_dir / "scene_objects.json").write_text(json.dumps(
        {"camera": {"name": CAMERA_NAME, "width": camera.width, "height": camera.height,
                    "fovy_deg": camera.fovy_deg, "K": camera.intrinsics().tolist(), "body": camera.body},
         "tool_bodies": tool_bodies, "robot_start_xy_yaw": list(robot_start_xy_yaw), "objects": sidecar}, indent=1, ensure_ascii=False), encoding="utf-8")
    return scene_path
