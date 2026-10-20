"""Generate the sim2sim MJCF for the whole-body-controller simulator from this package's own models.

The robot is the tool-equipped G1 (`tool_model.tool_model_xml`), the scene objects come from a bundle's
`environment.json` through the same collision proxies the editor uses (`scene_geometry`), and a D435 head camera
is mounted on the robot. The tools default to the pair the bundle was authored with (`metadata.json`
`mounted_tools`), so swapping tools in the editor changes the simulated robot without any export step.

Outputs `scene.xml` plus a `scene_objects.json` sidecar (camera, tool bodies, object names/geoms) that the
runtime (`ramen_controller.sim`) reads; the sidecar is the only contract between the two packages.

Unlike the editor's preview physics, objects use MuJoCo's default collision masks (they collide with the floor,
each other and the whole robot) instead of an explicit contact allow-list.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene_geometry import append_collision_geoms, grounded_position
from .tool_model import DEFAULT_TOOLS, parse_tools, tool_model_xml

CAMERA_NAME = "head"

# The whole-body policy was trained on the stock G1 model, whose joints carry these dynamics (see
# g1_29dof_with_hand.xml classes). The URDF-derived tool model has none, so they are applied explicitly.
STOCK_JOINT_DAMPING = 0.05
STOCK_FRICTIONLOSS = 0.2
STOCK_WRIST_FRICTIONLOSS = 0.1
STOCK_FLOOR_FRICTION = (1.0, 0.005, 0.0001)


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole head camera rigidly mounted on a body. Defaults approximate a D435 colour stream (42 deg vertical)."""
    body: str = "d435_link"
    width: int = 640
    height: int = 480
    fovy_deg: float = 42.0
    # d435_link follows the ROS convention (x forward, y left, z up); MuJoCo cameras look along -z with +y up.
    xyaxes: tuple[float, ...] = (0, -1, 0, 0, 0, 1)

    def intrinsics(self) -> np.ndarray:
        f = 0.5 * self.height / np.tan(np.radians(self.fovy_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])


def bundle_tools(bundle_dir) -> tuple[str, str]:
    """(left, right) tool ids a bundle was authored with; the stock pair if the bundle does not say."""
    path = Path(bundle_dir) / "metadata.json"
    mounted = json.loads(path.read_text(encoding="utf-8")).get("mounted_tools") if path.is_file() else None
    return parse_tools(":".join(mounted)) if mounted else DEFAULT_TOOLS


def _numbers(values) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def _wxyz(quaternion_xyzw) -> str:
    return _numbers(np.asarray(quaternion_xyzw, dtype=float)[[3, 0, 1, 2]])


def _apply_stock_dynamics(root: ET.Element) -> None:
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if joint.get("type", "hinge") != "hinge" or not name.endswith("_joint"):
            continue
        joint.set("damping", str(STOCK_JOINT_DAMPING))
        joint.set("frictionloss", str(STOCK_WRIST_FRICTIONLOSS if "wrist" in name else STOCK_FRICTIONLOSS))
    floor = root.find("worldbody/geom[@name='floor']")
    if floor is not None:
        floor.set("friction", _numbers(STOCK_FLOOR_FRICTION))


def build_scene(environment_json, out_dir, tools: tuple[str, str] = DEFAULT_TOOLS, camera: CameraSpec = CameraSpec(),
                target_name: str | None = None, robot_start_xy_yaw: tuple[float, float, float] = (-1.0, 0.0, 0.0)) -> Path:
    """Write out_dir/scene.xml and out_dir/scene_objects.json; return the scene path.

    Objects keep the world coordinates they were authored in (robot at the origin in its ready pose). The
    robot starts `robot_start_xy_yaw` away, because the default arms-down pose would intersect the tools
    with the box; the operator walks it to the authored spot (MANUAL_POSITION), as on the real robot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root, handles = tool_model_xml(tuple(tools))
    _apply_stock_dynamics(root)

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

    environment = json.loads(Path(environment_json).read_text(encoding="utf-8"))
    sidecar = []
    for index, item in enumerate(environment["scene_objects"]):
        body = ET.SubElement(world, "body", name=f"obj{index}", pos=_numbers(grounded_position(item)),
                             quat=_wxyz(item["quaternion_xyzw"]))
        if not item.get("fixed", False):
            ET.SubElement(body, "freejoint", name=f"obj{index}_joint")
        geoms = append_collision_geoms(body, "sim", index, item, root)
        for name in geoms:                                   # the editor's proxies are contact-free until paired
            geom = body.find(f"geom[@name='{name}']")
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
        sidecar.append({"index": index, "body": f"obj{index}", "id": item.get("id"), "name": item["name"],
                        "fixed": bool(item.get("fixed", False)), "geoms": geoms,
                        "is_target": item["name"] == target_name})
    if target_name and not any(entry["is_target"] for entry in sidecar):
        raise ValueError(f"no scene object named {target_name!r}")

    ET.indent(root)
    scene_path = out_dir / "scene.xml"
    scene_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    tool_bodies = {side: handles[f"{side}_hand"][0] for side in ("left", "right") if f"{side}_hand" in handles}
    (out_dir / "scene_objects.json").write_text(json.dumps(
        {"camera": {"name": CAMERA_NAME, "width": camera.width, "height": camera.height,
                    "fovy_deg": camera.fovy_deg, "K": camera.intrinsics().tolist(), "body": camera.body},
         "tools": list(tools), "tool_bodies": tool_bodies, "robot_start_xy_yaw": list(robot_start_xy_yaw),
         "objects": sidecar}, indent=1, ensure_ascii=False), encoding="utf-8")
    return scene_path
