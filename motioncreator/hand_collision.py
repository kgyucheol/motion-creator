"""Name the robot's real hand geoms for explicit object contact pairs."""
from __future__ import annotations

import xml.etree.ElementTree as ET


def physical_hand_geom_names(root: ET.Element, side: str, *, dex3: bool = False) -> list[str]:
    """Return stable names for real (non-auxiliary) hand collision geoms.

    The source MJCFs duplicate most meshes as a non-colliding visual geom and a
    default collision geom.  The rubber hand is visual-only, but an explicit
    MuJoCo contact pair can still use it without enabling broad collisions.
    """
    meshes = ([f"{side}_wrist_yaw_link", f"{side}_hand_palm_link",
               f"{side}_hand_middle_0_link", f"{side}_hand_middle_1_link",
               f"{side}_hand_index_0_link", f"{side}_hand_index_1_link"]
              if dex3 else [f"{side}_wrist_yaw_link", f"{side}_rubber_hand"])
    names = []
    for index, mesh in enumerate(meshes):
        candidates = root.findall(f".//geom[@mesh='{mesh}']")
        if not candidates:
            raise ValueError(f"Missing physical hand mesh: {mesh}")
        geom = next((item for item in candidates
                     if item.get("contype", "1") != "0" or item.get("conaffinity", "1") != "0"),
                    candidates[-1])
        name = f"{side}_physical_hand_{index}"
        geom.set("name", name)
        names.append(name)
    return names
