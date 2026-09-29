"""Name the robot's real hand geoms for explicit object contact pairs."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco


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


def body_contact_geom_names(root: ET.Element, side_hand_bodies=("left_wrist_yaw_link", "right_wrist_yaw_link")) -> list[str]:
    """Name the collision geoms of every robot body except the hands and tools.

    Hands and tools already have their own explicit object pairs.  Returning the
    rest lets pelvis, legs, torso, head and arm links push scene objects too, so a
    held object can be braced against the forearms and chest.  Contype/conaffinity
    are read from a compiled model because MJCF classes can hide them in the XML.
    """
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # Map XML geoms to compiled geoms body by body: a body's geoms are contiguous and in XML order.
    geoms = []
    for body_element in root.find("worldbody").iter("body"):
        elements = body_element.findall("geom")
        body = model.body(body_element.get("name"))
        if len(elements) != int(body.geomnum[0]):
            raise ValueError(f"Cannot map collision geoms of body {body_element.get('name')}")
        geoms.extend((int(body.geomadr[0]) + offset, element) for offset, element in enumerate(elements))
    excluded = {model.body(name).id for name in side_hand_bodies}

    def in_hand(body: int) -> bool:
        while body:
            if body in excluded:
                return True
            body = int(model.body_parentid[body])
        return False

    names = []
    for index, geom in geoms:
        body = int(model.geom_bodyid[index])
        if body == 0 or in_hand(body) or not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        name = geom.get("name") or f"body_collision_{index}"
        geom.set("name", name)
        names.append(name)
    return names
