"""Compile the supplied 29-axis G1 gripper URDF without changing its assets."""
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'assets/g1_scoop_endsupport/g1_29dof_rev_1_0_scoop_endsupport.urdf'


@lru_cache(maxsize=1)
def head_camera_spec():
    """Return the D435 mount transform declared by the gripper URDF.

    The URDF defines only the fixed link transform. It does not include a
    camera optical frame or RGB intrinsics, so callers must supply those from
    the calibrated camera driver before projecting pixels into 3-D.
    """
    source = ET.parse(URDF).getroot()
    joint = source.find("joint[@name='d435_joint']")
    if joint is None or joint.get('type') != 'fixed':
        raise ValueError('The gripper URDF does not declare a fixed d435_joint')
    origin = joint.find('origin')
    parent = joint.find('parent')
    child = joint.find('child')
    return {
        'link': child.get('link'),
        'parent_link': parent.get('link'),
        'mount_xyz_m': [float(value) for value in origin.get('xyz', '0 0 0').split()],
        'mount_rpy_rad': [float(value) for value in origin.get('rpy', '0 0 0').split()],
        'optical_frame_declared': False,
        'intrinsics_declared': False,
    }


def tool_model_xml():
    source = ET.parse(URDF).getroot()
    source.find('mujoco/compiler').set('meshdir', str(URDF.parent / 'meshes'))
    source.find('mujoco/compiler').set('fusestatic', 'false')
    for link in source.findall('link'):
        if not any(word in link.get('name', '') for word in ('scoop', 'end_support')):
            continue
        for origin in link.findall('.//origin'):
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') * [1, -1, 1]
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') * [-1, 1, -1]
            origin.set('xyz', ' '.join(map(str, xyz)))
            origin.set('rpy', ' '.join(map(str, rpy)))
        for mesh in link.findall('.//mesh'):
            mesh.set('scale', '1 -1 1')
        for inertia in link.findall('.//inertia'):
            for key in ('ixy', 'iyz'):
                inertia.set(key, str(-float(inertia.get(key, '0'))))
    for joint in source.findall('joint'):
        if any(word in joint.get('name', '') for word in ('scoop', 'end_support')):
            parent = joint.find('parent')
            if parent.get('link') == 'right_wrist_yaw_link':
                parent.set('link', 'left_wrist_yaw_link')
            elif parent.get('link') == 'left_wrist_yaw_link':
                parent.set('link', 'right_wrist_yaw_link')
            origin = joint.find('origin')
            if origin is not None:
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') * [1, -1, 1]
                origin.set('xyz', ' '.join(map(str, xyz)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding='unicode'))
    with tempfile.NamedTemporaryFile(suffix='.xml') as output:
        mujoco.mj_saveLastXML(output.name, model)
        root = ET.parse(output.name).getroot()
    root.find('compiler').set('meshdir', str(URDF.parent / 'meshes'))
    world = root.find('worldbody')
    pelvis = world.find("body[@name='pelvis']")
    pelvis.set('pos', '0 0 .79')
    ET.SubElement(pelvis, 'freejoint', name='floating_base')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='10 10 .1',
                  rgba='.12 .15 .18 1', friction='.8 .005 .0001')
    motors = ET.SubElement(root, 'actuator')
    for joint in source.findall('joint'):
        if joint.get('type') not in ('revolute', 'continuous'):
            continue
        name = joint.get('name')
        compiled = root.find(f".//joint[@name='{name}']")
        effort = float(joint.find('limit').get('effort'))
        compiled.set('actuatorfrcrange', f'{-effort} {effort}')
        compiled.set('armature', '.01')
        ET.SubElement(motors, 'motor', name=name, joint=name, gear='1')
    # Fixed TCP links are fused by the URDF importer; retain their exact offsets.
    handles = {}
    for side, original, tool in (('left', 'right', 'scoop'), ('right', 'left', 'end_support')):
        mount = source.find(f"joint[@name='{original}_{tool}_joint']/origin")
        tcp = source.find(f"joint[@name='{original}_{tool}_tcp_joint']/origin")
        position = [float(a) + float(b) for a, b in zip(mount.get('xyz').split(), tcp.get('xyz').split())]
        handles[f'{side}_hand'] = (f'{side}_wrist_yaw_link', tuple(position), f'{side} {tool} TCP')
    for index, geom in enumerate(root.findall('.//geom')):
        if geom.get('name') is None:
            geom.set('name', f'urdf_geom_{index}')
    return root, handles
