"""Compile the supplied 29-axis G1 gripper URDF without changing its assets."""
import copy
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'assets/g1_scoop_endsupport/g1_29dof_rev_1_0_scoop_endsupport.urdf'
NSE_ROOT = ROOT / 'assets/tools'
GENERATED_ROOT = NSE_ROOT / '.generated'
FLANGE_X = .0415  # wrist flange / hand frame, same attach point as the stock hand
# (left, right) tools of the editor model; the original URDF carries the scoop on the right wrist
# and the end support on the left, and the editor mirrors both to the opposite hand.
DEFAULT_TOOLS = ('scoop', 'end_support')
# `native` is the hand each tool's URDF was authored for; on the other hand it is mirrored across y.
TOOLS = {
    'scoop': {'label': '주걱 (기본)', 'native': 'right', 'kind': 'legacy'},
    'end_support': {'label': '받침 (기본)', 'native': 'left', 'kind': 'legacy'},
    'nse_scoop': {'label': 'NSE 주걱', 'native': 'left', 'kind': 'nse', 'urdf': 'nse_left_scoop.urdf'},
    'nse_spatula': {'label': 'NSE 스패출러', 'native': 'right', 'kind': 'nse', 'urdf': 'nse_right_spatula.urdf'},
}


def parse_tools(value=None):
    """Return (left, right) tool ids from 'left:right' text, defaulting to the stock pair."""
    if not value:
        return DEFAULT_TOOLS
    parts = tuple(str(value).split(':'))
    if len(parts) != 2 or any(part not in TOOLS for part in parts):
        raise ValueError(f'Unknown tool selection: {value!r}; choose from {", ".join(TOOLS)}')
    return parts


def tools_key(tools):
    return ':'.join(tools)


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


def _default_tool_source():
    source = ET.parse(URDF).getroot()
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
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') * [-1, 1, -1]
                origin.set('rpy', ' '.join(map(str, rpy)))
    return source, None


def _mirror_element(element):
    """Mirror a link or joint authored for one hand so it fits the other (y -> -y)."""
    for origin in element.iter('origin'):
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') * [1, -1, 1]
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') * [-1, 1, -1]
        origin.set('xyz', ' '.join(map(str, xyz)))
        origin.set('rpy', ' '.join(map(str, rpy)))
    for mesh in element.iter('mesh'):
        mesh.set('scale', '1 -1 1')
    for inertia in element.iter('inertia'):
        for key in ('ixy', 'iyz'):
            inertia.set(key, str(-float(inertia.get(key, '0'))))


def _tool_template(tool, source):
    """Return (links, joints, mount_joint_name, tcp_link, tcp_xyz, mesh_files) authored for the tool's native hand."""
    spec = TOOLS[tool]
    native = spec['native']
    if spec['kind'] == 'legacy':
        prefix = f'{native}_{tool}'
        links = [copy.deepcopy(link) for link in source.findall('link') if link.get('name') in (f'{prefix}_link', f'{prefix}_tcp')]
        joints = [copy.deepcopy(joint) for joint in source.findall('joint') if joint.get('name') in (f'{prefix}_joint', f'{prefix}_tcp_joint')]
        tcp_xyz = [float(v) for v in next(j for j in joints if j.get('name') == f'{prefix}_tcp_joint').find('origin').get('xyz').split()]
        return links, joints, f'{prefix}_joint', f'{prefix}_link', tcp_xyz
    urdf = ET.parse(NSE_ROOT / spec['urdf']).getroot()
    tip = next(j for j in urdf.findall('joint') if j.find('child').get('link') == f'{native}_tip')
    tip_link, tip_xyz = tip.find('parent').get('link'), [float(v) for v in tip.find('origin').get('xyz').split()]
    links = [copy.deepcopy(link) for link in urdf.findall('link') if link.get('name') != f'{native}_tip']
    joints = [copy.deepcopy(joint) for joint in urdf.findall('joint') if joint is not tip]
    root_link = f'{native}_adapter'
    mount = ET.fromstring(
        f'<joint name="{native}_{tool}_joint" type="fixed"><parent link="{native}_wrist_yaw_link"/>'
        f'<child link="{root_link}"/><origin xyz="{FLANGE_X} 0 0" rpy="{1.5707963267948966 if native == "left" else -1.5707963267948966} 0 0"/></joint>')
    return links, [mount, *joints], f'{native}_{tool}_joint', tip_link, tip_xyz


def _custom_tool_source(tools):
    """Build the base G1 URDF with each hand carrying the chosen tool; also link every mesh uniquely."""
    source = ET.parse(URDF).getroot()
    templates = {tool: _tool_template(tool, source) for tool in set(tools)}
    for element in [*source.findall('link'), *source.findall('joint')]:
        if any(word in element.get('name', '') for word in ('scoop', 'end_support')):
            source.remove(element)
    key = hashlib.sha256(tools_key(tools).encode()).hexdigest()[:12]
    meshdir = GENERATED_ROOT / key / 'meshes'
    meshdir.mkdir(parents=True, exist_ok=True)
    for path in (URDF.parent / 'meshes').iterdir():  # stock G1 meshes, reachable from the same directory
        link = meshdir / path.name
        if not link.exists():
            link.symlink_to(path)
    handles, used = {}, set()
    for side, tool in zip(('left', 'right'), tools):
        links, joints, mount_name, tcp_link, tcp_xyz = templates[tool]
        links, joints = [copy.deepcopy(e) for e in links], [copy.deepcopy(e) for e in joints]
        native = TOOLS[tool]['native']
        suffix = '' if tool not in used else f'__{side}'
        used.add(tool)
        mirrored = side != native
        rename = {e.get('name'): e.get('name') + suffix for e in (*links, *joints)}
        for element in (*links, *joints):
            element.set('name', rename[element.get('name')])
            if mirrored:
                _mirror_element(element)
        for joint in joints:
            for tag in ('parent', 'child'):
                node = joint.find(tag)
                node.set('link', rename.get(node.get('link'), node.get('link')))
            parent = joint.find('parent')
            if joint.get('name') == mount_name + suffix and mirrored:
                parent.set('link', f'{side}_wrist_yaw_link')
        for number, mesh in enumerate(m for link in links for m in link.iter('mesh')):
            name = Path(mesh.get('filename')).name
            origin = Path(mesh.get('filename').replace('package://nse_gripper/', ''))
            target = f'{tool}_{side}_{"col_" if "collision" in origin.parts else ""}{name}'
            sourcefile = URDF.parent / 'meshes' / name if TOOLS[tool]['kind'] == 'legacy' else NSE_ROOT / origin
            link = meshdir / target
            if not link.exists():
                link.symlink_to(sourcefile)
            mesh.set('filename', f'meshes/{target}')
        for element in links:
            for material in element.iter('material'):
                if material.get('name') and material.find('color') is not None:
                    material.set('name', f'{tool}_{side}_{material.get("name")}')
        for element in (*links, *joints):
            source.append(element)
        body = rename[tcp_link]
        position = np.array(tcp_xyz) * ([1, -1, 1] if mirrored else [1, 1, 1])
        handles[f'{side}_hand'] = (body, tuple(float(v) for v in position), f'{side} {tool} TCP')
    source.find('mujoco/compiler').set('meshdir', str(meshdir))
    return source, (handles, meshdir)


def tool_model_xml(tools=DEFAULT_TOOLS, force_generic=False):
    tools = tuple(tools)
    source, custom = _default_tool_source() if tools == DEFAULT_TOOLS and not force_generic else _custom_tool_source(tools)
    meshdir = custom[1] if custom else URDF.parent / 'meshes'
    if custom is None:
        source.find('mujoco/compiler').set('meshdir', str(meshdir))
    source.find('mujoco/compiler').set('fusestatic', 'false')
    model = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding='unicode'))
    with tempfile.NamedTemporaryFile(suffix='.xml') as output:
        mujoco.mj_saveLastXML(output.name, model)
        root = ET.parse(output.name).getroot()
    root.find('compiler').set('meshdir', str(meshdir))
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
    # Fixed TCP links are fused by the URDF importer; locate each handle in its
    # tool link so both the mount rotation and TCP orientation reach FK/IK.
    handles = {}
    if custom is not None:
        handles = custom[0]
    for side, original, tool in (('left', 'right', 'scoop'), ('right', 'left', 'end_support')) if custom is None else ():
        tcp = source.find(f"joint[@name='{original}_{tool}_tcp_joint']/origin")
        position = tuple(float(value) for value in tcp.get('xyz').split())
        handles[f'{side}_hand'] = (f'{original}_{tool}_link', position, f'{side} {tool} TCP')
    for index, geom in enumerate(root.findall('.//geom')):
        if geom.get('name') is None:
            geom.set('name', f'urdf_geom_{index}')
    return root, handles
