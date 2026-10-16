import mujoco
import numpy as np
import pytest
import trimesh
import xml.etree.ElementTree as ET
from fastapi.testclient import TestClient

from motioncreator import server, tool_model
from motioncreator.robot import Robot, FEET
from motioncreator.tool_model import DEFAULT_TOOLS, TOOLS, parse_tools, tool_model_xml
from motioncreator.tool_section import tool_section


def compiled(tools, **kwargs):
    root, handles = tool_model_xml(tools, **kwargs)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, handles


def tcp_pose(model, data, handle):
    body, offset, _ = handle
    b = model.body(body).id
    rotation = data.xmat[b].reshape(3, 3)
    return data.xpos[b] + rotation @ np.array(offset), rotation


def test_generic_builder_reproduces_the_stock_tool_pair():
    stock = compiled(DEFAULT_TOOLS)
    generic = compiled(DEFAULT_TOOLS, force_generic=True)
    assert stock[0].nq == generic[0].nq == 36 and stock[0].nbody == generic[0].nbody
    assert stock[0].body_mass.sum() == pytest.approx(generic[0].body_mass.sum(), rel=1e-9)
    for side in ('left_hand', 'right_hand'):
        (p1, r1), (p2, r2) = tcp_pose(*stock[:2], stock[2][side]), tcp_pose(*generic[:2], generic[2][side])
        np.testing.assert_allclose(p1, p2, atol=1e-9)
        np.testing.assert_allclose(r1, r2, atol=1e-9)


def test_parse_tools_rejects_unknown_ids():
    assert parse_tools(None) == DEFAULT_TOOLS and parse_tools('nse_scoop:end_support') == ('nse_scoop', 'end_support')
    for bad in ('scoop', 'scoop:unknown', 'a:b:c'):
        with pytest.raises(ValueError):
            parse_tools(bad)


@pytest.mark.parametrize('tools', [('nse_scoop', 'nse_spatula'), ('nse_spatula', 'nse_scoop'), ('scoop', 'scoop'),
                                   ('end_support', 'scoop'), ('nse_scoop', 'end_support'),
                                   ('rubber_hand', 'rubber_hand'), ('rubber_hand', 'nse_spatula'),
                                   ('rubber_hand_tip', 'rubber_hand_tip')])
def test_any_tool_pair_keeps_the_joint_layout_and_supports_ik(tools):
    stock = Robot('g1-tools', DEFAULT_TOOLS)
    robot = Robot('g1-tools', tools)
    assert robot.names == stock.names and robot.model.nq == 36 and robot.model.nu == 29
    # Motions stay portable between pairs, while caches key on the mounted tools.
    assert robot.fingerprint == stock.fingerprint and robot.compatible_fingerprints == stock.compatible_fingerprints
    assert robot.cache_key != stock.cache_key
    assert tools[0] in robot.handles['left_hand'][2] and tools[1] in robot.handles['right_hand'][2]
    home = robot.data(robot.home)
    for side in ('left', 'right'):
        target = robot.point(home, f'{side}_hand')[0] + [.01, 0, .01]
        _, result = robot.solve(robot.home, robot.home, selected_targets={f'{side}_hand': target}, pins=FEET)
        assert result['target_error_mm'] < 1 and not result['rejected']


def test_same_tool_on_both_hands_is_mirrored():
    robot = Robot('g1-tools', ('scoop', 'scoop'))
    data = robot.data(robot.home)
    left, right = robot.point(data, 'left_hand'), robot.point(data, 'right_hand')
    np.testing.assert_allclose(right[0], left[0] * [1, -1, 1], atol=2e-5)
    assert len({name for name in (robot.handles['left_hand'][0], robot.handles['right_hand'][0])}) == 2


def test_stock_tool_tcps_reach_the_mesh_tips_and_support_plate_enters_lab_collision():
    robot = Robot('g1-tools', DEFAULT_TOOLS)
    mesh_dir = tool_model.URDF.parent / 'meshes'
    scoop = trimesh.load(mesh_dir / 'right_scoop_body.STL')
    support = trimesh.load(mesh_dir / 'left_end_support_body.STL')
    scoop_tip = np.asarray(robot.handles['left_hand'][1])
    support_tip = np.asarray(robot.handles['right_hand'][1])
    np.testing.assert_allclose(support_tip, scoop_tip, atol=1e-9)
    assert scoop_tip[0] == pytest.approx(scoop.bounds[1, 0], abs=1e-6)
    assert support_tip[0] == pytest.approx(support.bounds[1, 0], abs=1e-6)
    plate = support.vertices[(support.vertices[:, 0] > .2) & (abs(support.vertices[:, 1]) < .01)]
    assert plate[:, 2].max() == pytest.approx(support_tip[2], abs=1e-6)
    section = tool_section(robot, 'right')
    assert section['tool_id'] == 'end_support'
    assert section['area_m2'] > .002  # the plate box is included alongside the wall mesh
    assert min(point[0] for outline in section['outlines'] for point in outline) < -.24


def test_nse_tools_are_mounted_on_their_own_hands_and_hang_inward():
    robot = Robot('g1-tools', ('nse_scoop', 'nse_spatula'))
    data = robot.data(robot.home)
    left, right = robot.point(data, 'left_hand'), robot.point(data, 'right_hand')
    assert left[0][1] > 0 > right[0][1]  # left tool on the robot's left, right tool on the right
    # The tool's top face (link z) points at the torso: left tool -Y, right tool +Y.
    np.testing.assert_allclose(left[1][:, 2], [0, -1, 0], atol=2e-3)
    np.testing.assert_allclose(right[1][:, 2], [0, 1, 0], atol=2e-3)
    # The blade tip reaches about 0.32 m beyond the hand frame.
    wrist = robot.data(robot.home).xpos[robot.model.body('left_wrist_yaw_link').id]
    assert 0.3 < np.linalg.norm(left[0] - wrist) < 0.4


def test_tool_endpoints_swap_the_robot_and_refuse_bad_requests(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    monkeypatch.setenv('MOTIONCREATOR_TOOLS', ':'.join(DEFAULT_TOOLS))
    monkeypatch.setattr(Robot, 'export_visual', lambda self, path: None)  # keep the served GLB untouched
    monkeypatch.setattr(server, 'robot', Robot('g1-tools', DEFAULT_TOOLS))
    client = TestClient(server.app)
    listing = client.get('/api/tools').json()
    assert listing['tools'] == list(DEFAULT_TOOLS) and {o['id'] for o in listing['options']} == set(TOOLS)
    assert client.post('/api/tools', json={'left': 'nope', 'right': 'scoop'}).status_code == 422
    first = client.get('/api/init').json()
    assert client.post('/api/tools', json={'left': 'nse_scoop', 'right': 'nse_spatula'}).json()['tools'] == ['nse_scoop', 'nse_spatula']
    after = client.get('/api/init').json()
    assert after['tools'] == ['nse_scoop', 'nse_spatula'] and after['visual_revision'] != first['visual_revision']
    assert client.post('/api/tools', json={'left': 'nse_scoop', 'right': 'nse_spatula'}).status_code == 200
    import os
    assert os.environ['MOTIONCREATOR_TOOLS'] == 'nse_scoop:nse_spatula'
    monkeypatch.setattr(server, 'robot', Robot('g1'))
    assert client.post('/api/tools', json={'left': 'scoop', 'right': 'scoop'}).status_code == 409


def test_rubber_hand_tool_matches_the_stock_g1_hand():
    base, hands = Robot('g1'), Robot('g1-tools', ('rubber_hand', 'rubber_hand'))
    for side in ('left', 'right'):
        np.testing.assert_allclose(hands.point(hands.data(hands.home), f'{side}_hand')[0],
                                   base.point(base.data(base.home), f'{side}_hand')[0], atol=2e-5)
    assert hands.model.nq == base.model.nq


def test_rubber_hand_tip_puts_the_tcp_five_millimetres_inside_the_middle_fingertip_of_both_hands():
    palm, tip = Robot('g1-tools', ('rubber_hand', 'rubber_hand')), Robot('g1-tools', ('rubber_hand_tip', 'rubber_hand_tip'))
    mesh = trimesh.load(tool_model.G1_MESHES / 'left_rubber_hand.STL')
    farthest = mesh.vertices[mesh.vertices[:, 0] > mesh.vertices[:, 0].max() - 1e-4]
    for side, mirror in (('left', 1.), ('right', -1.)):
        offset = np.asarray(tip.handles[f'{side}_hand'][1])
        assert abs(offset[0] - (mesh.vertices[:, 0].max() - .005)) < 5e-4
        assert np.linalg.norm((offset * [1, mirror, 1])[1:] - farthest[:, 1:].mean(axis=0)) < 8e-3      # on the fingertip, not beside it
        assert offset[1] * mirror < 0
        # same joints and mesh as the palm model: the two TCPs differ only by the offset along the hand
        d_palm, d_tip = palm.data(palm.home), tip.data(tip.home)
        palm_point, rotation = palm.point(d_palm, f'{side}_hand')
        tip_point, _ = tip.point(d_tip, f'{side}_hand')
        local = rotation.T @ (tip_point - palm_point)
        assert abs(local[0] - (offset[0] - np.asarray(palm.handles[f'{side}_hand'][1])[0])) < 1e-6
    assert palm.fingerprint == tip.fingerprint
