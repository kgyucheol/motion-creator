import copy
import json

import mujoco
import numpy as np
import pytest

from motioncreator.grasp import GRASP_FORMAT, fit_two_hand_grasp
from motioncreator.grip_geometry import (GRIP_PAD_FORMAT, grip_pad_contact_anchor,
                                         grip_pad_contact_normal, grip_pad_half_size,
                                         grip_pad_rotation)
from motioncreator.motion import new_project, save_bundle, validate_project
from motioncreator.policy_preview import build_model, compile_preview_motion, simulate
from motioncreator.robot import Robot


def box_at_hands():
    return {'id': 'grasp-box', 'name': 'Grip box', 'shape': 'box', 'position': [.35, 0., .85],
            'quaternion_xyzw': [0., 0., 0., 1.], 'size': [.2, .2, .2], 'mass_kg': 1.,
            'friction': .8, 'color': '#886633', 'opacity': .7, 'visible': True}


def event():
    return {'format': GRASP_FORMAT, 'object_id': 'grasp-box',
            'left_surface_uv': [0., 0.], 'right_surface_uv': [0., 0.],
            'inward_offset_m': .01, 'closure_seconds': .1,
            'target_force_n': 2., 'max_force_n': 100.}


def fitted_project():
    robot = Robot()
    project = new_project(robot)
    project['scene_objects'] = [box_at_hands()]
    result = fit_two_hand_grasp(robot, project['keyframes'][0]['qpos'],
                                project['keyframes'][0]['pins'], box_at_hands(), event())
    project['keyframes'][0].update(qpos=result['state']['qpos'], grasp=result['grasp'], duration=.1)
    return robot, project, result


def test_grasp_fit_uses_imported_pose_and_stores_physics_closure(tmp_path):
    robot, project, result = fitted_project()
    validate_project(robot, project)
    assert result['solver']['contact']['target_error_mm'] < 1
    assert result['solver']['closure']['target_error_mm'] < 1
    assert result['grasp']['contact_anchor'] == 'finger_wrist_pad'
    assert len(result['grasp']['closure_qpos']) == robot.model.nq
    assert not np.allclose(result['state']['qpos'][7:], result['grasp']['closure_qpos'][7:])
    json.dumps(project, allow_nan=False)
    bundle = save_bundle(robot, project, fps=15, directory=tmp_path)
    reopened = json.loads((tmp_path / bundle['project_file']).read_text())
    assert reopened['keyframes'][0]['grasp']['object_id'] == 'grasp-box'
    validate_project(robot, reopened)


def test_grasp_uses_visible_finger_to_wrist_box_pads():
    robot, project, _ = fitted_project()
    model = build_model(robot, project)
    for side in ('left', 'right'):
        geom = model.geom(f'{side}_preview_grip')
        assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
        assert geom.contype == 0 and geom.conaffinity == 0
        np.testing.assert_allclose(geom.size, grip_pad_half_size())
        geom_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(geom_matrix, model.geom_quat[geom.id])
        np.testing.assert_allclose(geom_matrix.reshape(3, 3), grip_pad_rotation(side), atol=1e-7)
        contact = geom.pos + grip_pad_contact_normal(side) * geom.size[1]
        np.testing.assert_allclose(contact, grip_pad_contact_anchor(side), atol=1e-7)
        actual = {model.geom(f'{side}_physical_hand_{index}').id for index in range(2)}
        paired = [{int(model.pair_geom1[index]), int(model.pair_geom2[index])}
                  for index in range(model.npair)]
        assert geom.id not in set().union(*paired)
        assert any(actual & pair for pair in paired)
    assert GRIP_PAD_FORMAT.endswith('four-finger-wrist-pad.v2')


def test_grip_pad_spans_four_fingers_but_excludes_thumb():
    # Measured wrist-local bounds of the four long rubber-hand fingers. The
    # thumb begins at z=47.4 mm and must remain outside the auxiliary surface.
    four_finger_z = np.array([-.0430289, .03604572])
    thumb_z_min = .04737843
    left_support_points = np.array([
        [.02899996, -.02383722, -.002614],     # wrist-yaw link
        [.16557813, -.04289171, -.01588409],  # index/ring finger surfaces
        [.16991816, -.04487687, .00655421],
        [.15710959, -.04102468, -.0383332],
        [.16417232, -.0435376, .03054256],
    ])
    for side in ('left', 'right'):
        rotation = grip_pad_rotation(side)
        center = grip_pad_contact_anchor(side)
        half = grip_pad_half_size()
        corners = np.array([center + rotation @ np.array([x, 0., z])
                            for x in (-half[0], half[0]) for z in (-half[2], half[2])])
        assert corners[:, 2].min() <= four_finger_z[0]
        assert corners[:, 2].max() >= four_finger_z[1]
        assert corners[:, 2].max() < thumb_z_min

        # The angled centerline overlaps both measured support points: the
        # wrist-yaw link near x=29 mm and the four-finger tips near x=170 mm.
        endpoints = np.array([center + rotation[:, 0] * value
                              for value in (-half[0], half[0])])
        assert endpoints[0, 0] < .029
        assert endpoints[1, 0] > .169

        support_points = left_support_points.copy()
        if side == 'right':
            support_points[:, 1] *= -1
        local = (support_points - center) @ rotation
        assert np.max(np.abs(local[:, 1])) < .0015
        assert np.max(np.abs(local[:, 0])) <= half[0]
        assert np.max(np.abs(local[:, 2])) <= half[2]


def test_physics_preview_prepends_grasp_closure_and_reports_contact():
    robot, project, _ = fitted_project()
    times, poses, grasp = compile_preview_motion(robot, project, fps=50)
    assert grasp is not None
    assert times[0] == 0
    assert times[-1] == pytest.approx(.2)
    np.testing.assert_allclose(poses[0], project['keyframes'][0]['qpos'])
    np.testing.assert_allclose(poses[5], project['keyframes'][0]['grasp']['closure_qpos'])
    result = simulate(project, controller='pd')
    assert result['summary']['grasp']['object_id'] == 'grasp-box'
    assert result['summary']['grasp']['bilateral_contact']
    assert set(result['summary']['grasp']) >= {'bilateral_contact', 'target_reached', 'force_limit_exceeded'}
    json.dumps(result, allow_nan=False)


def test_moved_box_requires_grasp_refit():
    robot, project, _ = fitted_project()
    moved = copy.deepcopy(project)
    moved['scene_objects'][0]['position'][0] += .1
    validate_project(robot, moved)
    with pytest.raises(ValueError, match='다시 맞춰'):
        compile_preview_motion(robot, moved)
