import copy
import json

import numpy as np
import pytest

from motioncreator.grasp import GRASP_FORMAT, fit_two_hand_grasp
from motioncreator.motion import new_project, save_bundle, validate_project
from motioncreator.policy_preview import compile_preview_motion, simulate
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
    assert result['grasp']['contact_anchor'] == 'lower_palm_wrist'
    assert len(result['grasp']['closure_qpos']) == robot.model.nq
    assert not np.allclose(result['state']['qpos'][7:], result['grasp']['closure_qpos'][7:])
    json.dumps(project, allow_nan=False)
    bundle = save_bundle(robot, project, fps=15, directory=tmp_path)
    reopened = json.loads((tmp_path / bundle['project_file']).read_text())
    assert reopened['keyframes'][0]['grasp']['object_id'] == 'grasp-box'
    validate_project(robot, reopened)


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
