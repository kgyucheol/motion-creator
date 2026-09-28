import numpy as np
from scipy.spatial.transform import Rotation

from motioncreator.motion import compile_motion, new_project, validate_project
from motioncreator.policy_preview import _interaction_feedback, build_model
from motioncreator.ramen_sequence import plan_ramen_sequence
from motioncreator.robot import Robot


SETTINGS = {
    'approach_clearance_m': .04,
    'lift_height_m': .05,
    'extraction_distance_m': .03,
    'extraction_cycles': 3,
    'carry_tilt_deg': 12.,
    'phase_seconds': .3,
    'insertion_seconds': .4,
    'hold_seconds': .2,
    'force_limit_n': 35.,
    'lateral_stiffness_n_per_m': 420.,
    'insertion_stiffness_n_per_m': 90.,
    'translation_damping_ns_per_m': 28.,
    'orientation_stiffness_nm_per_rad': 38.,
    'orientation_damping_nms_per_rad': 4.5,
    'maximum_feedback_torque_fraction': .35,
}


def _project(monkeypatch, tilt_deg=0.):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    data = robot.data(robot.home)
    hands = [robot.point(data, f'{side}_hand')[0] for side in ('left', 'right')]
    center = np.mean(hands, axis=0)
    object_rotation = (Rotation.from_euler('x', tilt_deg, degrees=True)
                       * Rotation.from_euler('y', 90, degrees=True)).as_quat()
    item = {
        'id': 'ramen-bundle', 'name': '라면 묶음', 'shape': 'cylinder',
        'position': center.tolist(), 'quaternion_xyzw': object_rotation.tolist(),
        'size': [.1, .1, .2], 'mass_kg': .2, 'friction': .7,
        'color': '#ffffff', 'opacity': 1., 'visible': True, 'fixed': False,
    }
    result = plan_ramen_sequence(robot, robot.home, ['left_foot', 'right_foot'], [], item, SETTINGS)
    project = new_project(robot)
    project['scene_objects'] = [item]
    project['keyframes'] = result['keyframes']
    return robot, project, result


def test_ramen_sequence_uses_taught_tcp_pose_and_compiles_cartesian_path(monkeypatch):
    robot, project, result = _project(monkeypatch)
    assert [frame['interaction']['phase'] for frame in result['keyframes']] == [
        'default_pose', 'object_align', 'left_insert', 'right_insert', 'load_check',
        'angle_relax_1', 'lift_1', 'pull_1', 'angle_relax_2', 'lift_2', 'pull_2',
        'angle_relax_3', 'lift_3', 'pull_3', 'carry_hold',
    ]
    assert all(frame['interaction']['object_id'] == 'ramen-bundle' for frame in result['keyframes'])
    np.testing.assert_allclose(result['keyframes'][0]['qpos'], robot.home)
    reference = result['object_reference']
    np.testing.assert_allclose(reference['center_world'], project['scene_objects'][0]['position'])
    np.testing.assert_allclose(reference['box_up_world'], [0., 0., 1.], atol=1e-7)
    assert reference['diameter_m'] == .1
    preinsert = np.asarray(result['keyframes'][1]['interaction']['tcp_targets']['left']['position'])
    inserted = np.asarray(result['keyframes'][2]['interaction']['tcp_targets']['left']['position'])
    assert np.dot(preinsert - inserted, reference['box_up_world']) >= reference['diameter_m'] - 1e-7
    validate_project(robot, project)
    motion = compile_motion(robot, project, fps=20)
    assert len(motion['qpos']) > len(result['keyframes'])
    assert motion['max_pin_error_mm'] < 1.


def test_interaction_impedance_produces_bounded_restoring_torque(monkeypatch):
    robot, project, result = _project(monkeypatch)
    model = build_model(robot, project)
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:36] = result['keyframes'][0]['qpos']
    mujoco.mj_forward(model, data)
    joints = np.array([model.joint(name).id for name in robot.names])
    joint_dofs = model.jnt_dofadr[joints]
    limits = model.jnt_actfrcrange[joints]
    interaction = result['keyframes'][0]['interaction']
    interaction['tcp_targets']['left']['position'][2] += .02
    torque, metrics = _interaction_feedback(model, data, robot, interaction, joint_dofs, limits)
    assert np.max(np.abs(torque[15:22])) > 0
    assert metrics['position_error_mm'] >= 19
    allowed = np.max(np.abs(limits), axis=1) * SETTINGS['maximum_feedback_torque_fraction']
    assert np.all(np.abs(torque) <= allowed + 1e-9)


def test_tilted_bundle_rotates_vertical_insertion_axis_with_box(monkeypatch):
    _, _, result = _project(monkeypatch, tilt_deg=20.)
    reference = result['object_reference']
    expected_up = Rotation.from_euler('x', 20., degrees=True).apply([0., 0., 1.])
    np.testing.assert_allclose(reference['box_up_world'], expected_up, atol=1e-7)
    before = np.asarray(result['keyframes'][1]['interaction']['tcp_targets']['left']['position'])
    after = np.asarray(result['keyframes'][2]['interaction']['tcp_targets']['left']['position'])
    displacement = before - after
    np.testing.assert_allclose(displacement / np.linalg.norm(displacement), expected_up, atol=1e-6)
    assert np.linalg.norm(displacement) >= reference['diameter_m'] - 1e-7
