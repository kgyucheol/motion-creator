import numpy as np

from motioncreator.motion import compile_motion, new_project, validate_project
from motioncreator.policy_preview import _interaction_feedback, build_model
from motioncreator.ramen_sequence import plan_ramen_sequence
from motioncreator.robot import Robot


SETTINGS = {
    'approach_clearance_m': .04,
    'lift_height_m': .05,
    'carry_offset_m': [-.03, 0., 0.],
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


def _project(monkeypatch):
    monkeypatch.setenv('MOTIONCREATOR_MODEL', 'g1-tools')
    robot = Robot()
    data = robot.data(robot.home)
    hands = [robot.point(data, f'{side}_hand')[0] for side in ('left', 'right')]
    center = np.mean(hands, axis=0)
    item = {
        'id': 'ramen-bundle', 'name': '라면 묶음', 'shape': 'cylinder',
        'position': center.tolist(), 'quaternion_xyzw': [0., 0., 0., 1.],
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
        'gap_approach', 'left_insert', 'right_insert', 'support_hold',
        'vertical_lift', 'extract', 'carry_hold',
    ]
    assert all(frame['interaction']['object_id'] == 'ramen-bundle' for frame in result['keyframes'])
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
