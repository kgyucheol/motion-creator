from .robot import Robot, FEET
from .motion import new_project


def crouch_demo(robot: Robot, depth=.16):
    project = new_project(robot, 'G1_crouch_reference')
    target = robot.home[:3].copy()
    target[2] -= depth
    crouch, info = robot.solve(robot.home, robot.home, 'pelvis', target, FEET, max_nfev=80)
    if not info['converged']:
        raise ValueError(f'Crouch depth cannot be reached: {info}')
    project['keyframes'] += [
        {'name': 'Crouch', 'duration': 2., 'qpos': crouch.tolist(), 'pins': list(FEET)},
        {'name': 'Stand', 'duration': 2., 'qpos': robot.home.tolist(), 'pins': list(FEET)},
    ]
    return project
