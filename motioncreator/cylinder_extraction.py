"""Tool-agnostic scoop path for lifting a lying cylinder out of a box, in its cross-section plane.

The cylinder lies with its axis across the robot (world Y), so every tool moves in the XZ plane.
Positions here are 2-D ``(x, z)`` world coordinates in metres and a tool attitude is one angle:
the direction of the tool's forward axis, measured from +X toward +Z. The TCP is a point on the
blade's top face; how a specific tool maps onto it is left to the caller, so the path does not
change when a tool is swapped.

The path follows the drawn idea: from the grasp point, wedge down the slit between the target and
its robot-side neighbour, follow the target's surface round to its bottom, slide on under it, tilt
the tip up so the cylinder rolls onto the blade, lift it clear of the box and pull it toward the
robot. Every stage is cut into small steps so it can be moved "a little at a time".
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_PARAMS = {
    # Grasp point, as the editor defines it (shared ramen grasp coefficients).
    'clearance_m': .10, 'height_m': 0., 'pitch_deg': 0.,
    # Approach: hover above the slit between the target and its robot-side neighbour, pointing into the box.
    'approach_height_m': .03,
    # Side of the target the blade goes down: -1 = robot side, +1 = far side (the box's upper wall).
    'side': -1.,
    # Wedge: the blade's top face runs this far inside the target surface, prying the slit and the gap under it open.
    'wedge_m': .001,
    # Motion under the target: 0 = wrap (tip follows the target's surface round to its bottom),
    # 1 = pry (lever the blade about the slit mouth so the part below it swings under the target),
    # 2 = clamp (both tools go down opposite sides - left tool robot side, right tool far side - squeeze
    # the target through the wedge and lift it by friction; `side` is ignored).
    'mode': 1.,
    # Pry: how deep the tip goes down the slit and where the lever pivots, as fractions of the radius
    # below / above the target centre, and how far the blade swings.
    'insert_depth': .3, 'pivot_up': .6, 'pry_deg': 35.,
    # Wrap: follow the target's surface from its robot side (180 deg) round to its bottom (270 deg).
    'wrap_start_deg': 180., 'wrap_end_deg': 270.,
    # Slide the tip on under the target, as a fraction of its radius past the centre line.
    'slide_extent': .35,
    # Tilt: raise the tip (positive, toward the box opening) so the target rolls back onto the blade.
    'tilt_deg': 15., 'tilt_rise_m': .01,
    # Lift out of the box and carry toward the robot.
    'lift_margin_m': .03, 'carry_m': .12,
    # Resolution: every stage is split into steps no longer than these.
    'step_m': .02, 'step_deg': 10.,
    'speed_m_s': .06, 'turn_deg_s': 40., 'min_step_s': .15,
}
LIMITS = {
    'clearance_m': (0., .4), 'height_m': (-.3, .3), 'pitch_deg': (-90., 90.), 'approach_height_m': (0., .2), 'side': (-1., 1.),
    'mode': (0., 2.), 'insert_depth': (-.5, 1.2), 'pivot_up': (-.5, 1.5), 'pry_deg': (0., 80.),
    'wedge_m': (-.02, .03), 'wrap_start_deg': (90., 270.), 'wrap_end_deg': (180., 300.), 'slide_extent': (-.5, 1.),
    'tilt_deg': (-70., 70.), 'tilt_rise_m': (0., .1), 'lift_margin_m': (0., .3), 'carry_m': (0., .5),
    'step_m': (.002, .3), 'step_deg': (.5, 90.), 'speed_m_s': (.005, 1.), 'turn_deg_s': (1., 360.),
    'min_step_s': (.02, 5.),
}


def _xz(vector):
    return np.array([float(vector[0]), float(vector[2])])


def _rotation(item):
    return Rotation.from_quat(np.asarray(item['quaternion_xyzw'], dtype=float)).as_matrix()


def _size(item):
    return np.asarray(item.get('collision_size') or item['size'], dtype=float)


def _rot90(vector):
    return np.array([-vector[1], vector[0]])


def _lying_cylinder(item):
    """Return (radius, axis) when the object is a cylinder-like body lying across the robot."""
    if item.get('shape') not in ('cylinder', 'box') and item.get('collision_shape') not in ('convex_hull', 'cylinder'):
        return None
    size, rotation = _size(item), _rotation(item)
    if item.get('shape') == 'cylinder':
        axis_index, radius = 2, size[0] / 2
    else:
        axis_index = int(np.argmax(size))
        others = [value for index, value in enumerate(size) if index != axis_index]
        if abs(others[0] - others[1]) > .01 * max(others) or size[axis_index] < 1.5 * max(others):
            return None
        radius = min(others) / 2
    axis = rotation[:, axis_index]
    return (float(radius), axis) if abs(axis[1]) > .95 else None


def _spans_plane(item, y):
    rotation, half = _rotation(item), _size(item) / 2
    extent = np.abs(rotation[1]) @ half
    return abs(float(item['position'][1]) - y) <= extent + 1e-9


def _robot_pose(project):
    """Pelvis position and xyzw quaternion of the project's current pose (or first keyframe), else the origin."""
    frames = project.get('keyframes') or []
    qpos = project.get('current_qpos') or (frames[0].get('qpos') if frames else None)
    if not qpos or len(qpos) < 7:
        return [0., 0., 0.], [0., 0., 0., 1.]
    return [float(v) for v in qpos[:3]], [float(qpos[4]), float(qpos[5]), float(qpos[6]), float(qpos[3])]


def scene_2d(project, target_id=None, robot_xz=None):
    """Cut the project's scene at the target cylinder: circles, rectangles and box walls in XZ."""
    objects = project.get('scene_objects') or []
    robot_position, robot_quaternion = _robot_pose(project)
    if robot_xz is None:
        robot_xz = (robot_position[0], robot_position[2])
    cylinders = [(item, _lying_cylinder(item)) for item in objects if not item.get('fixed')]
    cylinders = [(item, found) for item, found in cylinders if found]
    if not cylinders:
        raise ValueError('로봇 좌우 방향으로 누운 원통 오브젝트가 없습니다.')
    if target_id is None:
        # Default: the cylinder closest to the robot among the highest ones.
        target_item = max(cylinders, key=lambda entry: (round(entry[0]['position'][2], 2), -entry[0]['position'][0]))[0]
    else:
        target_item = next((item for item, _ in cylinders if item['id'] == target_id), None)
        if target_item is None:
            raise ValueError('대상 원통을 찾을 수 없습니다. 로봇 좌우 방향으로 누운 원통만 지원합니다.')
    plane_y = float(target_item['position'][1])
    center = _xz(target_item['position'])
    toward_robot = np.array([robot_xz[0], 0.]) - np.array([center[0], 0.])
    toward_robot = toward_robot / np.linalg.norm(toward_robot) if np.linalg.norm(toward_robot) > 1e-9 else np.array([-1., 0.])
    circles, rectangles, walls, box = [], [], [], None
    for item in objects:
        if not _spans_plane(item, plane_y) or not item.get('visible', True) and not item.get('fixed'):
            continue
        rotation = _rotation(item)
        if abs(rotation[1, 1]) < .99:  # only bodies rotated about Y keep a rigid 2-D section
            continue
        angle = math.atan2(rotation[2, 0], rotation[0, 0])
        size = _size(item)
        base = {'id': item['id'], 'name': item.get('name', item['id']), 'mass_kg': float(item.get('mass_kg', 1.)),
                'friction': float(item.get('friction', .7)), 'fixed': bool(item.get('fixed'))}
        cylinder = _lying_cylinder(item) if not item.get('fixed') else None
        if cylinder:
            circles.append({**base, 'center': _xz(item['position']).tolist(), 'radius': cylinder[0]})
        elif item.get('shape') == 'open_box':
            thickness = float(item.get('wall_thickness_m', .02))
            width, height = size[0], size[2]
            local = [((0., -height / 2 + thickness / 2), (width / 2, thickness / 2)),
                     ((-width / 2 + thickness / 2, 0.), (thickness / 2, height / 2)),
                     ((width / 2 - thickness / 2, 0.), (thickness / 2, height / 2))]
            origin = _xz(item['position'])
            e, u = np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])
            for (a, b), half in local:
                walls.append({**base, 'center': (origin + a * e + b * u).tolist(), 'half': list(half), 'angle': angle})
            box = {'id': item['id'], 'center': origin.tolist(), 'angle': angle, 'width': float(width),
                   'height': float(height), 'thickness': thickness, 'e': e.tolist(), 'u': u.tolist()}
        elif item.get('shape') == 'box':
            entry = {**base, 'center': _xz(item['position']).tolist(), 'half': [size[0] / 2, size[2] / 2], 'angle': angle}
            (walls if item.get('fixed') else rectangles).append(entry)
    target = next(circle for circle in circles if circle['id'] == target_item['id'])
    if box is not None:
        offset = center - np.array(box['center'])
        e, u = np.array(box['e']), np.array(box['u'])
        inside = abs(offset @ e) < box['width'] / 2 and abs(offset @ u) < box['height'] / 2
        if not inside:
            box = None
    if box is not None:
        up = np.array(box['u'])
        rim = float((np.array(box['center']) + up * box['height'] / 2 - center) @ up)
    else:
        up, rim = np.array([0., 1.]), 0.
    far = _rot90(up) * -1.  # (far, up) is right-handed like (x, z)
    if far @ toward_robot > 0:
        far = -far
    return {'format': 'motioncreator.cylinder-section.v1', 'plane_y': plane_y, 'target_id': target['id'],
            'target': target, 'circles': circles, 'rectangles': rectangles, 'walls': walls, 'box': box,
            'frame': {'origin': center.tolist(), 'far': far.tolist(), 'up': up.tolist(), 'rim_up_m': rim},
            'toward_robot': toward_robot.tolist(),
            # The robot pose the editor's grasp-point calculation takes (pelvis position, xyzw quaternion).
            'robot_position': robot_position, 'robot_quaternion_xyzw': robot_quaternion}


def with_positions(scene, positions):
    """Return the scene with circles moved to `positions` ({id: [x, z]}), e.g. after settling under gravity.

    Paths must be planned from where the cylinders actually rest, which can differ from the authored
    layout by a centimetre in a packed box. The frame's rim height is kept relative to the box.
    """
    moved = [{**circle, 'center': [float(v) for v in positions.get(circle['id'], circle['center'])]} for circle in scene['circles']]
    target = next(circle for circle in moved if circle['id'] == scene['target_id'])
    frame = dict(scene['frame'])
    old_center, new_center = np.array(frame['origin']), np.array(target['center'])
    frame['rim_up_m'] = float(frame['rim_up_m'] - (new_center - old_center) @ np.array(frame['up']))
    frame['origin'] = new_center.tolist()
    return {**scene, 'circles': moved, 'target': target, 'frame': frame}


def _params(overrides):
    params = dict(DEFAULT_PARAMS)
    for key, value in (overrides or {}).items():
        if key not in DEFAULT_PARAMS:
            raise ValueError(f'Unknown scoop parameter: {key}')
        value = float(value)
        low, high = LIMITS[key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be within {low}..{high}')
        params[key] = value
    return params


def plan_scoop(scene, overrides=None):
    """Return TCP waypoints (position, tool angle) for scooping the scene's target cylinder."""
    params = _params(overrides)
    frame = scene['frame']
    center, far, up = (np.array(frame[key]) for key in ('origin', 'far', 'up'))
    radius = float(scene['target']['radius'])
    toward_robot = np.array(scene['toward_robot'])

    def world(a, b):
        return center + a * far + b * up

    def local_angle(degrees):  # tool angle for a direction given in the (far, up) frame
        direction = math.cos(math.radians(degrees)) * far + math.sin(math.radians(degrees)) * up
        return math.atan2(direction[1], direction[0])

    # Grasp point as the editor's candidateRamenGraspPoints puts it (roll and yaw are 0 here): the pitch
    # tilts the inward axis toward world up, whichever side the robot stands on.
    start = center + toward_robot * (radius + params['clearance_m']) + np.array([0., params['height_m']])
    pitch = math.radians(params['pitch_deg'])
    direction = -toward_robot * math.cos(pitch) + np.array([0., 1.]) * math.sin(pitch)
    start_angle = math.atan2(direction[1], direction[0])
    if params['mode'] > 1.5:
        return _plan_clamp(scene, params, center, far, up, radius, toward_robot, world, local_angle, start, start_angle)
    rho = radius - params['wedge_m']  # the blade's top face (TCP) runs on this circle
    # The path is authored for the robot side and mirrored across the target's vertical for the far
    # side; there the tool is rolled over so its top face still faces the target.
    mirror = params['side'] > 0
    sign = -1. if mirror else 1.

    def on_circle(degrees):  # wrap angles use the robot-side convention: 180 = beside, 270 = bottom
        phi = math.radians(degrees)
        point = world(sign * rho * math.cos(phi), rho * math.sin(phi))
        return point, local_angle(degrees + 90. if not mirror else 90. - degrees)  # tangent of the motion

    wrap_start, wrap_start_angle = on_circle(params['wrap_start_deg'])
    above = wrap_start + up * (radius + params['approach_height_m'] - (wrap_start - center) @ up)
    slide_end = world(sign * radius * params['slide_extent'], -rho)
    tilt_end = slide_end + up * params['tilt_rise_m']
    tilted = local_angle(params['tilt_deg'] if not mirror else 180. - params['tilt_deg'])
    keys = [('approach', '틈 위로 이동', above, wrap_start_angle)]
    if params['mode'] < .5:
        keys.append(('insert', '틈으로 쐐기 삽입', wrap_start, wrap_start_angle))
        span = params['wrap_end_deg'] - params['wrap_start_deg']
        arc_steps = max(1, math.ceil(abs(span) / params['step_deg']))
        for step in range(1, arc_steps + 1):  # the arc is sampled here; straight chords would cut into the target
            point, tangent = on_circle(params['wrap_start_deg'] + span * step / arc_steps)
            keys.append(('wrap', f'원통 둘레 따라 감기 {step}/{arc_steps}', point, tangent))
        keys += [('slide', '원통 아래로 밀어 넣기', slide_end, local_angle(0. if not mirror else 180.)),
                 ('tilt', '날 끝 들기', tilt_end, tilted)]
        top = tilt_end
    else:
        # Lever about the slit mouth: the blade below the pivot swings under the target and lifts it,
        # the handle above swings into the free space over the neighbour.
        inserted = world(sign * rho * -1., -radius * params['insert_depth'])
        pivot = world(sign * rho * -1., radius * params['pivot_up'])
        keys.append(('insert', '틈으로 쐐기 삽입', inserted, wrap_start_angle))
        pry = math.radians(params['pry_deg']) * (1. if not mirror else -1.)
        pry_steps = max(1, math.ceil(params['pry_deg'] / params['step_deg']))
        for step in range(1, pry_steps + 1):
            angle = pry * step / pry_steps
            c, s_ = math.cos(angle), math.sin(angle)
            offset = inserted - pivot
            point = pivot + np.array([c * offset[0] - s_ * offset[1], s_ * offset[0] + c * offset[1]])
            keys.append(('pry', f'지렛대로 젖혀 들기 {step}/{pry_steps}', point, wrap_start_angle + angle))
        top = keys[-1][2]
        tilted = keys[-1][3]
    lift = max(0., frame['rim_up_m']) + radius + params['lift_margin_m'] - float((top - center) @ up) + radius
    lifted = top + up * lift
    carried = lifted + toward_robot * params['carry_m']
    keys += [('lift', '상자 밖으로 들어올리기', lifted, tilted),
             ('carry', '몸쪽으로 가져오기', carried, tilted)]
    waypoints = _discretize([(start, start_angle, keys)], params)[0]
    return {'format': 'motioncreator.cylinder-scoop.v1', 'params': params, 'waypoints': waypoints,
            'stage_steps': _stage_steps(waypoints), 'lift_m': float(lift), 'target_id': scene['target_id'],
            # With the blade rolled over, the top face is on the tool's right side in this plane.
            'blade_flip': bool(mirror), 'blades': [{'side': 'robot' if not mirror else 'far', 'waypoints': waypoints, 'flip': bool(mirror)}]}


def _stage_steps(waypoints):
    stages = dict.fromkeys(w['stage'] for w in waypoints[1:])
    return {stage: sum(1 for w in waypoints if w['stage'] == stage) for stage in stages}


def _discretize(tools, params):
    """Cut each tool's keyed path into small steps; all tools share step counts and durations.

    `tools` is a list of (start position, start angle, [(stage, label, position, angle), ...]) with the
    same stage sequence, so tools that move together stay in lock-step keyframe by keyframe.
    """
    lists = [[{'stage': 'start', 'label': '파지점', 'tcp': start.tolist(), 'angle': angle, 'duration': .1}]
             for start, angle, _ in tools]
    current = [(start, angle) for start, angle, _ in tools]
    for index in range(len(tools[0][2])):
        moves = []
        for (position, angle), (_, _, keys) in zip(current, tools):
            stage, label, target, target_angle = keys[index]
            turn = (target_angle - angle + math.pi) % (2 * math.pi) - math.pi
            moves.append((stage, label, position, angle, target, turn, float(np.linalg.norm(target - position))))
        if all(distance < 1e-9 and abs(turn) < 1e-9 for *_, turn, distance in moves):
            continue
        steps = max(max(1, math.ceil(distance / params['step_m']), math.ceil(abs(math.degrees(turn)) / params['step_deg']))
                    for *_, turn, distance in moves)
        seconds = max(max(params['min_step_s'], distance / steps / params['speed_m_s'],
                          abs(math.degrees(turn)) / steps / params['turn_deg_s']) for *_, turn, distance in moves)
        for waypoints, (stage, label, position, angle, target, turn, _) in zip(lists, moves):
            for step in range(1, steps + 1):
                fraction = step / steps
                waypoints.append({'stage': stage, 'label': label if stage in ('wrap', 'pry') or steps == 1 else f'{label} {step}/{steps}',
                                  'tcp': (position + (target - position) * fraction).tolist(),
                                  'angle': angle + turn * fraction, 'duration': seconds})
        current = [(target, angle + turn) for _, _, _, angle, target, turn, _ in moves]
    return lists


def _plan_clamp(scene, params, center, far, up, radius, toward_robot, world, local_angle, start, start_angle):
    """Left tool down the robot-side slit, right tool down the far side; squeeze, then lift together."""
    rho = radius - params['wedge_m']
    frame = scene['frame']
    tools = []
    for sign, flip in ((1., False), (-1., True)):  # robot side, far side (rolled over)
        mouth = world(-sign * rho, 0.)
        above = mouth + up * (radius + params['approach_height_m'])
        inserted = world(-sign * rho, -radius * params['insert_depth'])
        pointing = local_angle(-90.)
        lift = max(0., frame['rim_up_m']) + radius + params['lift_margin_m'] + radius * params['insert_depth']
        lifted = inserted + up * lift
        carried = lifted + toward_robot * params['carry_m']
        keys = [('approach', '틈 위로 이동', above, pointing),
                ('insert', '양쪽 틈으로 쐐기 삽입', inserted, pointing),
                ('squeeze', '양쪽에서 조이기', inserted + far * sign * params['wedge_m'], pointing),
                ('lift', '상자 밖으로 들어올리기', lifted + far * sign * params['wedge_m'], pointing),
                ('carry', '몸쪽으로 가져오기', carried + far * sign * params['wedge_m'], pointing)]
        tools.append((start, start_angle, keys, flip))
    lists = _discretize([(s_, a, k) for s_, a, k, _ in tools], params)
    blades = [{'side': side, 'waypoints': waypoints, 'flip': flip}
              for side, waypoints, (*_, flip) in zip(('robot', 'far'), lists, tools)]
    return {'format': 'motioncreator.cylinder-scoop.v1', 'params': params, 'waypoints': lists[0],
            'stage_steps': _stage_steps(lists[0]), 'lift_m': float(lift),
            'target_id': scene['target_id'], 'blade_flip': False, 'blades': blades}
