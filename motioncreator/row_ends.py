"""Hands at the two ends of an object (a row of cups lying on its side, a box, a sphere...).

A card is authored against a `reference` object. Each hand keeps its recorded place relative to the end of the object
it is nearest to, so the card can be re-aimed at another object (a detected row, any length and heading): the hand goes
to the same end of the new object, the same distance outside it, and the hands turn with the object's heading.

The ends lie on the object's grasp axis: its longest horizontal axis (a lying cylinder's own axis; on a tie, the one
closest to world y), or world y for a sphere or an upright cylinder, whose width is then the diameter. The row frame has
x along that axis (pointing toward world +y, the robot's left), z up and y = z cross x; the sign and the frame depend only
on the axis, so a cylinder's roll about its axis never matters.
"""
import numpy as np
from scipy.spatial.transform import Rotation

SHAPES = ('box', 'open_box', 'cylinder', 'sphere')
MAX_AXIS_TILT = .9        # |axis z| above this is not a horizontal axis


def row_of(item):
    """(center, unit axis toward +y, length) of a scene object's grasp axis (see the module docstring)."""
    shape, size = item['shape'], np.asarray(item['size'], dtype=float)
    rotation = Rotation.from_quat(item['quaternion_xyzw'])
    local = () if shape == 'sphere' else (2,) if shape == 'cylinder' else (0, 1, 2)
    axes = [(rotation.apply(np.eye(3)[i]), float(size[i])) for i in local]
    axes = [(axis, length) for axis, length in axes if abs(axis[2]) <= MAX_AXIS_TILT]
    if axes:
        axis, length = max(axes, key=lambda pair: (round(pair[1] * 1000), abs(pair[0][1])))
    else:
        axis, length = np.array([0., 1., 0.]), float(size[0])
    if (axis[1] if abs(axis[1]) > 1e-6 else axis[0]) < 0:
        axis = -axis
    return np.asarray(item['position'], dtype=float), axis, length


def row_frame(axis):
    z = np.array([0., 0., 1.]) - axis[2] * axis
    z /= np.linalg.norm(z)
    return np.column_stack([axis, np.cross(z, axis), z])


def heading_deg(axis):
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def row_goals(reference, row, hands, tcp=None):
    """Goals {hand: (position, Rotation)} that put each hand at its recorded place relative to its end of `row`.

    `reference` and `row` are `row_of` results; `hands` maps a hand name to its recorded (position, rotation matrix).
    Also returns {hand: (end sign, offset in the row frame with x counted outward from the end)}.

    With `tcp` = (inward, lateral, up) in meters the recorded distance is ignored: each hand's TCP goes to that point
    measured from the end of its row (inward along the axis, lateral along the row y axis, up), and the recorded
    hand only decides which end it takes and its orientation."""
    ref_center, ref_axis, ref_length = reference
    center, axis, length = row
    ref_frame, frame = row_frame(ref_axis), row_frame(axis)
    turn = Rotation.from_euler('z', heading_deg(axis) - heading_deg(ref_axis), degrees=True)
    goals, relations = {}, {}
    for hand, (position, matrix) in hands.items():
        end = 1. if (position - ref_center) @ ref_axis >= 0 else -1.
        if tcp is None:
            offset = ref_frame.T @ (position - (ref_center + end * ref_axis * ref_length / 2))
            offset[0] *= end
        else:
            offset = np.array([-tcp[0], tcp[1], tcp[2]])
        target = center + end * axis * length / 2 + frame @ np.array([end * offset[0], offset[1], offset[2]])
        goals[hand] = (target, turn * Rotation.from_matrix(matrix))
        relations[hand] = (end, offset)
    return goals, relations
