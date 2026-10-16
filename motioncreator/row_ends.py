"""Hands at the two ends of a row of cups lying on its side.

A card can be authored against a stand-in box (the `reference` row). Each hand keeps its recorded place relative to the
end of the row it is nearest to, so the card can be re-aimed at another row (a detected one, any length and heading):
the hand goes to the same end of the new row, the same distance outside it, and the hands turn with the row's heading.

The row frame has x along the long axis (pointing toward world +y, the robot's left), z up and y = z cross x. Rows must
lie roughly horizontal; the sign and the frame depend only on the axis, so a cylinder's roll about its axis never matters.
"""
import numpy as np
from scipy.spatial.transform import Rotation

STAND_IN_SHAPES = ('box', 'open_box', 'cylinder')
MAX_AXIS_TILT = .9        # |axis z| above this is not a lying row


def row_of(item):
    """(center, unit axis toward +y, length) of a scene object's long axis, or None for a sphere or an upright object."""
    shape, size = item['shape'], np.asarray(item['size'], dtype=float)
    if shape == 'sphere':
        return None
    local = 2 if shape == 'cylinder' else int(np.argmax(size))
    axis = Rotation.from_quat(item['quaternion_xyzw']).apply(np.eye(3)[local])
    if abs(axis[2]) > MAX_AXIS_TILT:
        return None
    if (axis[1] if abs(axis[1]) > 1e-6 else axis[0]) < 0:
        axis = -axis
    return np.asarray(item['position'], dtype=float), axis, float(size[local])


def row_frame(axis):
    z = np.array([0., 0., 1.]) - axis[2] * axis
    z /= np.linalg.norm(z)
    return np.column_stack([axis, np.cross(z, axis), z])


def heading_deg(axis):
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def row_goals(reference, row, hands):
    """Goals {hand: (position, Rotation)} that put each hand at its recorded place relative to its end of `row`.

    `reference` and `row` are `row_of` results; `hands` maps a hand name to its recorded (position, rotation matrix).
    Also returns {hand: (end sign, offset in the reference row frame with x counted outward from the end)}."""
    ref_center, ref_axis, ref_length = reference
    center, axis, length = row
    ref_frame, frame = row_frame(ref_axis), row_frame(axis)
    turn = Rotation.from_euler('z', heading_deg(axis) - heading_deg(ref_axis), degrees=True)
    goals, relations = {}, {}
    for hand, (position, matrix) in hands.items():
        end = 1. if (position - ref_center) @ ref_axis >= 0 else -1.
        offset = ref_frame.T @ (position - (ref_center + end * ref_axis * ref_length / 2))
        offset[0] *= end
        target = center + end * axis * length / 2 + frame @ np.array([end * offset[0], offset[1], offset[2]])
        goals[hand] = (target, turn * Rotation.from_matrix(matrix))
        relations[hand] = (end, offset)
    return goals, relations
