"""Cartesian paths between two pose keyframes, like a robot arm's MoveL and MoveC.

`joint` (MoveJ, the default) blends the joint angles. `linear` (MoveL) moves both hand TCPs along straight lines and
`circular` (MoveC) along circular arcs through a via point, turning each hand's orientation with a slerp. The
keyframe pose itself is not changed: only what happens between two keyframes.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

KINDS = ('joint', 'linear', 'circular')
MOVE_HANDS = ('left_hand', 'right_hand')
MAX_VIA_M = 1.0


def validate_path(spec):
    """Return the normalised `path` of a pose keyframe or raise ValueError (the message is shown to the user)."""
    if not isinstance(spec, dict):
        raise ValueError('Keyframe path must be an object')
    kind = spec.get('kind', 'joint')
    if kind not in KINDS:
        raise ValueError('Keyframe path kind must be joint, linear or circular')
    out = {'kind': kind}
    if kind == 'circular':
        via = np.asarray(spec.get('via_offset', [0., 0., .05]), dtype=float)
        if via.shape != (3,) or not np.isfinite(via).all() or np.abs(via).max() > MAX_VIA_M:
            raise ValueError('The arc offset must be three finite values within 1 m')
        out['via_offset'] = via.tolist()
    return out


class Arc:
    """The circular arc from `start` to `end` through `via`; a straight line when the three points are on one line."""

    def __init__(self, start, via, end):
        self.start, self.end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        ab, ac = np.asarray(via, dtype=float) - self.start, self.end - self.start
        normal = np.cross(ab, ac)
        squared = float(normal @ normal)
        self.straight = squared < 1e-12 * max(1., float(ab @ ab) * float(ac @ ac))
        if self.straight:
            return
        self.center = self.start + (float(ac @ ac) * np.cross(normal, ab) + float(ab @ ab) * np.cross(ac, normal)) / (2 * squared)
        self.axis = normal / np.sqrt(squared)
        self.radial = self.start - self.center
        to_via, to_end = np.asarray(via, dtype=float) - self.center, self.end - self.center

        def angle(vector):
            return float(np.arctan2(self.axis @ np.cross(self.radial, vector), self.radial @ vector))
        sense = 1. if angle(to_via) >= 0 else -1.
        via_angle = angle(to_via) * sense                                  # in (0, pi]: the way round to the via point
        end_angle = (angle(to_end) * sense) % (2 * np.pi)
        # going the same way round must meet the via point before the end; if not, the arc runs the other way
        self.sweep = sense * end_angle if end_angle > via_angle else -sense * (2 * np.pi - end_angle)

    def point(self, s):
        if self.straight:
            return self.start + s * (self.end - self.start)
        return self.center + Rotation.from_rotvec(self.axis * self.sweep * s).apply(self.radial)


def hand_path(kind, pa, pb, via_offset=None):
    """Return `at(s) -> {hand: (position, quaternion xyzw)}` for the two hands, s in [0, 1] from pose `pa` to pose `pb`."""
    arcs, turns = {}, {}
    for hand in MOVE_HANDS:
        start, end = pa[hand][0], pb[hand][0]
        via = (start + end) / 2 + np.asarray(via_offset if via_offset is not None else [0., 0., 0.], dtype=float)
        arcs[hand] = Arc(start, via, end) if kind == 'circular' else Arc(start, (start + end) / 2, end)
        turns[hand] = Slerp([0., 1.], Rotation.from_matrix(np.stack([pa[hand][1], pb[hand][1]])))

    def at(s):
        return {hand: (arcs[hand].point(s), turns[hand]([s])[0].as_quat()) for hand in MOVE_HANDS}
    return at
