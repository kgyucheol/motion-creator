"""Shared four-finger-to-wrist virtual alignment-plane geometry."""
from __future__ import annotations

import numpy as np


GRIP_PAD_FORMAT = "motioncreator.four-finger-wrist-pad.v2"

# The inner envelopes of the rubber-hand mesh were measured in the wrist-yaw
# frame. The thumb is the short component around z=+58 mm and is deliberately
# excluded. A plane fit through the other four distal components and the
# wrist-yaw link gives y = ax + bz + c on the left hand. The mirrored right
# hand uses the same fit with the y component reflected.
_PLANE_DY_DX = -.14328804111206117
_PLANE_DY_DZ = -.025285375784806514
_CONTACT_CENTER_LEFT = np.array([.09960862, -.03317451, -.00349159])

GRIP_PAD_LENGTH_M = .14444387196808448
GRIP_PAD_THICKNESS_M = .006
GRIP_PAD_HEIGHT_M = .082


def _side_sign(side: str) -> float:
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side: {side}")
    return -1. if side == "left" else 1.


def grip_pad_rotation(side: str) -> np.ndarray:
    """Return the pad-to-wrist rotation matrix.

    Local X follows wrist-to-fingertips, local Z spans the four non-thumb
    fingers, and the side-dependent local Y face points toward the object.
    """
    sign = _side_sign(side)
    contact_normal = np.array([_PLANE_DY_DX, sign, _PLANE_DY_DZ])
    contact_normal /= np.linalg.norm(contact_normal)
    local_y = sign * contact_normal
    local_x = np.array([1., 0., 0.])
    local_x -= np.dot(local_x, local_y) * local_y
    local_x /= np.linalg.norm(local_x)
    local_z = np.cross(local_x, local_y)
    return np.column_stack((local_x, local_y, local_z))


def grip_pad_quaternion_wxyz(side: str) -> np.ndarray:
    """Return the MuJoCo quaternion for :func:`grip_pad_rotation`."""
    rotation = grip_pad_rotation(side)
    w = np.sqrt(1. + np.trace(rotation)) / 2.
    x = (rotation[2, 1] - rotation[1, 2]) / (4. * w)
    y = (rotation[0, 2] - rotation[2, 0]) / (4. * w)
    z = (rotation[1, 0] - rotation[0, 1]) / (4. * w)
    return np.array([w, x, y, z])


def grip_pad_contact_normal(side: str) -> np.ndarray:
    """Wrist-local unit normal from the hand toward the grasped object."""
    sign = _side_sign(side)
    return sign * grip_pad_rotation(side)[:, 1]


def grip_pad_center(side: str) -> np.ndarray:
    """Center of the thin, non-colliding visualization box."""
    return grip_pad_contact_anchor(side) - grip_pad_contact_normal(side) * GRIP_PAD_THICKNESS_M / 2.


def grip_pad_half_size() -> np.ndarray:
    return np.array([GRIP_PAD_LENGTH_M, GRIP_PAD_THICKNESS_M, GRIP_PAD_HEIGHT_M]) / 2


def grip_pad_contact_anchor(side: str) -> np.ndarray:
    """Center of the box-facing plane used by grasp IK, in wrist coordinates."""
    center = _CONTACT_CENTER_LEFT.copy()
    center[1] *= -_side_sign(side)
    return center
