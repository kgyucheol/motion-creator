"""Shared finger-tip-to-wrist contact-pad geometry for authored and physical grasps."""
from __future__ import annotations

import numpy as np


GRIP_PAD_FORMAT = "motioncreator.finger-wrist-pad.v1"
GRIP_PAD_LENGTH_M = .13
GRIP_PAD_THICKNESS_M = .006
GRIP_PAD_HEIGHT_M = .09
GRIP_PAD_CENTER_X_M = .1075
GRIP_PAD_CENTER_Z_M = .01
GRIP_PAD_SURFACE_Y_M = {"left": -.045, "right": .045}


def grip_pad_center(side: str) -> np.ndarray:
    """Wrist-local center of the thin collision box behind the inner contact face."""
    sign = -1. if side == "left" else 1.
    surface = GRIP_PAD_SURFACE_Y_M[side]
    return np.array([GRIP_PAD_CENTER_X_M, surface - sign * GRIP_PAD_THICKNESS_M / 2,
                     GRIP_PAD_CENTER_Z_M])


def grip_pad_half_size() -> np.ndarray:
    return np.array([GRIP_PAD_LENGTH_M, GRIP_PAD_THICKNESS_M, GRIP_PAD_HEIGHT_M]) / 2


def grip_pad_contact_anchor(side: str) -> np.ndarray:
    """Wrist-local center of the box-facing plane used by grasp IK."""
    return np.array([GRIP_PAD_CENTER_X_M, GRIP_PAD_SURFACE_Y_M[side], GRIP_PAD_CENTER_Z_M])
