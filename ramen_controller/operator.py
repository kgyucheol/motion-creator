"""OperatorPort implementations: keyboard for the manual positioning stage, scripted for tests/sim."""
from __future__ import annotations

import select
import sys
import threading

import numpy as np

LINEAR_STEP, ANGULAR_STEP = 0.1, 0.2          # per key press
MAX_LINEAR, MAX_ANGULAR = 0.5, 1.0            # decoupled_wbc command limits


class ScriptedOperator:
    def __init__(self):
        self.velocity = np.zeros(3)
        self.confirmed = False
        self._reset = False

    def nav(self) -> np.ndarray:
        return self.velocity

    def position_confirmed(self) -> bool:
        return self.confirmed

    def reset_requested(self) -> bool:
        flag, self._reset = self._reset, False
        return flag

    def request_reset(self) -> None:
        self._reset = True


class KeyboardOperator(ScriptedOperator):
    """W/S forward/back, A/D left/right, Q/E turn, Z stop, Enter confirm position, R reset.

    `handle_key` holds the logic; `start()` feeds it from a raw terminal on a background thread.
    """

    def handle_key(self, key: str) -> None:
        key = key.lower()
        v = self.velocity
        if key == "w":
            v[0] += LINEAR_STEP
        elif key == "s":
            v[0] -= LINEAR_STEP
        elif key == "a":
            v[1] += LINEAR_STEP
        elif key == "d":
            v[1] -= LINEAR_STEP
        elif key == "q":
            v[2] += ANGULAR_STEP
        elif key == "e":
            v[2] -= ANGULAR_STEP
        elif key == "z":
            v[:] = 0.0
        elif key in ("\n", "\r", " "):
            self.confirmed = True
        elif key == "r":
            self.request_reset()
            self.confirmed = False
            v[:] = 0.0
        v[:2] = np.clip(v[:2], -MAX_LINEAR, MAX_LINEAR)
        v[2] = np.clip(v[2], -MAX_ANGULAR, MAX_ANGULAR)
        v[np.abs(v) < 1e-9] = 0.0                     # avoid -0.0 / float residue after +step -step

    def start(self) -> None:
        import termios
        import tty
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        self._stop = threading.Event()

        def loop():
            try:
                tty.setcbreak(fd)
                while not self._stop.is_set():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        self.handle_key(sys.stdin.read(1))
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
