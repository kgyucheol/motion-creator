"""Ported from g1-live-studio g1live/arm_sdk_writer.py (8411a3d); the config import, the module path and the optional waist changed.

Child process: publishes arm targets on `rt/arm_sdk`, the channel that lets a program move the arms while the robot's own motion
controller keeps it standing. It is the only place that talks to the Unitree SDK for the replay; the studio sends it JSON lines.

    python -m motioncreator.replay.arm_sdk_writer <interface>

stdin   {"op": "engage", "joints": [...], ...settings}   hold the commanded joints where they are and fade the weight 0 -> 1;
                                                   joints: the motors the targets are for, 15..28 (the arms, default) or 12..28
                                                   (the waist too, as robot-studio does); settings: kp_arm, kd_arm, kp_wrist,
                                                   kd_wrist, hz (defaults and limits in config.ARM_SDK_DEFAULTS / _LIMITS)
        {"op": "target", "q": [n], "tau": [n]}     joint targets (rad) and feed-forward torque (Nm), one per joint; also the heartbeat
        {"op": "release"}                          fade the weight 1 -> 0, then exit
stdout  {"event": "ready" | "engaged" | "released" | "error", ...}

The weight is `motor_cmd[29].q` (kNotUsedJoint0): 0 = the robot's controller owns the arms, 1 = this program does. If the studio goes
silent (no message for ARM_SDK_WATCHDOG_S once engaged, or stdin closes) the weight is faded back to 0 on its own.
Gains and the joint table are those of Unitree's `G1_29_ArmController` (unitree_lerobot): every joint is filled, the ones not
commanded hold the pose they had when engaging (the waist at 300/3, as robot-studio sends its waist yaw).
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

import numpy as np

from motioncreator.replay import config as C

WEIGHT_JOINT = 29                                    # kNotUsedJoint0
WRIST = (19, 20, 21, 26, 27, 28)
ARM = range(15, 29)
WEAK_NON_ARM = (4, 10)                               # ankle pitch


JOINT_CHOICES = (list(range(15, 29)), list(range(12, 29)))     # the arms, or the waist and the arms


def validate_joints(raw) -> list[int]:
    joints = list(range(15, 29)) if raw is None else [int(j) for j in raw]
    if joints not in JOINT_CHOICES:
        raise ValueError("제어 관절은 팔 14축(15~28) 또는 허리와 팔 17축(12~28)이어야 합니다")
    return joints


def validate_settings(raw: dict) -> dict:
    """The arm_sdk settings with defaults filled in; ValueError (shown to the user) if one is outside the studio's limits."""
    out = {**C.ARM_SDK_DEFAULTS, **{k: raw[k] for k in C.ARM_SDK_DEFAULTS if k in raw}}
    for key, (lo, hi) in C.ARM_SDK_LIMITS.items():
        value = float(out[key])
        if not lo <= value <= hi:                                     # NaN fails this too
            raise ValueError(f"{key}는 {lo:g}~{hi:g} 범위여야 합니다 (입력 {value:g})")
        out[key] = value
    if out["hz"] not in C.ARM_SDK_HZ_CHOICES:
        raise ValueError(f"송신 주기는 {', '.join(map(str, C.ARM_SDK_HZ_CHOICES))} Hz 중 하나여야 합니다")
    out["hz"] = int(out["hz"])
    return out


def gains(joint: int, settings: dict | None = None) -> tuple[float, float]:
    """(kp, kd) of one motor. Arms and wrists follow `settings` (default: unitree_lerobot's); the other joints hold the pose they had."""
    s = settings or C.ARM_SDK_DEFAULTS
    if joint in WRIST:
        return s["kp_wrist"], s["kd_wrist"]
    if joint in ARM:
        return s["kp_arm"], s["kd_arm"]
    return (80.0, 3.0) if joint in WEAK_NON_ARM else (300.0, 3.0)


class Weight:
    """Linear ramp of the arm_sdk weight between 0 and 1."""

    def __init__(self):
        self.value, self._from, self._to, self._t0, self._dur = 0.0, 0.0, 0.0, 0.0, 0.0

    def ramp_to(self, target: float, seconds: float, now: float) -> None:
        self._from, self._to, self._t0, self._dur = self.value, target, now, max(seconds, 1e-6)

    def step(self, now: float) -> float:
        tau = min(max((now - self._t0) / self._dur, 0.0), 1.0)
        self.value = self._to if tau >= 1.0 else self._from + (self._to - self._from) * tau
        return self.value

    @property
    def done(self) -> bool:
        return self.value == self._to


def clip_step(current: np.ndarray, target: np.ndarray, vmax: float, dt: float) -> np.ndarray:
    """Move from `current` towards `target` by at most vmax*dt on the joint that has the farthest to go (the others scale with it)."""
    delta = target - current
    scale = np.max(np.abs(delta)) / (vmax * dt)
    return current + delta / max(scale, 1.0)


def limit_command(previous: np.ndarray, target: np.ndarray, measured: np.ndarray, vmax: float, dt: float, lead: float) -> np.ndarray:
    """The command for this cycle. Its speed is limited against the *previous command*, not against where the joint measures: limiting
    against the measurement caps how far the command can lead it (vmax*dt, under 1 degree at 250 Hz) and with it the torque the arm can
    be pushed with, so an arm that sags under its weight would take its command down with it. A separate cap keeps the command within
    `lead` of the measurement, so a blocked arm is not pushed ever harder."""
    return np.clip(clip_step(previous, target, vmax, dt), measured - lead, measured + lead)


def child_env() -> dict:
    """The environment the writer runs in. It needs CYCLONEDDS_HOME, as the control loop has it: without it the Python binding loads the
    libddsc that ROS ships (the studio is started with ROS sourced), which does not match it, and the first write aborts the process with
    an assertion in dds_write.c. (Reading, as dds_reader does, is not affected.)"""
    return {**os.environ, "CYCLONEDDS_HOME": str(C.CYCLONEDDS_HOME)}


def emit(**event) -> None:
    print(json.dumps(event), flush=True)


def main(interface: str) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, interface)
    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    held = {"low": None}
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(lambda msg: held.__setitem__("low", msg), 1)
    inbox: queue.Queue = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            try:
                inbox.put(json.loads(line))
            except ValueError:
                continue
        inbox.put({"op": "eof"})
    threading.Thread(target=read_stdin, daemon=True).start()

    while held["low"] is None:
        time.sleep(0.05)
    emit(event="ready")

    crc, msg, weight = CRC(), unitree_hg_msg_dds__LowCmd_(), Weight()
    msg.mode_pr, msg.mode_machine = 0, held["low"].mode_machine
    joints = list(ARM)
    target, tau, sent = np.zeros(14), np.zeros(14), np.zeros(14)
    engaged, releasing, last_heard, announced = False, False, time.monotonic(), False
    settings = dict(C.ARM_SDK_DEFAULTS)
    dt = 1.0 / settings["hz"]
    nxt = time.monotonic()
    while True:
        now = time.monotonic()
        while not inbox.empty():
            op = inbox.get()
            last_heard = now
            if op["op"] == "engage" and not engaged:
                try:
                    settings, joints = validate_settings(op), validate_joints(op.get("joints"))
                except (ValueError, TypeError) as error:
                    emit(event="error", message=f"arm_sdk 설정 오류: {error}")
                    return
                dt = 1.0 / settings["hz"]
                low = held["low"]
                for j in range(29):
                    cmd = msg.motor_cmd[j]
                    cmd.mode, cmd.q, cmd.dq, cmd.tau = 1, low.motor_state[j].q, 0.0, 0.0
                    cmd.kp, cmd.kd = gains(j, settings)
                target = np.array([low.motor_state[j].q for j in joints])
                sent, tau = target.copy(), np.zeros(len(joints))         # the command starts where the joints are
                engaged, announced = True, False
                weight.ramp_to(1.0, C.ARM_SDK_ENGAGE_S, now)
            elif op["op"] == "target" and engaged and not releasing:
                q, t = np.array(op["q"], float), np.array(op["tau"], float)
                if q.shape == t.shape == (len(joints),) and np.isfinite(q).all() and np.isfinite(t).all():
                    target, tau = q, t
            elif op["op"] in ("release", "eof") and engaged and not releasing:
                releasing = True
                weight.ramp_to(0.0, C.ARM_SDK_RELEASE_S, now)
            elif op["op"] == "eof" and not engaged:
                return
        if engaged and not releasing and announced and now - last_heard > C.ARM_SDK_WATCHDOG_S:       # not during the engage ramp: the studio
            # sends its first target only once "engaged" has been announced; if it died meanwhile, stdin closes and that releases
            releasing = True
            weight.ramp_to(0.0, C.ARM_SDK_RELEASE_S, now)
            emit(event="error", message="스튜디오에서 목표가 오지 않아 arm_sdk 제어를 해제합니다")
        if engaged:
            current = np.array([held["low"].motor_state[j].q for j in joints])
            sent = limit_command(sent, target, current, C.ARM_SDK_VLIMIT, dt, C.ARM_SDK_MAX_LEAD_RAD)
            for k, j in enumerate(joints):
                msg.motor_cmd[j].q, msg.motor_cmd[j].dq, msg.motor_cmd[j].tau = float(sent[k]), 0.0, float(tau[k])
            msg.motor_cmd[WEIGHT_JOINT].q = weight.step(now)
            msg.crc = crc.Crc(msg)
            publisher.Write(msg)
            if not announced and weight.done and not releasing:
                announced, last_heard = True, now                       # the watchdog starts here
                emit(event="engaged")
            if releasing and weight.done:
                emit(event="released")
                return
        nxt += dt
        time.sleep(max(0.0, nxt - time.monotonic()))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        pass
