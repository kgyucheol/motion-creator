"""Ported from g1-live-studio g1live/dds_reader.py (8411a3d).

Child process: read-only subscriber to the robot's DDS topics on one NIC; prints each state as a JSON line.

CycloneDDS can create one domain per process, so changing the NIC means a new process. `DdsRobot` starts and stops this one.
    python -m motioncreator.replay.dds_reader <interface>
"""
from __future__ import annotations

import json
import sys
import time


PERIOD_S = 0.02            # at most 50 states/s to the studio
BATTERY_FRESH_S = 10


def snapshot(low, stamp: float, bms, bms_stamp: float) -> dict:
    motors, n, imu = low.motor_state, 29, low.imu_state
    battery = None
    if bms is not None and time.monotonic() - bms_stamp < BATTERY_FRESH_S:
        temps = [t for t in bms.temperature if t > 0]
        battery = {"soc": float(bms.soc), "voltage": bms.bmsvoltage[0] / 1000.0, "current": bms.current / 1000.0,
                   "temp": float(max(temps)) if temps else None, "cycle": int(bms.cycle)}
    return {"stamp": stamp, "q": [motors[i].q for i in range(n)], "dq": [motors[i].dq for i in range(n)],
            "tau": [motors[i].tau_est for i in range(n)], "temp": [float(max(motors[i].temperature)) for i in range(n)],
            "volt": [motors[i].vol for i in range(n)], "imu_quat": list(imu.quaternion), "imu_rpy": list(imu.rpy),
            "gyro": list(imu.gyroscope), "acc": list(imu.accelerometer), "imu_temp": float(imu.temperature),
            "mode_machine": int(low.mode_machine), "battery": battery}


def main(interface: str) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, LowState_
    held = {"low": None, "low_at": 0.0, "bms": None, "bms_at": 0.0}

    def on_low(msg):
        held["low"], held["low_at"] = msg, time.monotonic()

    def on_bms(msg):
        held["bms"], held["bms_at"] = msg, time.monotonic()

    ChannelFactoryInitialize(0, interface)
    low = ChannelSubscriber("rt/lowstate", LowState_)
    low.Init(on_low, 1)
    bms = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
    bms.Init(on_bms, 1)
    sent = 0.0
    while True:
        time.sleep(PERIOD_S)
        if held["low"] is not None and held["low_at"] != sent:
            sent = held["low_at"]
            print(json.dumps(snapshot(held["low"], sent, held["bms"], held["bms_at"])), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        pass
