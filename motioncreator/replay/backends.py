"""Where the robot state comes from and where the arm targets go.

`ArmSdkBackend` talks to the robot through two child processes (the SDK is never imported by the server): a read-only
state reader that lives as long as the connection, and an arm_sdk writer that lives from engage to release.
`MockBackend` simulates a robot that follows its commands, so the whole replay can be exercised without hardware.
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import config as C


class CommandError(Exception):
    """A refusal or failure whose message is shown to the user as is."""


@dataclass
class RobotState:
    stamp: float                       # time.monotonic() when the sample was received
    q: np.ndarray                      # 29 joint angles, Unitree order
    dq: np.ndarray
    imu_quat: np.ndarray               # wxyz

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.stamp

    @property
    def tilt_deg(self) -> float:
        w, x, y, z = self.imu_quat
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def list_interfaces(sysfs: str = '/sys/class/net') -> list[dict]:
    """Physical NICs with an IPv4 address (the ones DDS can use), wired first. `robot` marks the robot's subnet."""
    out = subprocess.run(['ip', '-4', '-o', 'addr', 'show'], capture_output=True, text=True).stdout
    nics = []
    for line in out.splitlines():
        cols = line.split()
        if len(cols) > 3 and cols[1] != 'lo' and os.path.exists(f'{sysfs}/{cols[1]}/device'):
            ip = cols[3].split('/')[0]
            nics.append({'name': cols[1], 'ip': ip, 'wifi': os.path.isdir(f'{sysfs}/{cols[1]}/wireless'), 'robot': ip.startswith(C.ROBOT_NET)})
    return sorted(nics, key=lambda n: (n['wifi'], n['name']))


def child_env() -> dict:
    """The writer needs CYCLONEDDS_HOME (and this repository importable): without the matching libddsc the first write aborts the process."""
    return {**os.environ, 'CYCLONEDDS_HOME': str(C.CYCLONEDDS_HOME), 'PYTHONPATH': str(C.ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}


def robot_python() -> str:
    """The Python the SDK children run with: MOTIONCREATOR_ROBOT_PYTHON, else GR00T's venv (it has unitree_sdk2py), else this one."""
    if C.ROBOT_PYTHON:
        return C.ROBOT_PYTHON
    return str(C.SDK_VENV_PYTHON) if C.SDK_VENV_PYTHON.exists() else sys.executable


# ---- real robot --------------------------------------------------------------------------------------------------------------
class DdsStateReader:
    """The read-only child process; keeps the newest state."""
    START_TIMEOUT_S = 8.0

    def __init__(self, interface: str):
        self.interface = interface
        self._state: RobotState | None = None
        self.python = robot_python()
        self._errors: list[str] = []
        self._proc = subprocess.Popen([self.python, '-W', 'ignore', '-m', 'motioncreator.replay.dds_reader', interface], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, bufsize=1, env=child_env(), cwd=str(C.ROOT))
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump_errors, daemon=True).start()

    def _pump_errors(self) -> None:
        for line in self._proc.stderr:
            if line.strip():
                self._errors = [*self._errors[-4:], line.rstrip()]

    def _pump(self) -> None:
        for line in self._proc.stdout:
            try:
                raw = json.loads(line)
                self._state = RobotState(time.monotonic(), np.array(raw['q'], float), np.array(raw['dq'], float), np.array(raw['imu_quat'], float))
            except (ValueError, KeyError):
                continue

    def wait_first(self) -> None:
        deadline = time.monotonic() + self.START_TIMEOUT_S
        while self._state is None:
            if self._proc.poll() is not None or time.monotonic() > deadline:
                died = self._proc.poll() is not None
                self.close()
                time.sleep(0.05)
                cause = f' 오류: {self._errors[-1][-200:]}' if died and self._errors else ''
                raise CommandError(f'로봇 상태를 수신하지 못했습니다 ({"수신 프로세스가 종료됨" if died else f"{self.START_TIMEOUT_S:.0f}초 동안 수신 없음"}, '
                                   f'Python: {self.python}).{cause} 인터페이스 이름, 로봇 전원, 로봇 Python(MOTIONCREATOR_ROBOT_PYTHON)을 확인하세요. 명령은 보내지 않았습니다')
            time.sleep(0.05)

    def state(self) -> RobotState | None:
        return self._state

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class ArmSdkBackend:
    """Arm (and waist) targets over `rt/arm_sdk`, written by a child process. `feedforward(q14, waist)` gives the gravity torque
    (Nm) of the arms sent with each target; the waist gets none."""
    name, label = 'arm_sdk', '기본 정책 (arm_sdk)'
    START_TIMEOUT_S = 10.0

    def __init__(self, interface: str, feedforward, settings):
        self.interface, self.feedforward, self.settings = interface, feedforward, settings     # settings(): the current arm_sdk settings
        self.reader: DdsStateReader | None = None
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._error: str | None = None
        self._retiring: threading.Thread | None = None
        self.joints = C.ARM_SLICE

    # connection (read-only)
    def connect(self) -> None:
        self.reader = DdsStateReader(self.interface)
        self.reader.wait_first()

    def disconnect(self) -> None:
        self.release()
        if self._retiring is not None:
            self._retiring.join(C.ARM_SDK_RELEASE_S + 3.0)
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def state(self) -> RobotState | None:
        return None if self.reader is None else self.reader.state()

    # control
    def tick_hz(self) -> float:
        return float(self.settings()['hz'])

    def check(self) -> str | None:
        if self.reader is None or not self.reader.alive():
            return '로봇과 연결되어 있지 않습니다'
        return '다른 arm_sdk 송신기가 실행 중입니다' if self.alive() else None

    def _write(self, **op) -> None:
        try:
            self._proc.stdin.write(json.dumps(op) + '\n')
            self._proc.stdin.flush()
        except (OSError, ValueError, AttributeError):
            raise CommandError('arm_sdk 송신기와의 연결이 끊겼습니다')

    def _wait(self, event: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                got = self._events.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise CommandError(f'arm_sdk 송신기가 {timeout:.0f}초 안에 응답하지 않았습니다 ({event})')
            if got.get('event') == event:
                return
            if got.get('event') == 'error':
                raise CommandError(got.get('message', 'arm_sdk 송신기 오류'))

    def _pump(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('event') == 'error' and proc is self._proc:
                self._error = event.get('message')
            self._events.put(event)

    def engage(self, q: np.ndarray, joints: slice = C.ARM_SLICE) -> np.ndarray:
        if self._retiring is not None:                      # two writers on rt/arm_sdk at once would fight
            self._retiring.join(C.ARM_SDK_RELEASE_S + 4.0)
            if self._retiring.is_alive():
                raise CommandError('이전 arm_sdk 송신기가 아직 종료되지 않았습니다. 잠시 후 다시 시도하세요')
        self._events, self._error = queue.Queue(), None
        self._proc = subprocess.Popen([robot_python(), '-W', 'ignore', '-m', 'motioncreator.replay.arm_sdk_writer', self.interface],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=child_env(), cwd=str(C.ROOT))
        threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()
        try:
            self._wait('ready', self.START_TIMEOUT_S)
            self._write(op='engage', joints=list(range(joints.start, joints.stop)), **{k: v for k, v in self.settings().items() if k != 'ff'})
            self._wait('engaged', C.ARM_SDK_ENGAGE_S + 3.0)
        except CommandError:
            self.release()
            raise
        self.joints = joints
        return np.asarray(q, float).copy()

    def send(self, q: np.ndarray) -> None:
        waist = q[:C.ARM_SLICE.start - self.joints.start]
        arms = q[C.ARM_SLICE.start - self.joints.start:]
        tau = np.concatenate([np.zeros(len(waist)), self.feedforward(arms, waist if len(waist) else None) * float(self.settings()['ff'])])
        self._write(op='target', q=np.round(q, 5).tolist(), tau=np.round(tau, 3).tolist())

    def release(self) -> None:
        """Ask the writer to fade the weight out and go; do not wait for it (the player's lock is held)."""
        proc, self._proc = self._proc, None
        if proc is None:
            return

        def finish():
            try:
                proc.stdin.write(json.dumps({'op': 'release'}) + '\n')
                proc.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(C.ARM_SDK_RELEASE_S + 3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._retiring = threading.Thread(target=finish, daemon=True)
        self._retiring.start()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and self._error is None

    def lost_reason(self) -> str:
        return self._error or 'arm_sdk 송신기가 종료되어 재생을 끝냈습니다'


# ---- simulated robot ---------------------------------------------------------------------------------------------------------
class MockBackend:
    """A robot that stands still and whose arms (and waist) reach whatever is commanded (with a little lag). Same interface as ArmSdkBackend."""
    name, label = 'mock', '가상 로봇'

    def __init__(self, home_arms: np.ndarray, clock=time.monotonic, threaded: bool = True, lag: float = 0.0):
        self.clock, self.threaded, self.lag = clock, threaded, lag
        self.q = np.zeros(C.N_JOINTS)
        self.q[C.ARM_SLICE] = home_arms
        self.target = np.asarray(home_arms, float).copy()
        self.joints = C.ARM_SLICE
        self.engaged = False
        self.sent: list[np.ndarray] = []
        self._stop = threading.Event()
        self._last = clock()

    def connect(self) -> None:
        if self.threaded:
            threading.Thread(target=self._run, daemon=True).start()

    def disconnect(self) -> None:
        self._stop.set()
        self.engaged = False

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self.step()

    def step(self) -> None:
        now = self.clock()
        dt, self._last = now - self._last, now
        if self.engaged:
            gain = 1.0 if self.lag <= 0 else min(1.0, dt / self.lag)
            self.q[self.joints] += gain * (self.target - self.q[self.joints])

    def state(self) -> RobotState:
        return RobotState(self.clock(), self.q.copy(), np.zeros(C.N_JOINTS), np.array([1.0, 0.0, 0.0, 0.0]))

    def tick_hz(self) -> float:
        return float(C.COMMAND_HZ)

    def check(self) -> str | None:
        return '가상 로봇이 이미 제어 중입니다' if self.engaged else None

    def engage(self, q: np.ndarray, joints: slice = C.ARM_SLICE) -> np.ndarray:
        self.engaged, self.joints = True, joints
        self.target = np.asarray(q, float).copy()
        return self.target.copy()

    def send(self, q: np.ndarray) -> None:
        self.target = np.asarray(q, float).copy()
        self.sent.append(self.target.copy())
        if not self.threaded and self.lag <= 0:
            self.q[self.joints] = self.target

    def release(self) -> None:
        self.engaged = False

    def alive(self) -> bool:
        return self.engaged

    def lost_reason(self) -> str:
        return '가상 로봇 제어가 해제되었습니다'
