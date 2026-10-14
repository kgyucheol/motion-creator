"""What the server holds for the robot replay: the connection (real or simulated), the prepared plan and the player."""
from __future__ import annotations

import threading
import time

import numpy as np

from . import config as C
from .backends import ArmSdkBackend, CommandError, MockBackend, list_interfaces
from .plan import ARM_COLUMNS, Gravity, build_plan
from .player import ReplayPlayer, check_approach


class ReplaySession:
    def __init__(self, get_robot, read_only: bool = False):
        self.get_robot, self.read_only = get_robot, read_only
        self.backend = None
        self.mode: str | None = None                 # 'real' | 'mock'
        self.interface: str | None = None
        self.settings = dict(C.ARM_SDK_DEFAULTS)
        self._gravity: tuple[object, Gravity] | None = None
        self._lock = threading.RLock()
        self.plan: dict | None = None
        self.player = ReplayPlayer(lambda: self.backend, self._state)

    # ---- state / gravity -------------------------------------------------------------------------------------
    def _state(self):
        return None if self.backend is None else self.backend.state()

    def _feedforward(self, q14: np.ndarray) -> np.ndarray:
        robot = self.get_robot()
        if self._gravity is None or self._gravity[0] is not robot:
            self._gravity = (robot, Gravity(robot))
        return self._gravity[1].torque(q14)

    # ---- connection ------------------------------------------------------------------------------------------
    def connect(self, interface: str | None, mock: bool) -> None:
        with self._lock:
            if self.player.engaged:
                raise CommandError('재생 중에는 연결을 바꿀 수 없습니다. 먼저 복귀·해제하세요')
            self.disconnect()
            if mock:
                backend = MockBackend(np.asarray(self.get_robot().home, float)[ARM_COLUMNS])
                backend.connect()
                self.backend, self.mode, self.interface = backend, 'mock', None
                return
            if self.read_only:
                raise CommandError('읽기 전용 모드에서는 실제 로봇에 연결하지 않습니다 (가상 로봇만 사용할 수 있습니다)')
            if not interface:
                raise CommandError('네트워크 인터페이스를 고르세요')
            backend = ArmSdkBackend(interface, self._feedforward, lambda: self.settings)
            backend.connect()                       # read-only: no command is sent
            self.backend, self.mode, self.interface = backend, 'real', interface

    def disconnect(self) -> None:
        with self._lock:
            if self.player.engaged:
                raise CommandError('재생 중에는 연결을 끊을 수 없습니다. 먼저 복귀·해제하세요')
            if self.backend is not None:
                self.backend.disconnect()
            self.backend, self.mode, self.interface = None, None, None

    def configure(self, **values) -> dict:
        from .arm_sdk_writer import validate_settings
        with self._lock:
            if self.player.engaged:
                raise CommandError('재생 중에는 설정을 바꿀 수 없습니다')
            try:
                self.settings = validate_settings({**self.settings, **values})
            except (ValueError, TypeError) as error:
                raise CommandError(str(error)) from error
            return dict(self.settings)

    # ---- plan ------------------------------------------------------------------------------------------------
    def prepare(self, project: dict, fps: int = 30) -> dict:
        """Compile the project (with the scene objects where they are now) and check it. Does not touch the robot."""
        from ..motion import compile_motion
        robot = self.get_robot()
        motion = compile_motion(robot, project, fps)
        plan = build_plan(robot, motion, fps)
        with self._lock:
            self.player.load(plan['traj'], plan['fps'], plan['checks'], plan['keyframe_times'])
            self.plan = plan
        return self.summary()

    def summary(self) -> dict | None:
        if self.plan is None:
            return None
        plan = self.plan
        extra = []
        state = self._state()
        if state is not None:
            approach = check_approach(plan['traj'][0], state.q[C.ARM_SLICE])
            extra.append({k: approach[k] for k in ('level', 'name', 'text')})
        return {'duration': plan['duration'], 'frames': len(plan['traj']), 'fps': plan['fps'],
                'items': [*plan['checks']['items'], *extra], 'ok': plan['checks']['ok'],
                'speed_cap': plan['checks']['speed_cap'], 'speed_hard': plan['checks']['speed_hard'],
                'warnings': sum(item['level'] == 'warn' for item in [*plan['checks']['items'], *extra])}

    # ---- run -------------------------------------------------------------------------------------------------
    def start(self, speed: float, auto_return: bool, allow_fast: bool, confirm: bool) -> dict:
        if not confirm:
            raise CommandError('실행 확인이 필요합니다')
        if self.mode == 'real' and self.read_only:
            raise CommandError('읽기 전용 모드에서는 실제 로봇을 움직이지 않습니다')
        if self.backend is None:
            raise CommandError('로봇(또는 가상 로봇)에 먼저 연결하세요')
        return self.player.start(speed, auto_return, allow_fast)

    def pause(self, on: bool) -> None:
        self.player.pause(on)

    def stop(self) -> None:
        self.player.stop()

    def home(self) -> None:
        self.player.go_home()

    def set_speed(self, speed: float) -> None:
        self.player.set_speed(speed)

    def shutdown(self) -> None:
        self.player.shutdown()
        if self.backend is not None:
            self.backend.disconnect()

    def status(self) -> dict:
        state = self._state()
        now = time.monotonic()
        return {'mode': self.mode, 'interface': self.interface, 'connected': self.backend is not None,
                'read_only': self.read_only, 'settings': dict(self.settings),
                'robot': None if state is None else {'age': round(state.age(now), 2), 'tilt_deg': round(state.tilt_deg, 1),
                                                     'arms': np.round(state.q[C.ARM_SLICE], 3).tolist()},
                'player': self.player.status(), 'plan': self.summary()}

    @staticmethod
    def interfaces() -> list[dict]:
        try:
            return list_interfaces()
        except (OSError, FileNotFoundError):
            return []
