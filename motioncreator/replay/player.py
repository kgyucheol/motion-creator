"""The replay player: approach the first frame, play, pause, stop, go home, release.

Ported from g1-live-studio g1live/replay.py (ReplayPlayer, 8411a3d) with the safety procedure of robot-studio's runner:
the arms move to the first frame gently, the recording plays at a speed the checks allow, a stop keeps commanding the last
target (so the arm_sdk weight is never dropped by accident), going home is measured against what the robot actually did, and
the control is handed back with a weight fade. `tick(now)` is called every 1/hz s by a thread (or by hand in tests).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import config as C
from .backends import CommandError


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def move_duration(delta: np.ndarray, vmax: float) -> float:
    """Shortest min-jerk duration that keeps every joint under `vmax` (peak speed of the profile is 1.875 x mean)."""
    return max(C.MIN_MOVE_S, 1.875 * float(np.max(np.abs(delta))) / vmax)


def check_approach(first: np.ndarray, current: np.ndarray, vmax: float = C.REPLAY_APPROACH_VMAX) -> dict:
    """How far the arms are from the first frame; the player moves them there first."""
    delta = first - current
    far = float(np.max(np.abs(delta)))
    seconds = 0.0 if far < 1e-3 else move_duration(delta, vmax)
    level = 'ok' if far < C.APPROACH_OK_RAD else 'warn'
    return {'level': level, 'name': '시작 자세', 'seconds': seconds,
            'text': f'첫 프레임과 최대 {np.degrees(far):.0f}° 차이 — 먼저 {seconds:.1f}초에 걸쳐 이동합니다' if far >= 1e-3 else '첫 프레임과 같은 자세입니다'}


class ReplayPlayer:
    """engaging -> approach -> playing <-> paused -> finished, then returning to the pose the arms started from and releasing.
    `stopped` holds the last target. Going home is a separate request.
    A single pose (the editor's, or a card's) is sent with `move_to`: engaging -> moving -> holding, which keeps commanding it
    until the next pose, a replay or going home."""
    ACTIVE = ('engaging', 'approach', 'playing', 'paused', 'stopped', 'finished', 'returning', 'moving', 'holding')
    SETTLED = ('stopped', 'finished', 'holding')            # still engaged, commanding a fixed target

    def __init__(self, get_backend, get_state, clock=time.monotonic, threaded: bool = True):
        self.get_backend, self.get_state, self.clock, self.threaded = get_backend, get_state, clock, threaded
        self._lock = threading.RLock()
        self.traj: np.ndarray | None = None
        self.fps, self.checks, self.keyframe_times = 30.0, None, []
        self.state, self.fault, self.backend = 'idle', None, None
        self.pos, self.speed, self.auto_return, self.start_idx = 0.0, 1.0, True, 0
        self.hz, self.allow_fast = float(C.COMMAND_HZ), False
        self.cmd, self.origin = np.zeros(14), np.zeros(14)
        self.joints = C.ARM_SLICE                      # the Unitree motors the plan commands: the arms, or the waist and the arms
        self._phase: tuple[np.ndarray, np.ndarray, float, float] | None = None       # q0, q1, t0, duration of the current min-jerk move
        self._last = 0.0
        self._lag_since: float | None = None
        self._cancel_engage = False
        self._gen = 0                                   # one tick thread per start; an older one sees the number move on and ends
        self._goal: np.ndarray | None = None            # the pose `move_to` was asked for, until the arms are engaged

    @property
    def engaged(self) -> bool:
        return self.state in self.ACTIVE

    def load(self, traj: np.ndarray, fps: float, checks: dict, keyframe_times=(), joints: slice = C.ARM_SLICE) -> None:
        with self._lock:
            if self.engaged and self.state not in self.SETTLED:
                raise CommandError('재생 중에는 모션을 바꿀 수 없습니다. 먼저 정지하거나 복귀·해제하세요')
            traj = np.asarray(traj, float)
            if traj.shape[1] != joints.stop - joints.start:
                raise ValueError('궤적의 관절 수가 제어할 관절과 다릅니다')
            if self.engaged and joints != self.joints:          # holding a pose: the next replay continues from it
                raise CommandError('제어 중인 관절(허리 포함 여부)과 다릅니다. 복귀·해제한 뒤 다시 점검하세요')
            self.traj, self.fps, self.pos, self.fault, self.joints = traj, float(fps), 0.0, None, joints
            self.checks, self.keyframe_times = checks, list(keyframe_times)

    def _at(self, pos: float) -> np.ndarray:
        i = min(int(pos), len(self.traj) - 2)
        return self.traj[i] + (pos - i) * (self.traj[i + 1] - self.traj[i])

    # ---- requests --------------------------------------------------------------------------------------------
    def start(self, speed: float, auto_return: bool, allow_fast: bool = False) -> dict:
        with self._lock:
            if self.traj is None or self.checks is None:
                raise CommandError('재생할 모션을 먼저 준비하세요')
            if self.state not in ('idle', *self.SETTLED):
                raise CommandError('이미 재생 중입니다')
            if self.state != 'idle' and self.cmd.shape != self.traj.shape[1:]:
                raise CommandError('제어 중인 관절(허리 포함 여부)이 이 모션과 다릅니다. 복귀·해제한 뒤 다시 시작하세요')
            if not np.isfinite(speed):
                raise CommandError('잘못된 재생 속도입니다')
            if not self.checks['ok']:
                raise CommandError('사전 점검을 통과하지 못했습니다: ' + next(i['text'] for i in self.checks['items'] if i['level'] == 'bad'))
            st = self.get_state()
            now = self.clock()
            if st is None or st.age(now) > C.STATE_STALE_S:
                raise CommandError('로봇 상태가 수신되지 않습니다')
            if st.tilt_deg > C.TILT_FAULT_DEG:
                raise CommandError(f'몸통 기울기 {st.tilt_deg:.0f}° — 로봇이 안정적이지 않습니다')
            resuming = self.state in self.SETTLED
            if not resuming:
                if np.max(np.abs(st.dq[self.joints])) > C.SETTLED_SPEED:
                    raise CommandError('팔이 아직 움직이는 중입니다. 멈춘 뒤 다시 시도하세요')
                backend = self.get_backend()
                if (reason := backend.check()) is not None:
                    raise CommandError(reason)
                self.backend, self.hz = backend, float(backend.tick_hz())
            self.start_idx = int(self.pos) if self.pos < len(self.traj) - 2 else 0
            self.allow_fast = bool(allow_fast)
            limit = self._speed_limit()
            self.speed = float(np.clip(speed, min(C.REPLAY_SPEED_MIN, limit), limit))
            self.auto_return, self.fault, self._lag_since, self._cancel_engage = bool(auto_return), None, None, False
            current = self.cmd if resuming else st.q[self.joints]
            approach = check_approach(self.traj[self.start_idx], current)
            if resuming:
                self._begin_move(self.cmd, self.traj[self.start_idx], now, 'approach')
            else:
                self.state, self._gen, self._goal = 'engaging', self._gen + 1, None
                gen = self._gen
                self._spawn(lambda: self._engage_and_run(gen))
            return {'speed': self.speed, 'frequency': self.speed * self.fps, 'capped': self.speed < speed - 1e-9, 'approach_s': approach['seconds'],
                    'fast': self.speed > self.checks['speed_cap'] + 1e-9}

    def move_to(self, q: np.ndarray, joints: slice, lower: np.ndarray, upper: np.ndarray) -> dict:
        """Move the commanded joints to one pose (min-jerk at REPLAY_APPROACH_VMAX) and hold it there."""
        with self._lock:
            q = np.asarray(q, float)
            if q.shape != (joints.stop - joints.start,) or not np.isfinite(q).all():
                raise CommandError('보낼 자세의 관절 수가 맞지 않습니다')
            if np.any(q < lower - 1e-4) or np.any(q > upper + 1e-4):
                raise CommandError('보낼 자세가 관절 한계를 벗어납니다')
            st, now = self.get_state(), self.clock()
            if st is None or st.age(now) > C.STATE_STALE_S:
                raise CommandError('로봇 상태가 수신되지 않습니다')
            if st.tilt_deg > C.TILT_FAULT_DEG:
                raise CommandError(f'몸통 기울기 {st.tilt_deg:.0f}° — 로봇이 안정적이지 않습니다')
            if self.state in self.SETTLED:
                if self.joints != joints:
                    raise CommandError('제어 중인 관절(허리 포함 여부)과 다릅니다. 복귀·해제한 뒤 다시 보내세요')
                self.fault, self._lag_since = None, None
                self._begin_move(self.cmd, q, now, 'moving')
                return {'seconds': self._phase[3]}
            if self.state != 'idle':
                raise CommandError('재생이나 이동이 진행 중입니다. 정지한 뒤 보내세요')
            if np.max(np.abs(st.dq[joints])) > C.SETTLED_SPEED:
                raise CommandError('팔이 아직 움직이는 중입니다. 멈춘 뒤 다시 시도하세요')
            backend = self.get_backend()
            if (reason := backend.check()) is not None:
                raise CommandError(reason)
            self.backend, self.hz, self.joints, self.fault, self._lag_since = backend, float(backend.tick_hz()), joints, None, None
            self._goal, self._cancel_engage = q, False
            self.state, self._gen = 'engaging', self._gen + 1
            gen = self._gen
            self._spawn(lambda: self._engage_and_run(gen))
            return {'seconds': move_duration(q - st.q[joints], C.REPLAY_APPROACH_VMAX)}

    def pause(self, on: bool) -> None:
        with self._lock:
            if on:
                if self.state != 'playing':
                    raise CommandError('재생 중이 아닙니다')
                self.state = 'paused'
            else:
                if self.state != 'paused':
                    raise CommandError('일시정지 상태가 아닙니다')
                self.state, self._last = 'playing', self.clock()

    def set_speed(self, speed: float) -> None:
        with self._lock:
            if self.checks is None or not np.isfinite(speed):
                return
            limit = self._speed_limit()
            self.speed = float(np.clip(speed, min(C.REPLAY_SPEED_MIN, limit), limit))

    def _speed_limit(self) -> float:
        return self.checks['speed_hard' if self.allow_fast else 'speed_cap']

    def stop(self) -> None:
        """Hold the last target where it is (the arms keep being commanded). Going home and releasing is a separate request."""
        with self._lock:
            if self.state == 'engaging':
                self._cancel_engage = True
            elif self.state in ('approach', 'playing', 'paused', 'returning', 'moving'):
                self.state, self._phase = 'stopped', None

    def go_home(self) -> None:
        """Move back to the pose the arms had when the replay was started, then hand the control back."""
        with self._lock:
            if self.state not in ('stopped', 'finished', 'paused', 'holding'):
                raise CommandError('복귀할 수 있는 상태가 아닙니다 (먼저 정지하세요)' if self.engaged else '재생 중이 아닙니다')
            self._begin_move(self.cmd, self.origin, self.clock(), 'returning', C.RETURN_VMAX)

    def shutdown(self) -> None:
        with self._lock:
            if self.state == 'engaging':
                self._cancel_engage = True
            elif self.engaged and self.backend is not None:
                self.backend.release()
            self.state = 'idle'

    # ---- internals -------------------------------------------------------------------------------------------
    def _spawn(self, fn) -> None:
        if self.threaded:
            threading.Thread(target=fn, daemon=True).start()
        else:
            fn()

    def _begin_move(self, a, b, now: float, state: str, vmax: float = C.REPLAY_APPROACH_VMAX) -> None:
        a, b = np.asarray(a, float).copy(), np.asarray(b, float).copy()
        near = float(np.max(np.abs(b - a))) < 1e-3                       # already there: no minimum-length move
        self._phase = (a, b, now, 0.02 if near else move_duration(b - a, vmax))
        self.state = state

    def _engage_and_run(self, gen: int) -> None:
        try:
            st = self.get_state()
            if st is None:
                raise CommandError('로봇 상태가 수신되지 않습니다')
            q0 = self.backend.engage(st.q[self.joints].copy(), self.joints)
        except CommandError as error:
            with self._lock:
                self.state, self.fault = 'idle', str(error)
            return
        with self._lock:
            if self._cancel_engage:
                self.backend.release()
                self.state = 'idle'
                return
            self.cmd, self.origin = q0.copy(), q0.copy()
            if self._goal is not None:
                self._begin_move(q0, self._goal, self.clock(), 'moving')
                self._goal = None
            else:
                self._begin_move(q0, self.traj[self.start_idx], self.clock(), 'approach')
        if not self.threaded:
            return
        period, nxt = 1.0 / self.hz, time.monotonic()
        while True:
            with self._lock:
                if self.state == 'idle' or self._gen != gen:
                    return
            try:
                self.tick()
            except Exception as error:               # never leave a half-broken stream running
                with self._lock:
                    self._release_with(f'내부 오류로 재생을 해제했습니다: {error}')
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    def _release_with(self, reason: str | None) -> None:
        if self.backend is not None:
            self.backend.release()
        self.state, self._phase = 'idle', None
        if reason:
            self.fault = reason

    def _hold(self, reason: str) -> None:
        self.state, self._phase, self.fault = 'stopped', None, reason

    def tick(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        with self._lock:
            if self.state in ('idle', 'engaging'):
                return
            st = self.get_state()
            if st is None or st.age(now) > C.STATE_STALE_S:
                return self._release_with('로봇 상태 수신이 끊겨 재생을 해제했습니다')
            if st.tilt_deg > C.TILT_FAULT_DEG:
                return self._release_with(f'몸통이 {st.tilt_deg:.0f}° 기울어 재생을 해제했습니다')
            if not self.backend.alive():
                return self._release_with(self.backend.lost_reason())
            self.cmd = self._target(now)
            if self.state == 'idle':                                                 # the return just finished
                return
            try:
                self.backend.send(self.cmd)
            except CommandError as error:
                return self._release_with(str(error))
            self._check_lag(st, now)

    def _target(self, now: float) -> np.ndarray:
        if self.state in ('approach', 'returning', 'moving'):
            a, b, t0, duration = self._phase
            tau = (now - t0) / duration
            q = a + (b - a) * min_jerk(tau)
            if tau >= 1.0:
                if self.state == 'approach':
                    self.state, self.pos, self._last, self._phase = 'playing', float(self.start_idx), now, None
                elif self.state == 'moving':
                    self.state, self._phase = 'holding', None
                else:
                    self._release_with(None)
            return q
        if self.state == 'playing':
            self.pos += (now - self._last) * self.fps * self.speed
            self._last = now
            if self.pos >= len(self.traj) - 1:
                self.pos, self.state = float(len(self.traj) - 1), 'finished'
            return self._at(self.pos)
        if self.state == 'finished' and self.auto_return:
            self._begin_move(self.cmd, self.origin, now, 'returning', C.RETURN_VMAX)
        return self.cmd

    def _check_lag(self, st, now: float) -> None:
        if self.state not in ('approach', 'playing', 'returning', 'moving'):
            self._lag_since = None
            return
        lag = float(np.max(np.abs(self.cmd - st.q[self.joints])))
        if lag > C.TRACKING_FAULT_RAD:
            self._lag_since = self._lag_since or now
            if now - self._lag_since > C.TRACKING_FAULT_S:
                self._hold(f'관절이 명령을 따라가지 못해({lag:.2f} rad) 재생을 정지했습니다')
        else:
            self._lag_since = None

    def status(self) -> dict:
        with self._lock:
            n = 0 if self.traj is None else len(self.traj)
            progress = None
            if self._phase is not None and self.state in ('approach', 'returning', 'moving'):
                progress = min(1.0, (self.clock() - self._phase[2]) / self._phase[3])
            checks = self.checks
            return {'state': self.state, 'frame': round(self.pos, 2), 'frames': n, 'fps': self.fps, 'speed': round(self.speed, 3),
                    'speed_cap': None if checks is None else round(checks['speed_cap'], 3),
                    'speed_hard': None if checks is None else round(checks['speed_hard'], 3), 'auto_return': self.auto_return,
                    'time': round(self.pos / self.fps, 2), 'duration': round(max(n - 1, 0) / self.fps, 2), 'phase': progress,
                    'keyframe_times': self.keyframe_times, 'fault': self.fault, 'fast': self.allow_fast,
                    'backend': None if self.backend is None or not self.engaged else self.backend.name,
                    'checks': None if checks is None else checks['items'], 'ok': bool(checks and checks['ok'])}
