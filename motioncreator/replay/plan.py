"""From a compiled motion to what the arm_sdk path can play: the 14 arm joints over time, plus the checks that decide whether it may start.

Under Unitree's default policy the robot balances and walks by itself; this program only drives the arms. A motion that moves
the legs, the pelvis or the waist roll/pitch therefore cannot be replayed faithfully, and the plan says so instead of silently
dropping that part.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import config as C

LEG_COLUMNS = slice(7, 7 + 12)
WAIST_YAW_COLUMN = 7 + 12
WAIST_RP_COLUMNS = slice(7 + 13, 7 + 15)
ARM_COLUMNS = slice(7 + 15, 7 + 29)


def resample(times: np.ndarray, qpos: np.ndarray, fps: float) -> np.ndarray:
    """The motion on a uniform grid of `fps` frames per second (joint columns only; the root is checked, not played)."""
    times, qpos = np.asarray(times, float), np.asarray(qpos, float)
    grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / fps)
    if len(grid) == len(times) and np.allclose(grid, times, atol=1e-6):
        return qpos
    return np.stack([np.interp(grid, times, qpos[:, c]) for c in range(qpos.shape[1])], axis=1)


def check_trajectory(traj: np.ndarray, fps: float, lower: np.ndarray, upper: np.ndarray, vmax: float = C.REPLAY_VMAX) -> dict:
    """Joint limits and speed of the arm motion. `speed_cap`: the largest playback multiplier that keeps every joint under `vmax`;
    `speed_hard`: the same for REPLAY_VMAX_HARD, which needs a confirmation. Both are also bounded by the frame-rate limit."""
    items, ok = [], True
    if not np.isfinite(traj).all():
        return {'ok': False, 'speed_cap': 0.0, 'speed_hard': 0.0, 'items': [{'level': 'bad', 'name': '데이터', 'text': 'NaN/무한대 값이 있습니다'}]}
    outside = (traj < lower - 1e-4) | (traj > upper + 1e-4)
    if outside.any():
        ok = False
        frames = int(outside.any(axis=1).sum())
        items.append({'level': 'bad', 'name': '관절 한계', 'text': f'{frames}개 프레임이 팔 관절 한계를 벗어납니다 (첫 프레임 {int(np.argmax(outside.any(axis=1)))})'})
    else:
        items.append({'level': 'ok', 'name': '관절 한계', 'text': '모든 프레임이 한계 안입니다'})
    speed = np.abs(np.diff(traj, axis=0)) * fps
    peak = float(speed.max()) if speed.size else 0.0
    frame, joint = np.unravel_index(int(np.argmax(speed)), speed.shape) if speed.size else (0, 0)
    frame_cap = C.REPLAY_FREQ_MAX / fps
    cap = min(vmax / peak if peak > 0 else np.inf, frame_cap)
    hard = min(C.REPLAY_VMAX_HARD / peak if peak > 0 else np.inf, frame_cap)
    where = f'프레임 {frame}, 팔 관절 {joint}'
    if hard < C.REPLAY_SPEED_MIN:
        ok = False
        items.append({'level': 'bad', 'name': '최대 속도', 'text': f'{peak:.1f} rad/s ({where}) — 배율 {hard:.2f} 이하로만 재생할 수 있어 거절합니다'})
    elif cap < 1.0:
        items.append({'level': 'warn', 'name': '최대 속도', 'text': f'{peak:.2f} rad/s ({where}) — 권장 상한 {vmax:.1f}을 넘습니다. 권장 배율 {cap:.2f} 이하, 확인 후 {hard:.2f}까지'})
    else:
        items.append({'level': 'ok', 'name': '최대 속도', 'text': f'{peak:.2f} rad/s ({where})'})
    return {'ok': ok, 'speed_cap': cap if ok else 0.0, 'speed_hard': hard if ok else 0.0, 'items': items}


def check_replayable(qpos: np.ndarray) -> list[dict]:
    """What in the motion the arm_sdk path will not do: legs, pelvis height/tilt, waist roll/pitch and waist yaw, measured against the first frame."""
    items = []
    legs = float(np.abs(qpos[:, LEG_COLUMNS] - qpos[0, LEG_COLUMNS]).max())
    waist_rp = float(np.abs(qpos[:, WAIST_RP_COLUMNS] - qpos[0, WAIST_RP_COLUMNS]).max())
    waist_yaw = float(np.abs(qpos[:, WAIST_YAW_COLUMN] - qpos[0, WAIST_YAW_COLUMN]).max())
    height = float(np.abs(qpos[:, 2] - qpos[0, 2]).max())
    dots = np.clip(np.abs(qpos[:, 3:7] @ qpos[0, 3:7]), 0.0, 1.0)
    tilt = float((2 * np.arccos(dots)).max())
    found = []
    if legs > C.LOWER_BODY_TOLERANCE_RAD:
        found.append(f'다리 관절 최대 {np.degrees(legs):.1f}° 변화')
    if height > C.ROOT_HEIGHT_TOLERANCE_M:
        found.append(f'골반 높이 {height * 1000:.0f} mm 변화')
    if tilt > C.ROOT_TILT_TOLERANCE_RAD:
        found.append(f'골반 기울기 {np.degrees(tilt):.1f}° 변화')
    if waist_rp > C.LOWER_BODY_TOLERANCE_RAD:
        found.append(f'허리 롤/피치 최대 {np.degrees(waist_rp):.1f}° 변화')
    if waist_yaw > C.WAIST_YAW_TOLERANCE_RAD:
        found.append(f'허리 요 최대 {np.degrees(waist_yaw):.1f}° 변화')
    if found:
        items.append({'level': 'warn', 'name': '재생 가능성',
                      'text': '기본 정책 재생은 팔 14축만 보냅니다. 다음 변화는 재생되지 않습니다: ' + ' · '.join(found)})
    else:
        items.append({'level': 'ok', 'name': '재생 가능성', 'text': '팔 외 관절과 골반은 거의 변하지 않습니다'})
    return items


def check_dynamic(reports: list[dict]) -> list[dict]:
    """Dynamic clips: an unreachable or missing grip must not start; an approximate one is allowed but flagged."""
    items = []
    for report in reports:
        label = f'동적 클립 {report["index"] + 1}번'
        status = report['status']
        error = f' (최대 {report["max_position_error_mm"]:.1f} mm)' if report.get('max_position_error_mm') is not None else ''
        if status in ('ok', 'unchanged'):
            items.append({'level': 'ok', 'name': label, 'text': f'파지 구간 도달{error}' if status == 'ok' else '물체가 기준 위치에 있습니다'})
        elif status == 'approx':
            items.append({'level': 'warn', 'name': label, 'text': f'파지 구간이 근사입니다{error}'})
        elif status == 'too_far':
            items.append({'level': 'bad', 'name': label, 'text': f'물체가 기준에서 {report.get("shift_mm", 0):.0f} mm 떨어져 보정 한도를 넘었습니다'})
        elif status == 'missing_object':
            items.append({'level': 'bad', 'name': label, 'text': '묶인 물체가 장면에 없습니다'})
        else:
            items.append({'level': 'bad', 'name': label, 'text': f'파지 구간에 도달할 수 없습니다{error}'})
    return items


def build_plan(robot, motion: dict, fps: float) -> dict:
    """The replay plan of a compiled motion (`compile_motion` result). `traj` is (T, 14) at `fps`; `items` are the preflight lines."""
    qpos = resample(motion['time'], motion['qpos'], fps)
    traj = np.ascontiguousarray(qpos[:, ARM_COLUMNS])
    lower = np.asarray(robot.lower)[3 + 15:3 + 29]
    upper = np.asarray(robot.upper)[3 + 15:3 + 29]
    checks = check_trajectory(traj, fps, lower, upper)
    items = [*checks['items'], *check_replayable(qpos), *check_dynamic(motion.get('dynamic_reports') or [])]
    ok = checks['ok'] and not any(item['level'] == 'bad' for item in items)
    return {'traj': traj, 'fps': float(fps), 'checks': {**checks, 'ok': ok, 'items': items},
            'duration': (len(traj) - 1) / fps, 'keyframe_times': list(motion.get('keyframe_times') or []),
            'lower': lower, 'upper': upper}


class Gravity:
    """Static gravity torque of the 14 arm joints (Nm) from the editor's MuJoCo model: inverse dynamics at rest, legs and waist as in the standing pose."""

    def __init__(self, robot):
        self.model = robot.model
        self.data = mujoco.MjData(robot.model)
        self.home = np.asarray(robot.home, float).copy()
        self.dof = np.asarray(robot.v_indices)[3 + 15:3 + 29]

    def torque(self, q14: np.ndarray) -> np.ndarray:
        q = self.home.copy()
        q[ARM_COLUMNS] = q14
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[self.dof].copy()
