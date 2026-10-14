"""LeRobot/Unitree episode (data.json) to G1 qpos frames."""
import json
import math
import numpy as np

ARM_START = {'left_arm': 15, 'right_arm': 22}
WAIST_START = 12
SOURCES = ('actions', 'states')


def lerobot_episode_to_qpos(robot, content: bytes, source: str = 'actions'):
    """Return (qpos[T, nq], fps). Joints the episode does not record keep the stand pose."""
    if source not in SOURCES:
        raise ValueError('source must be actions or states')
    try:
        episode = json.loads(content.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('Could not read LeRobot JSON episode') from exc
    frames = episode.get('data') if isinstance(episode, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError('LeRobot episode has no frames')
    fps = float((episode.get('info') or {}).get('image', {}).get('fps', 30))
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError('LeRobot episode FPS must be 1–120')

    qpos = np.tile(robot.home, (len(frames), 1))
    for index, frame in enumerate(frames):
        groups = frame.get(source) or {}
        for name, start in ARM_START.items():
            values = (groups.get(name) or {}).get('qpos') or []
            if len(values) != 7:
                raise ValueError(f'Frame {index}: {name} needs 7 joint values')
            qpos[index, 7 + start:7 + start + 7] = values
        body = (groups.get('body') or {}).get('qpos') or []
        if len(body) == 3:
            qpos[index, 7 + WAIST_START:7 + WAIST_START + 3] = body
        elif len(body) == 29:
            qpos[index, 7:] = body
    if not np.isfinite(qpos).all():
        raise ValueError('LeRobot episode contains non-finite joint values')
    return qpos, fps


def is_lerobot_episode(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get('data'), list) and 'info' in value


def qpos_to_lerobot_episode(qpos, fps: float, goal: str = '') -> dict:
    """The inverse of lerobot_episode_to_qpos for the arms and waist: an episode robot-studio (or unitree_lerobot) can replay.

    `states` repeats `actions`; robot-studio replays the arms from `actions` and takes waist moves from its own options.
    """
    qpos = np.asarray(qpos, dtype=float)
    frames = []
    for row in qpos:
        group = {name: {'qpos': row[7 + start:7 + start + 7].tolist(), 'qvel': [], 'torque': []} for name, start in ARM_START.items()}
        group['body'] = {'qpos': row[7 + WAIST_START:7 + WAIST_START + 3].tolist(), 'qvel': [], 'torque': []}
        frames.append({'actions': group, 'states': json.loads(json.dumps(group))})
    return {'info': {'image': {'fps': float(fps)}}, 'text': {'goal': goal}, 'data': frames}
