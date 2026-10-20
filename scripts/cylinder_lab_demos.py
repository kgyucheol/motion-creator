"""Write the cylinder lab's demo cases (frontend/lib/cylinder-demos.json) from verified successes.

Usage: .conda/bin/python scripts/cylinder_lab_demos.py <project.json> [target name]
Each demo stores the cut scene, the planner parameters, the planned blade paths and the physics
settings, so the page and the tests replay it without the saved project. Re-run it when the
planner changes; frontend/tests/cylinder-sim.test.mjs checks that every demo still succeeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from motioncreator.cylinder_extraction import plan_scoop, scene_2d  # noqa: E402
from cylinder_scoop_sweep import settled  # noqa: E402

SOFT = {'mmPer10N': 5, 'scope': 'neighbors', 'shape': 'smooth'}
PRY = {'mode': 1, 'side': -1, 'wedge_m': .003, 'insert_depth': .9, 'step_deg': 5}
DEMOS = [
    {'id': 'pry-soft', 'name': '한쪽 날 · 로봇 쪽 지렛대 · 말랑한 용기',
     'description': '왼손 주걱 하나를 로봇 쪽 틈에 쐐기로 넣고 틈 입구를 받침점으로 35° 젖힌 뒤 들어 올립니다. '
                    '매끈한 말랑 원통(10 N에 5 mm). 원통이 크게 찌그러지는 점에 주의하세요.',
     'params': {**PRY, 'pivot_up': 1.1, 'pry_deg': 35}, 'options': {'soft': SOFT}},
    {'id': 'pry-soft-low-friction', 'name': '한쪽 날 · 지렛대 20° · 원통 마찰 0.4',
     'description': '같은 동작을 더 작게(20°) 젖힙니다. 원통 마찰계수를 0.4로 낮춘 경우입니다.',
     'params': {**PRY, 'pivot_up': 1.1, 'pry_deg': 20}, 'options': {'soft': SOFT, 'friction': {'cylinder': .4}}},
    {'id': 'clamp-rigid', 'name': '양쪽 집기 · 강체 (양손, 참고)',
     'description': '왼손은 로봇 쪽 틈, 오른손은 윗벽 쪽 틈에 넣고 1 mm 쐐기로 조여 들어 올립니다. 그리퍼 두 개가 필요한 참고용입니다.',
     'params': {'mode': 2, 'wedge_m': .001, 'insert_depth': .3, 'step_m': .04}, 'options': {}},
]


def main():
    project_file = Path(sys.argv[1])
    target_name = sys.argv[2] if len(sys.argv) > 2 else None
    project = json.loads(project_file.read_text(encoding='utf-8'))
    target_id = None
    if target_name:
        target_id = next(item['id'] for item in project['scene_objects'] if item.get('name') == target_name)
    base = scene_2d(project, target_id)
    rest = settled(base)
    demos = []
    for demo in DEMOS:
        path = plan_scoop(rest, demo['params'])
        demos.append({**demo, 'source': f"{project_file.parent.name} · {base['target']['name']}",
                      'base_scene': base, 'blades': path['blades'], 'stage_steps': path['stage_steps']})
    output = ROOT / 'frontend/lib/cylinder-demos.json'
    output.write_text(json.dumps(demos, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {len(demos)} demos to {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
