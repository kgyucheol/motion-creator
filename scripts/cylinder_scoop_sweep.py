"""Sweep scoop parameters in the 2-D physics check and list the variants that lift the cylinder out.

Usage: .conda/bin/python scripts/cylinder_scoop_sweep.py <project.json> [--target ID ...] [--remove ID ...]
       [--quick | --wrap | --clamp] [--soft MM [--soft-all]]
       default grid: single-tool pry; --wrap: single-tool wrap; --clamp: two-tool clamp; --quick: small pry grid
Paths come from motioncreator.cylinder_extraction (the same code the editor uses); the physics runs
in frontend/lib/cylinder-sim.ts under Node, several processes in parallel.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motioncreator.cylinder_extraction import plan_scoop, scene_2d, with_positions  # noqa: E402

GRID = {
    'mode': [1.],
    'side': [-1., 1.],
    'step_deg': [20., 5.],
    'wedge_m': [0., .003, .006],
    'insert_depth': [.3, .6, .9],
    'pivot_up': [.3, .7, 1.1],
    'pry_deg': [20., 35., 50.],
}
WRAP = {'mode': [0.], 'side': [-1., 1.], 'step_deg': [10.], 'wedge_m': [0., .004, .008, .012], 'wrap_end_deg': [240., 270.],
        'slide_extent': [0., .35], 'tilt_deg': [-20., 0., 20.]}
CLAMP = {'mode': [2.], 'side': [-1.], 'step_deg': [10.], 'step_m': [.3, .15, .08, .04, .02, .01], 'wedge_m': [0., .001, .002, .003, .004, .008],
         'insert_depth': [0., .3, .6, .9]}
QUICK = {'mode': [1.], 'side': [-1., 1.], 'wedge_m': [.003], 'insert_depth': [.9], 'pivot_up': [.7, 1.1], 'pry_deg': [20., 35.], 'step_deg': [5.]}


def run_chunk(scene, chunk, options):
    with tempfile.TemporaryDirectory() as folder:
        source, output = Path(folder) / 'in.json', Path(folder) / 'out.json'
        source.write_text(json.dumps({'scene': scene, 'variants': chunk, 'options': options}))
        subprocess.run(['node', '--no-warnings', '--experimental-transform-types', 'scripts/cylinder-sweep.ts',
                        str(source), str(output)], cwd=ROOT / 'frontend', check=True)
        return json.loads(output.read_text())


def settled(scene):
    """Plan from resting positions: let Node settle the scene, then move the circles there."""
    with tempfile.TemporaryDirectory() as folder:
        source, output = Path(folder) / 'scene.json', Path(folder) / 'settled.json'
        source.write_text(json.dumps(scene))
        subprocess.run(['node', '--no-warnings', '--experimental-transform-types', 'scripts/cylinder-settle.ts',
                        str(source), str(output)], cwd=ROOT / 'frontend', check=True)
        rest = json.loads(output.read_text())  # settled as rigid bodies: a soft pile rests within a few mm of this
    return with_positions(scene, {circle['id']: circle['center'] for circle in rest['circles']})


def sweep(project, targets, grid, options=None, workers=None):
    results = []
    for target in targets:
        scene = settled(scene_2d(project, target))
        variants = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            path = plan_scoop(scene, params)
            variants.append({'name': scene['target']['name'], 'params': params, 'blades': path['blades'],
                             'stage_steps': path['stage_steps']})
        count = workers or max(1, (os.cpu_count() or 2) - 1)
        chunks = [variants[index::count] for index in range(count) if variants[index::count]]
        with ThreadPoolExecutor(len(chunks)) as pool:
            for chunk, chunk_results in zip(chunks, pool.map(lambda c: run_chunk(scene, c, options or {}), chunks)):
                for variant, result in zip(chunk, chunk_results):
                    results.append({**result, 'target': scene['target']['name'], 'target_id': target or scene['target_id'],
                                    'stage_steps': variant['stage_steps']})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('--target', action='append', help='cylinder object id (repeatable); default: the top one')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--clamp', action='store_true', help='sweep the two-tool clamp mode')
    parser.add_argument('--wrap', action='store_true', help='sweep the single-tool wrap (surface following) mode')
    parser.add_argument('--out', help='write all results as JSON')
    parser.add_argument('--remove', action='append', default=[], help='object id already taken out (repeatable)')
    parser.add_argument('--soft', type=float, help='soft cylinders: squash in mm under 10 N between plates')
    parser.add_argument('--soft-all', action='store_true', help='make every cylinder soft, not only the target and its neighbours')
    args = parser.parse_args()
    project = json.loads(Path(args.project).read_text(encoding='utf-8'))
    project['scene_objects'] = [item for item in project.get('scene_objects', []) if item['id'] not in args.remove]
    options = {'soft': {'mmPer10N': args.soft, 'scope': 'all' if args.soft_all else 'neighbors', 'nodes': 16}} if args.soft else {}
    grid = CLAMP if args.clamp else WRAP if args.wrap else QUICK if args.quick else GRID
    results = sweep(project, args.target or [None], grid, options)
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    wins = [r for r in results if r['success']]
    print(f'{len(wins)} / {len(results)} succeeded')
    for r in sorted(wins, key=lambda r: (r['target'], r['steps']))[:40]:
        print(f"OK  {r['target']:22s} " + ' '.join(f'{k}={v:g}' for k, v in r['params'].items()) + f" -> {r['steps']} steps "
              f"{r['seconds']:.1f}s lift={r['liftM'] * 1000:.0f}mm force={r['peakForceN']:.0f}N neighbours={r['neighborMaxM'] * 1000:.0f}mm "
              f"stalled={r['stalledM'] * 1000:.0f}mm")


if __name__ == '__main__':
    main()
