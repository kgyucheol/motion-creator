import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from motioncreator.cylinder_extraction import DEFAULT_PARAMS, plan_scoop, scene_2d, with_positions

ROOT = Path(__file__).resolve().parents[1]
TILT = [0., -.4226, 0., .9063]  # 50 deg about Y, like the ramen box scene


def ramen_box_project():
    """The tilted carton with 4 rows x 3 layers of lying 13.5 cm bundles used for ramen extraction."""
    rotation = np.array([[.6428, 0, -.766], [0, 1, 0], [.766, 0, .6428]])  # columns: box x, y, z (up)
    center = np.array([.506, 0., .727])
    objects = [{'id': 'carton', 'name': 'Carton', 'shape': 'open_box', 'position': center.tolist(), 'quaternion_xyzw': TILT,
                'size': [.59, .76, .42], 'wall_thickness_m': .02, 'mass_kg': 2., 'friction': .7, 'fixed': True, 'visible': True}]
    for layer in range(3):
        for row in range(4):
            local = np.array([-.207 + row * .138, 0., -.122 + layer * .1325])
            objects.append({'id': f'bundle-L{layer + 1}-R{row + 1}', 'name': f'Bundle_L{layer + 1:02d}_R{row + 1:02d}', 'shape': 'box',
                            'collision_shape': 'convex_hull', 'position': (center + rotation @ local).tolist(),
                            'quaternion_xyzw': TILT, 'size': [.135, .7, .135], 'mass_kg': 1., 'friction': .7,
                            'fixed': False, 'visible': True})
    return {'scene_objects': objects}


def test_scene_cuts_the_box_and_bundles_in_the_cross_section():
    scene = scene_2d(ramen_box_project())
    assert len(scene['circles']) == 12 and len(scene['walls']) == 3
    assert scene['target_id'] == 'bundle-L3-R4'  # the highest one by default
    assert scene['target']['radius'] == pytest.approx(.0675)
    up, far = np.array(scene['frame']['up']), np.array(scene['frame']['far'])
    np.testing.assert_allclose(up, [-.766, .643], atol=1e-3)  # the box opening
    assert far @ np.array(scene['toward_robot']) < 0
    assert scene['frame']['rim_up_m'] == pytest.approx(.0675, abs=.003)
    with pytest.raises(ValueError):
        scene_2d(ramen_box_project(), 'carton')


def test_with_positions_moves_the_target_frame():
    scene = scene_2d(ramen_box_project())
    target = scene['target_id']
    center = np.array(scene['target']['center'])
    moved = with_positions(scene, {target: (center + np.array(scene['frame']['up']) * .01).tolist()})
    assert moved['frame']['rim_up_m'] == pytest.approx(scene['frame']['rim_up_m'] - .01)
    np.testing.assert_allclose(moved['frame']['origin'], moved['target']['center'])


def test_clamp_path_puts_both_tools_on_opposite_sides_in_lock_step():
    scene = scene_2d(ramen_box_project())
    path = plan_scoop(scene, {'mode': 2})
    left, right = path['blades']
    assert (left['side'], right['side'], left['flip'], right['flip']) == ('robot', 'far', False, True)
    assert [w['duration'] for w in left['waypoints']] == [w['duration'] for w in right['waypoints']]
    assert [w['stage'] for w in left['waypoints']] == [w['stage'] for w in right['waypoints']]
    center, far, up = (np.array(scene['frame'][key]) for key in ('origin', 'far', 'up'))
    squeeze = next(i for i, w in enumerate(left['waypoints']) if w['stage'] == 'squeeze')
    assert (np.array(left['waypoints'][squeeze]['tcp']) - center) @ far < 0 < (np.array(right['waypoints'][squeeze]['tcp']) - center) @ far
    lifted = next(w for w in reversed(left['waypoints']) if w['stage'] == 'lift')
    assert (np.array(lifted['tcp']) - center) @ up > scene['frame']['rim_up_m']


@pytest.mark.parametrize('mode', [0, 1])
def test_single_tool_paths_follow_the_target_and_get_finer_with_smaller_steps(mode):
    scene = scene_2d(ramen_box_project())
    coarse = plan_scoop(scene, {'mode': mode, 'step_m': .05, 'step_deg': 30})
    fine = plan_scoop(scene, {'mode': mode, 'step_m': .01, 'step_deg': 5})
    assert len(fine['waypoints']) > len(coarse['waypoints'])
    assert all(math.isfinite(w['tcp'][0]) and math.isfinite(w['angle']) for w in fine['waypoints'])
    if mode == 0:  # the wrap samples the circle: every wrap point sits wedge_m inside the surface
        center = np.array(scene['frame']['origin'])
        for w in fine['waypoints']:
            if w['stage'] == 'wrap':
                rho = scene['target']['radius'] - DEFAULT_PARAMS['wedge_m']  # keys on the circle, sub-steps on short chords
                assert rho - .0005 <= np.linalg.norm(np.array(w['tcp']) - center) <= rho + 1e-9


def test_parameters_are_range_checked():
    scene = scene_2d(ramen_box_project())
    for bad in ({'wedge_m': 1.}, {'unknown': 1.}, {'step_m': 0.}):
        with pytest.raises(ValueError):
            plan_scoop(scene, bad)


@pytest.mark.skipif(shutil.which('node') is None or not (ROOT / 'frontend/node_modules/planck').is_dir(),
                    reason='Node and frontend dependencies are required for the 2-D physics check')
def test_physics_clamp_lifts_the_top_bundle_and_the_single_scoop_does_not():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from cylinder_scoop_sweep import sweep
    results = sweep(ramen_box_project(), [None], {'mode': [2., 1.], 'wedge_m': [.001], 'insert_depth': [.3], 'step_m': [.04]}, workers=2)
    by_mode = {r['params']['mode']: r for r in results}
    assert by_mode[2.]['success'], by_mode[2.]['reason']
    assert by_mode[2.]['liftM'] > by_mode[2.]['requiredLiftM'] and by_mode[2.]['neighborMaxM'] < .02
    assert not by_mode[1.]['success']
