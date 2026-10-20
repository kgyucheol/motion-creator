"""Run plane-removal object masks on an rs_capture.npz and write overlay/mask PNGs.

Usage: python scripts/object_mask_demo.py capture.npz out_dir [--min-extent M --max-extent M]
"""
import argparse
from pathlib import Path

import numpy as np

from motioncreator.object_mask import overlay, segment_objects, write_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture')
    parser.add_argument('out_dir')
    parser.add_argument('--min-extent', type=float, help='keep objects whose longest side is at least this many meters')
    parser.add_argument('--max-extent', type=float, help='keep objects whose longest side is at most this many meters')
    args = parser.parse_args()
    data = np.load(args.capture)
    depth = data['depth'].astype(np.float64) * float(data['depth_scale'])
    extent_range = None
    if args.min_extent is not None or args.max_extent is not None:
        extent_range = (args.min_extent or 0., args.max_extent or 10.)
    plane, objects = segment_objects(depth, data['intrinsics'], extent_range=extent_range)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_png(out / 'overlay.png', overlay(data['color'], objects))
    write_png(out / 'color.png', data['color'][..., ::-1])
    print(f"plane normal {np.round(plane['normal'], 3)} distance {plane['d']:.3f} m, inliers {plane['plane_fraction']:.0%}")
    for number, item in enumerate(objects):
        write_png(out / f'mask_{number}.png', item['mask'])
        x, y, z = item['centroid']
        print(f"object {number}: {item['pixels']} px, centroid camera xyz ({x:.3f}, {y:.3f}, {z:.3f}) m, "
              f"extents {np.round(item['extents'], 3)} m, top {item['max_height'] * 100:.1f} cm above plane")
    if not objects:
        print('물체를 찾지 못했습니다.')


if __name__ == '__main__':
    main()
