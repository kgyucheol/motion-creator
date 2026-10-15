"""Measure camera-frame XYZ at nine Go-board star points (1/6, 1/2, 5/6 of the image width and height).

Runs on the robot (pyrealsense2 + numpy + cv2, exclusive camera access: stop teleimager.service first).
Writes points.csv and grid_points.jpg into --out. Depth is aligned to color and is the per-pixel median of --frames frames,
then the median of a --patch x --patch window at each point. XYZ is in meters in the color camera frame
(x right, y down, z forward, origin at the camera optical center).

Usage: python grid_points_xyz.py --out /tmp/grid_points
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT = 848, 480
FRACTIONS = (1 / 6, 1 / 2, 5 / 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--frames', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--patch', type=int, default=11)
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depths = []
    try:
        for index in range(args.warmup + args.frames):
            frames = align.process(pipeline.wait_for_frames(10000))
            if index >= args.warmup:
                depths.append(np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float64))
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()

    stack = np.stack(depths)
    stack[stack == 0] = np.nan
    depth_m = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0) * scale

    half = args.patch // 2
    rows = []
    for row, fy in enumerate(FRACTIONS):
        for column, fx in enumerate(FRACTIONS):
            u, v = int(round(WIDTH * fx)), int(round(HEIGHT * fy))
            window = depth_m[v - half:v + half + 1, u - half:u + half + 1]
            valid = window[window > 0]
            z = float(np.median(valid)) if valid.size else 0.0
            x, y, z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z) if z > 0 else (float('nan'),) * 3
            rows.append({'id': row * 3 + column + 1, 'u': u, 'v': v, 'valid_pixels': int(valid.size), 'x_m': x, 'y_m': y, 'z_m': z,
                         'range_m': float(np.sqrt(x * x + y * y + z * z)) if valid.size else float('nan')})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'points.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: (f'{value:.4f}' if isinstance(value, float) else value) for key, value in item.items()} for item in rows)

    for item in rows:
        u, v = item['u'], item['v']
        cv2.drawMarker(color, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.rectangle(color, (u - half, v - half), (u + half, v + half), (0, 255, 255), 1)
        cv2.circle(color, (u, v), 3, (0, 0, 255), -1)
        label = f"{item['id']}: X{item['x_m']:+.3f} Y{item['y_m']:+.3f} Z{item['z_m']:.3f}"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]
        origin = (min(max(u - 70, 4), WIDTH - text_width - 4), v + 28 if v < HEIGHT - 40 else v - 18)
        cv2.putText(color, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(color, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / 'grid_points.jpg'), color, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'intrinsics fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f}')
    for item in rows:
        print(item)


if __name__ == '__main__':
    main()
