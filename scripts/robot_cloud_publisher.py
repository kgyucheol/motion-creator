"""Robot-side RealSense publisher: color JPEG plus a decimated, colored point cloud over ZMQ PUB.

Runs on the robot in the teleimager conda env (pyrealsense2, numpy, cv2, pyzmq). It opens the camera exclusively, so
teleimager.service must be stopped while it runs. The color stream uses the same wire format as teleimager (raw JPEG
bytes, PUB with CONFLATE) so the motioncreator head camera view keeps working.

Cloud message (little endian): b'PC01' | uint32 n | uint16 stride | uint16 reserved | int16[n,3] xyz in mm | uint8[n,3] rgb
Points are in the color camera optical frame (x right, y down, z forward) and are not moved by any mount offset;
the receiver applies the d435_link conversion and any lens offset.

Usage: python robot_cloud_publisher.py [--color-port 55555] [--cloud-port 55557] [--stride 4] [--cloud-fps 10] [--max-range 3.0]
"""
import argparse
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

WIDTH, HEIGHT, FPS = 848, 480, 30


def pack_cloud(depth_m, color_bgr, intr, stride, max_range):
    u = np.arange(0, WIDTH, stride)
    v = np.arange(0, HEIGHT, stride)
    z = depth_m[::stride, ::stride]
    valid = (z > 0) & (z <= max_range)
    uu, vv = np.meshgrid(u, v)
    x = (uu - intr.ppx) * z / intr.fx
    y = (vv - intr.ppy) * z / intr.fy
    xyz = np.stack([x[valid], y[valid], z[valid]], axis=1)
    rgb = color_bgr[::stride, ::stride][valid][:, ::-1]
    count = len(xyz)
    header = struct.pack('<4sIHH', b'PC01', count, stride, 0)
    return header + np.round(xyz * 1000).astype('<i2').tobytes() + np.ascontiguousarray(rgb).tobytes()


def publisher(context, port):
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind(f'tcp://*:{port}')
    return socket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color-port', type=int, default=55555)
    parser.add_argument('--cloud-port', type=int, default=55557)
    parser.add_argument('--stride', type=int, default=4, help='keep every Nth pixel in x and y')
    parser.add_argument('--cloud-fps', type=float, default=10.0)
    parser.add_argument('--max-range', type=float, default=3.0, help='drop points farther than this (m)')
    parser.add_argument('--jpeg-quality', type=int, default=80)
    args = parser.parse_args()

    context = zmq.Context()
    color_socket = publisher(context, args.color_port)
    cloud_socket = publisher(context, args.cloud_port)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f'color JPEG on tcp://*:{args.color_port}, cloud on tcp://*:{args.cloud_port} (stride {args.stride}, {args.cloud_fps} fps, <= {args.max_range} m). Ctrl+C to stop')
    last_cloud = 0.0
    sent_color = sent_cloud = 0
    cloud_bytes = 0
    report_at = time.monotonic() + 5
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(10000))
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            color_socket.send(cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])[1].tobytes())
            sent_color += 1
            now = time.monotonic()
            if now - last_cloud >= 1 / args.cloud_fps - 1 / (2 * FPS):
                last_cloud = now
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
                message = pack_cloud(depth_m, color, intr, args.stride, args.max_range)
                cloud_socket.send(message)
                sent_cloud += 1
                cloud_bytes += len(message)
            if now >= report_at:
                print(f'color {sent_color / 5:.1f}/s, cloud {sent_cloud / 5:.1f}/s, {cloud_bytes / max(sent_cloud, 1) / 1024:.0f} KiB per cloud', flush=True)
                sent_color = sent_cloud = cloud_bytes = 0
                report_at = now + 5
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
