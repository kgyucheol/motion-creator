"""Compare pyrealsense2's depth deprojection with the numpy formula used in motioncreator/object_mask.py.

Runs on the robot (needs pyrealsense2 + numpy and exclusive camera access, so stop teleimager.service first).

  python deproject_benchmark.py            one-shot text report (center pixel, center patch, full-frame agreement, speed)
  python deproject_benchmark.py --serve    live: draws the center crosshair plus both results on the color image and
                                           publishes it as JPEG over ZMQ (needs cv2 + pyzmq); view it on the PC with
                                           MOTIONCREATOR_HEAD_CAMERA_ENDPOINT=tcp://<robot>:<port> ./run.sh
"""
import argparse
import time

import numpy as np
import pyrealsense2 as rs


def deproject(depth_m, fx, fy, cx, cy):
    """Same formula as motioncreator.object_mask.deproject."""
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    return np.dstack([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def timed_ms(function):
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000


def median_ms(function, repeats):
    return float(np.median([timed_ms(function)[1] for _ in range(repeats)]))


def library_cloud(depth_frame, height, width):
    return np.asanyarray(rs.pointcloud().calculate(depth_frame).get_vertices()).view(np.float32).reshape(height, width, 3)


def report(args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    try:
        for _ in range(args.frames):
            frames = pipeline.wait_for_frames(10000)
        depth_frame = frames.get_depth_frame()
    finally:
        pipeline.stop()

    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float64) * scale
    height, width = depth_m.shape
    u0, v0 = width // 2, height // 2
    print(f'stream {width}x{height} fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f} model={intr.model} coeffs={list(intr.coeffs)} scale={scale}')

    z = depth_frame.get_distance(u0, v0)
    library_point = np.array(rs.rs2_deproject_pixel_to_point(intr, [u0, v0], z))
    mine = deproject(depth_m, intr.fx, intr.fy, intr.ppx, intr.ppy)
    print(f'\ncenter pixel ({u0},{v0}) depth {z:.4f} m')
    print(f'  library {np.round(library_point, 6)}')
    print(f'  numpy   {np.round(mine[v0, u0], 6)}')
    print(f'  max abs diff {np.abs(library_point - mine[v0, u0]).max():.3e} m')

    half = args.patch // 2
    vertices = library_cloud(depth_frame, height, width)
    window = (slice(v0 - half, v0 + half + 1), slice(u0 - half, u0 + half + 1))
    valid = depth_m[window] > 0
    print(f'\ncenter {args.patch}x{args.patch} patch, valid {int(valid.sum())}/{valid.size}')
    if valid.any():
        print(f'  library pointcloud median {np.round(np.median(vertices[window][valid], axis=0), 6)}')
        print(f'  numpy              median {np.round(np.median(mine[window][valid], axis=0), 6)}')

    full_valid = depth_m > 0
    diff = np.abs(vertices[full_valid].astype(np.float64) - mine[full_valid])
    print(f'\nfull frame, valid pixels {int(full_valid.sum())}/{full_valid.size}')
    print(f'  max abs diff {diff.max():.3e} m, mean {diff.mean():.3e} m (library pointcloud is float32)')

    pixels = [(int(x), int(y)) for y, x in zip(*np.nonzero(full_valid))][:2000]
    z_values = [depth_frame.get_distance(x, y) for x, y in pixels]

    def library_per_pixel():
        for (x, y), z_value in zip(pixels, z_values):
            rs.rs2_deproject_pixel_to_point(intr, [x, y], z_value)

    per_pixel = median_ms(library_per_pixel, args.repeats) / len(pixels)
    print(f'\nspeed (median of {args.repeats}, ms)')
    print(f'  library rs2_deproject_pixel_to_point: {per_pixel * 1000:.2f} us/pixel -> {per_pixel * width * height:.0f} ms per full frame if looped in Python')
    print(f'  library rs.pointcloud (full frame):   {median_ms(lambda: library_cloud(depth_frame, height, width), args.repeats):.2f} ms')
    print(f'  numpy deproject (full frame):         {median_ms(lambda: deproject(depth_m, intr.fx, intr.fy, intr.ppx, intr.ppy), args.repeats):.2f} ms')


def serve(args):
    import cv2
    import zmq

    socket = zmq.Context().socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind(f'tcp://*:{args.port}')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    half = args.patch // 2
    print(f'publishing annotated frames on tcp://*:{args.port}  (Ctrl+C to stop)')
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(10000))
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float64) * scale
            height, width = depth_m.shape
            u0, v0 = width // 2, height // 2
            image = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))

            library_point = np.array(rs.rs2_deproject_pixel_to_point(intr, [u0, v0], depth_frame.get_distance(u0, v0)))
            window = (slice(v0 - half, v0 + half + 1), slice(u0 - half, u0 + half + 1))
            mine, numpy_ms = timed_ms(lambda: deproject(depth_m, intr.fx, intr.fy, intr.ppx, intr.ppy))
            vertices, cloud_ms = timed_ms(lambda: library_cloud(depth_frame, height, width))
            valid = depth_m[window] > 0
            patch_library = np.median(vertices[window][valid], axis=0) if valid.any() else None
            patch_numpy = np.median(mine[window][valid], axis=0) if valid.any() else None

            cv2.line(image, (u0 - 20, v0), (u0 + 20, v0), (0, 255, 0), 1)
            cv2.line(image, (u0, v0 - 20), (u0, v0 + 20), (0, 255, 0), 1)
            cv2.rectangle(image, (u0 - half, v0 - half), (u0 + half, v0 + half), (0, 255, 255), 1)
            lines = [f'center ({u0},{v0}) depth {depth_frame.get_distance(u0, v0):.3f} m',
                     'library %+.4f %+.4f %+.4f' % tuple(library_point),
                     'numpy   %+.4f %+.4f %+.4f' % tuple(mine[v0, u0])]
            if patch_library is not None:
                lines += ['patch %dx%d median' % (args.patch, args.patch),
                          'library %+.4f %+.4f %+.4f' % tuple(patch_library),
                          'numpy   %+.4f %+.4f %+.4f' % tuple(patch_numpy)]
            lines.append('full frame: pointcloud %.1f ms, numpy %.1f ms' % (cloud_ms, numpy_ms))
            for index, text in enumerate(lines):
                cv2.putText(image, text, (10, 24 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, text, (10, 24 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            socket.send(cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='publish an annotated live image over ZMQ instead of printing a report')
    parser.add_argument('--port', type=int, default=55556)
    parser.add_argument('--frames', type=int, default=30, help='report mode: frames to discard while auto-exposure settles')
    parser.add_argument('--patch', type=int, default=21, help='odd side length of the center patch (pixels)')
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    serve(args) if args.serve else report(args)


if __name__ == '__main__':
    main()
