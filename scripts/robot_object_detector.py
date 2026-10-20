"""Robot-side detector: YOLO11-seg ONNX on the color stream, 3-D ridge lines of the detected cup rows from the point cloud.

Runs on the robot next to robot_cloud_publisher.py (needs numpy, opencv, onnxruntime, pyzmq; ~/robot_cloud/venv). It
subscribes to the local color JPEG (55555) and cloud (55557) streams, segments the latest color frame about once per
`--interval` seconds, picks the cloud points under each mask and publishes one JSON message on tcp://*:55558 (PUB, CONFLATE).

Each detected object is one row of cups lying on their side. The message gives the *visible ridge line* of the row (the
line along the camera-facing surface), the camera-facing surface normal and the measured width; the cylinder axis is the
ridge moved `radius` along `-normal`, which the receiver does so the radius stays adjustable:

    {"type": "objects", "frame": "camera_optical", "stamp": <unix s>, "inference_ms": ..,
     "objects": [{"label", "confidence", "start": [x,y,z], "end": [x,y,z], "normal": [x,y,z], "axis": [x,y,z],
                  "length", "width", "depth_extent", "points", "clipped": [bool, bool]}]}

All vectors are meters in the color camera optical frame (x right, y down, z forward), the same frame as the cloud.
`start`/`end` are the ridge ends ordered along `axis` (which points to the image right); `clipped[i]` is true when that
end is at the image border, so the row continues beyond the camera view.

Usage: python robot_object_detector.py --model ~/robot_cloud/models/ramen_container.onnx [--interval 1.0] [--conf 0.4]
"""
import argparse
import json
import struct
import time

import numpy as np

INPUT_SIZE = 864
COLOR_SIZE = (848, 480)                       # width, height
INTRINSICS = (608.126, 608.878, 428.874, 256.104)      # fx, fy, cx, cy of the aligned color stream at 848x480
DEPTH_RANGE_M = (0.15, 3.0)
MIN_POINTS = 150
BORDER_PX = 8
LABELS = {0: 'ramen_container'}


def letterbox_params(width, height, size=INPUT_SIZE):
    """Scale and padding that ultralytics uses to fit (width, height) into a size x size square."""
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    return scale, new_w, new_h, round(pad_x - 0.1), round(pad_y - 0.1)


def preprocess(bgr, size=INPUT_SIZE):
    import cv2
    height, width = bgr.shape[:2]
    scale, new_w, new_h, left, top = letterbox_params(width, height, size)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return np.ascontiguousarray(tensor), (scale, new_w, new_h, left, top)


def nms(boxes, scores, iou_threshold):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        index = order[0]
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[index, 0], boxes[rest, 0]); y1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[index, 2], boxes[rest, 2]); y2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / (area(boxes[index]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode(output0, output1, placement, image_size=COLOR_SIZE, conf_threshold=0.4, iou_threshold=0.5):
    """(confidence, bool mask at image resolution) per instance from the raw YOLO-seg outputs (one class)."""
    import cv2
    scale, new_w, new_h, left, top = placement
    width, height = image_size
    prediction = output0[0].T                                  # (N, 4 + classes + 32)
    scores = prediction[:, 4]
    keep = scores >= conf_threshold
    prediction, scores = prediction[keep], scores[keep]
    if not len(scores):
        return []
    cx, cy, w, h = prediction[:, 0], prediction[:, 1], prediction[:, 2], prediction[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    picked = nms(boxes, scores, iou_threshold)
    proto = output1[0]                                          # (32, 216, 216)
    stride = INPUT_SIZE / proto.shape[1]
    results = []
    for index in picked:
        logits = np.tensordot(prediction[index, 5:], proto, axes=1)
        probability = 1 / (1 + np.exp(-logits))
        crop = probability[int(top / stride):int(np.ceil((top + new_h) / stride)), int(left / stride):int(np.ceil((left + new_w) / stride))]
        mask = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
        x1, y1, x2, y2 = (boxes[index] - [left, top, left, top]) / scale
        box_mask = np.zeros_like(mask)
        box_mask[max(int(y1), 0):int(np.ceil(y2)), max(int(x1), 0):int(np.ceil(x2))] = True
        results.append((float(scores[index]), mask & box_mask))
    return results


def parse_cloud(message):
    tag, count, stride, _ = struct.unpack('<4sIHH', message[:12])
    if tag != b'PC01' or len(message) != 12 + count * 9:
        raise ValueError('not a PC01 point cloud message')
    return np.frombuffer(message, '<i2', count * 3, 12).reshape(count, 3).astype(np.float64) / 1000.0


def project(points, intrinsics=INTRINSICS):
    fx, fy, cx, cy = intrinsics
    return points[:, 0] / points[:, 2] * fx + cx, points[:, 1] / points[:, 2] * fy + cy


def row_geometry(points, intrinsics=INTRINSICS, image_size=COLOR_SIZE, min_points=MIN_POINTS):
    """Ridge line, surface normal and width of the visible side of a row of cups, or None when too few points."""
    z = points[:, 2]
    points = points[np.isfinite(points).all(axis=1) & (z >= DEPTH_RANGE_M[0]) & (z <= DEPTH_RANGE_M[1])]
    if len(points) < min_points:
        return None
    median = np.median(points[:, 2])
    spread = 3 * 1.4826 * np.median(np.abs(points[:, 2] - median)) + 0.01     # background that bled into the mask
    points = points[np.abs(points[:, 2] - median) <= spread]
    if len(points) < min_points:
        return None
    centroid = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - centroid).T))
    axis, _, normal = (vectors[:, i] for i in np.argsort(values)[::-1])
    pivot = axis[0] if abs(axis[0]) > 1e-6 else axis[1]
    axis = axis if pivot >= 0 else -axis                       # points to the image right
    normal = normal if normal[2] <= 0 else -normal             # faces the camera (camera looks along +z)
    lateral = np.cross(normal, axis)
    local = (points - centroid) @ np.column_stack([axis, lateral, normal])
    a_low, a_high = np.percentile(local[:, 0], (0.5, 99.5))
    b_low, b_high = np.percentile(local[:, 1], (0.5, 99.5))
    width = b_high - b_low
    b_mid = (b_low + b_high) / 2
    band = np.abs(local[:, 1] - b_mid) <= 0.15 * width
    ridge = np.median(local[band, 2]) if band.sum() >= 20 else np.percentile(local[:, 2], 95)
    ends = [centroid + axis * a + lateral * b_mid + normal * ridge for a in (a_low, a_high)]
    u, v = project(np.array(ends), intrinsics)
    w, h = image_size
    clipped = [bool(u[i] < BORDER_PX or u[i] > w - BORDER_PX or v[i] < BORDER_PX or v[i] > h - BORDER_PX) for i in range(2)]
    return {'start': ends[0].tolist(), 'end': ends[1].tolist(), 'normal': normal.tolist(), 'axis': axis.tolist(),
            'length': float(a_high - a_low), 'width': float(width), 'depth_extent': float(np.ptp(local[:, 2])),
            'points': int(len(points)), 'clipped': clipped}


def detect_objects(session, bgr, cloud_points, conf_threshold=0.4, intrinsics=INTRINSICS):
    tensor, placement = preprocess(bgr)
    started = time.monotonic()
    output0, output1 = session.run(None, {session.get_inputs()[0].name: tensor})
    inference_ms = (time.monotonic() - started) * 1000
    height, width = bgr.shape[:2]
    u, v = project(cloud_points, intrinsics)
    column, row = np.round(u).astype(int), np.round(v).astype(int)
    on_image = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    objects = []
    for confidence, mask in decode(output0, output1, placement, (width, height), conf_threshold):
        selected = np.zeros(len(cloud_points), bool)
        selected[on_image] = mask[row[on_image], column[on_image]]
        geometry = row_geometry(cloud_points[selected], intrinsics, (width, height))
        if geometry is not None:
            objects.append({'label': LABELS[0], 'confidence': round(confidence, 3), **geometry})
    return objects, inference_ms


def main():
    import cv2
    import onnxruntime
    import zmq
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--color', default='tcp://127.0.0.1:55555')
    parser.add_argument('--cloud', default='tcp://127.0.0.1:55557')
    parser.add_argument('--port', type=int, default=55558)
    parser.add_argument('--interval', type=float, default=1.0, help='minimum seconds between detections')
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = onnxruntime.InferenceSession(args.model, options, providers=['CPUExecutionProvider'])
    context = zmq.Context()
    sockets = {}
    for name, endpoint in (('color', args.color), ('cloud', args.cloud)):
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        socket.connect(endpoint)
        sockets[name] = socket
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.CONFLATE, 1)
    output.bind(f'tcp://*:{args.port}')
    print(f'detecting on {args.color} + {args.cloud} every {args.interval}s, publishing JSON on tcp://*:{args.port}. Ctrl+C to stop', flush=True)
    last = 0.0
    while True:
        if not sockets['color'].poll(2000):
            continue
        color_message = sockets['color'].recv()
        if time.monotonic() - last < args.interval:
            continue
        sockets['cloud'].poll(300)                 # first cloud that arrives after this color frame
        try:
            cloud_message = sockets['cloud'].recv(zmq.NOBLOCK)
        except zmq.Again:
            continue
        frame = cv2.imdecode(np.frombuffer(color_message, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        objects, inference_ms = detect_objects(session, frame, parse_cloud(cloud_message), args.conf)
        last = time.monotonic()
        output.send_string(json.dumps({'type': 'objects', 'frame': 'camera_optical', 'stamp': time.time(),
                                       'inference_ms': round(inference_ms, 1), 'objects': objects}))
        print(f'{len(objects)} objects, inference {inference_ms:.0f} ms', flush=True)


if __name__ == '__main__':
    main()
