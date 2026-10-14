import threading
import time

import pytest

zmq = pytest.importorskip('zmq')

from motioncreator.head_camera import HeadCameraStream, jpeg_size


def publisher():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    port = socket.bind_to_random_port('tcp://127.0.0.1')
    stop = threading.Event()

    def run():
        count = 0
        while not stop.is_set():
            socket.send(b'jpeg-%d' % count)
            count += 1
            time.sleep(.01)
        socket.close(0)
        context.term()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return f'tcp://127.0.0.1:{port}', stop, thread


def test_receives_latest_frames_only_while_viewed():
    endpoint, stop, thread = publisher()
    stream = HeadCameraStream(endpoint)
    try:
        assert stream.status()['receiving'] is False
        stream.acquire()
        first = stream.wait_frame(0, 3.0)
        assert first is not None and first[1].startswith(b'jpeg-')
        second = stream.wait_frame(first[0], 3.0)
        assert second[0] > first[0]
        assert stream.status()['receiving'] is True
        stream.release()
        for _ in range(50):
            if stream.status()['age'] is None:
                break
            time.sleep(.05)
        assert stream.status()['age'] is None
    finally:
        stop.set()
        thread.join(2)


def test_wait_frame_times_out_without_publisher():
    stream = HeadCameraStream('tcp://127.0.0.1:9')
    stream.acquire()
    try:
        assert stream.wait_frame(0, .2) is None
        assert stream.status()['receiving'] is False
    finally:
        stream.release()


def test_jpeg_size_reads_the_frame_marker():
    # SOI, APP0 (16 bytes), SOF0 for 848x480
    data = b'\xff\xd8' + b'\xff\xe0\x00\x10' + b'JFIF\x00' + bytes(9) + b'\xff\xc0\x00\x11\x08\x01\xe0\x03\x50\x03' + bytes(20)
    assert jpeg_size(data) == (848, 480)
    assert jpeg_size(b'not a jpeg') is None
