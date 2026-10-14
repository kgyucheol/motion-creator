import os
import threading
import time

DEFAULT_ENDPOINT = 'tcp://192.168.123.164:55555'
POLL_MS = 300


def head_camera_endpoint():
    return os.environ.get('MOTIONCREATOR_HEAD_CAMERA_ENDPOINT') or DEFAULT_ENDPOINT


class HeadCameraStream:
    """teleimager ZMQ PUB 스트림(JPEG 바이트)을 구독해 최신 프레임만 유지한다. 시청자가 있을 때만 연결한다."""

    def __init__(self, endpoint=None, clock=time.monotonic):
        self.endpoint = endpoint or head_camera_endpoint()
        self._clock = clock
        self._condition = threading.Condition()
        self._viewers = 0
        self._thread = None
        self._frame = None
        self._sequence = 0
        self._frame_time = None

    def acquire(self):
        try:
            import zmq  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('pyzmq가 설치되어 있지 않습니다. pip install pyzmq') from exc
        with self._condition:
            self._viewers += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name='head-camera-zmq', daemon=True)
                self._thread.start()

    def release(self):
        with self._condition:
            self._viewers = max(0, self._viewers - 1)

    def wait_frame(self, after, timeout):
        """`after`보다 새로운 프레임이 오면 (순번, JPEG)를 반환하고, 시간 안에 없으면 None."""
        deadline = self._clock() + timeout
        with self._condition:
            while self._sequence <= after:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._sequence, self._frame

    def status(self):
        with self._condition:
            age = None if self._frame_time is None else self._clock() - self._frame_time
            return {'endpoint': self.endpoint, 'viewers': self._viewers,
                    'receiving': age is not None and age < 2.0, 'age': age}

    def _run(self):
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        socket.connect(self.endpoint)
        try:
            while True:
                with self._condition:
                    if self._viewers == 0:
                        self._frame = None
                        self._frame_time = None
                        self._thread = None
                        return
                if not socket.poll(POLL_MS):
                    continue
                data = socket.recv()
                with self._condition:
                    self._frame = data
                    self._sequence += 1
                    self._frame_time = self._clock()
                    self._condition.notify_all()
        finally:
            socket.close()
            context.term()


head_camera = HeadCameraStream()
