import collections
import os
import threading
import time

DEFAULT_ENDPOINT = 'tcp://192.168.123.164:55555'
POLL_MS = 300
FPS_WINDOW = 2.0


def jpeg_size(data):
    """(width, height) from the JPEG start-of-frame marker, or None if it is not found."""
    position = 2
    while position + 9 < len(data):
        if data[position] != 0xFF:
            return None
        marker = data[position + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[position + 7:position + 9], 'big'), int.from_bytes(data[position + 5:position + 7], 'big')
        position += 2 + int.from_bytes(data[position + 2:position + 4], 'big')
    return None


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
        self._size = None
        self._arrivals = collections.deque()

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
            while self._frame is None or self._sequence <= after:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._sequence, self._frame

    def status(self):
        with self._condition:
            age = None if self._frame_time is None else self._clock() - self._frame_time
            size = self._size if age is not None else None
            now = self._clock()
            recent = [t for t in self._arrivals if now - t <= FPS_WINDOW]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else None
            return {'endpoint': self.endpoint, 'viewers': self._viewers,
                    'receiving': age is not None and age < 2.0, 'age': age,
                    'width': size[0] if size else None, 'height': size[1] if size else None,
                    'fps': fps if age is not None and age < 2.0 else None}

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
                        self._size = None
                        self._arrivals.clear()
                        self._thread = None
                        return
                if not socket.poll(POLL_MS):
                    continue
                data = socket.recv()
                with self._condition:
                    self._frame = data
                    self._size = jpeg_size(data)
                    self._sequence += 1
                    self._frame_time = self._clock()
                    self._arrivals.append(self._frame_time)
                    while self._arrivals and self._frame_time - self._arrivals[0] > FPS_WINDOW:
                        self._arrivals.popleft()
                    self._condition.notify_all()
        finally:
            socket.close()
            context.term()


head_camera = HeadCameraStream()
