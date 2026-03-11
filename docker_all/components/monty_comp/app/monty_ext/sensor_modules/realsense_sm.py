"""Camera capture via ZMQ frame server in ros2_comp.

The RealSense camera is opened by the ``realsense2_camera_node`` in
ros2_comp (which also feeds RViz).  This module fetches frames from the
``scan_control_bridge``'s frame server (ZMQ REP on port 5561) instead
of opening the device directly, avoiding the "device busy" conflict.

Protocol:
    REQ sends:  b"capture"
    REP returns: 12-byte header (height, width, channels as uint32 LE)
                 + raw RGB bytes (height * width * channels)
    OR:          b"NO_FRAME" if no frame is available yet
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RealsenseConfig:
    """Configuration for the frame capture client."""
    frame_host: str = "ros2_comp"
    frame_port: int = 5561
    timeout_ms: int = 5000
    retry_delay: float = 0.2


class RealsenseCapture:
    """Fetches camera frames from the ZMQ frame server in ros2_comp."""

    def __init__(self, config: RealsenseConfig | None = None):
        self._config = config or RealsenseConfig()
        self._ctx = None
        self._socket = None
        self._started = False

    def start(self):
        if self._started:
            return

        import zmq

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._config.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._config.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)

        endpoint = f"tcp://{self._config.frame_host}:{self._config.frame_port}"
        self._socket.connect(endpoint)
        self._started = True
        logger.info("Frame client connected to %s", endpoint)

    def stop(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
        self._started = False

    def capture(self) -> np.ndarray:
        """Capture a single RGB frame from the frame server.

        Returns:
            (H, W, 3) uint8 numpy array in RGB order.
        """
        if not self._started:
            self.start()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._socket.send(b"capture")
                data = self._socket.recv()

                if data == b"NO_FRAME":
                    if attempt < max_retries - 1:
                        time.sleep(self._config.retry_delay)
                        continue
                    raise RuntimeError("Frame server has no frame available")

                h, w, c = struct.unpack("<III", data[:12])
                pixels = np.frombuffer(data[12:], dtype=np.uint8).reshape(h, w, c)
                return pixels.copy()

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning("Frame capture attempt %d failed: %s", attempt + 1, e)
                    time.sleep(self._config.retry_delay)
                    # Reconnect socket on failure
                    import zmq
                    self._socket.close()
                    self._socket = self._ctx.socket(zmq.REQ)
                    self._socket.setsockopt(zmq.RCVTIMEO, self._config.timeout_ms)
                    self._socket.setsockopt(zmq.SNDTIMEO, self._config.timeout_ms)
                    self._socket.setsockopt(zmq.LINGER, 0)
                    endpoint = f"tcp://{self._config.frame_host}:{self._config.frame_port}"
                    self._socket.connect(endpoint)
                else:
                    raise
