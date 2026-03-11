"""ROS 2 bridge node: forwards scan commands and camera frames to monty_comp.

Services:
    ~/start_scan  (std_srvs/Trigger)  — start a turntable scan
    ~/stop_scan   (std_srvs/Trigger)  — stop the current scan

Publishers:
    ~/scan_status (std_msgs/String)   — JSON status from monty_comp

ZMQ channels:
    Port 5560 (REQ → monty_comp REP): control commands (start/stop/status)
    Port 5561 (REP ← monty_comp REQ): frame capture server
"""

from __future__ import annotations

import json
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    import zmq
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False


class ScanControlBridge(Node):
    def __init__(self):
        super().__init__("scan_control_bridge")

        self.declare_parameter("zmq_host", "monty_comp")
        self.declare_parameter("zmq_port", 5560)
        self.declare_parameter("frame_port", 5561)
        self.declare_parameter("object_name", "unknown")
        self.declare_parameter("max_steps", 1000)
        self.declare_parameter("status_poll_hz", 0.1)

        self._zmq_host = self.get_parameter("zmq_host").value
        self._zmq_port = self.get_parameter("zmq_port").value
        self._frame_port = self.get_parameter("frame_port").value
        self._zmq_err_logged = False

        # Latest camera frame (guarded by lock)
        self._frame_lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._frame_h = 0
        self._frame_w = 0
        self._frame_encoding = ""

        self._ctx = None
        self._cmd_socket = None
        self._frame_thread = None

        if _ZMQ_AVAILABLE:
            self._ctx = zmq.Context()

            # Command socket (REQ) → monty_comp (REP) on port 5560
            self._cmd_socket = self._ctx.socket(zmq.REQ)
            self._cmd_socket.setsockopt(zmq.RCVTIMEO, 3000)
            self._cmd_socket.setsockopt(zmq.SNDTIMEO, 3000)
            self._cmd_socket.setsockopt(zmq.LINGER, 0)
            endpoint = f"tcp://{self._zmq_host}:{self._zmq_port}"
            self._cmd_socket.connect(endpoint)

            # Frame server (REP) on port 5561 — serves camera frames to monty_comp
            self._frame_thread = threading.Thread(
                target=self._frame_server_loop, daemon=True
            )
            self._frame_thread.start()
        else:
            self.get_logger().error(
                "pyzmq not installed in ros2_comp — scan bridge disabled"
            )

        # Subscribe to the camera image topic
        self._image_sub = self.create_subscription(
            Image, "/cam/color/image_raw", self._image_cb, 1
        )

        self._start_srv = self.create_service(
            Trigger, "~/start_scan", self._start_cb
        )
        self._stop_srv = self.create_service(
            Trigger, "~/stop_scan", self._stop_cb
        )
        self._status_pub = self.create_publisher(String, "~/scan_status", 10)

        poll_hz = self.get_parameter("status_poll_hz").value
        if poll_hz > 0:
            self._poll_timer = self.create_timer(1.0 / poll_hz, self._poll_status)

        self.get_logger().info(
            f"Scan control bridge ready — cmd zmq://{self._zmq_host}:{self._zmq_port}, "
            f"frame server on port {self._frame_port}"
        )

    # ------------------------------------------------------------------
    # Camera frame handling
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        with self._frame_lock:
            self._latest_frame = bytes(msg.data)
            self._frame_h = msg.height
            self._frame_w = msg.width
            self._frame_encoding = msg.encoding

    def _frame_server_loop(self):
        """REP socket that serves camera frames to monty_comp."""
        sock = self._ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{self._frame_port}")
        self.get_logger().info(
            f"Frame server listening on tcp://*:{self._frame_port}"
        )

        while rclpy.ok():
            try:
                if sock.poll(500):
                    _ = sock.recv()
                    with self._frame_lock:
                        if self._latest_frame is not None:
                            header = struct.pack(
                                "<III",
                                self._frame_h,
                                self._frame_w,
                                3 if "rgb" in self._frame_encoding.lower() else 3,
                            )
                            sock.send(header + self._latest_frame)
                        else:
                            sock.send(b"NO_FRAME")
            except zmq.ZMQError:
                break

        sock.close()

    # ------------------------------------------------------------------
    # Command socket helpers
    # ------------------------------------------------------------------

    def _zmq_request(self, msg: dict) -> dict:
        if self._cmd_socket is None:
            return {"status": "error", "message": "ZMQ not available"}
        try:
            self._cmd_socket.send_string(json.dumps(msg))
            raw = self._cmd_socket.recv_string()
            if self._zmq_err_logged:
                self.get_logger().info("ZMQ: monty_comp scan server connected")
                self._zmq_err_logged = False
            return json.loads(raw)
        except zmq.ZMQError as e:
            if not self._zmq_err_logged:
                self.get_logger().warn(
                    f"ZMQ: monty_comp scan server not reachable ({e}). "
                    "Will keep retrying silently."
                )
                self._zmq_err_logged = True
            self._cmd_socket.close()
            self._cmd_socket = self._ctx.socket(zmq.REQ)
            self._cmd_socket.setsockopt(zmq.RCVTIMEO, 3000)
            self._cmd_socket.setsockopt(zmq.SNDTIMEO, 3000)
            self._cmd_socket.setsockopt(zmq.LINGER, 0)
            endpoint = f"tcp://{self._zmq_host}:{self._zmq_port}"
            self._cmd_socket.connect(endpoint)
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _start_cb(self, request, response):
        object_name = self.get_parameter("object_name").value
        max_steps = self.get_parameter("max_steps").value

        result = self._zmq_request({
            "action": "start",
            "object_name": object_name,
            "max_steps": max_steps,
        })

        response.success = result.get("status") == "started"
        response.message = json.dumps(result)
        self.get_logger().info(f"start_scan → {result.get('status', '?')}")
        return response

    def _stop_cb(self, request, response):
        result = self._zmq_request({"action": "stop"})
        response.success = result.get("status") == "stopped"
        response.message = json.dumps(result)
        self.get_logger().info(f"stop_scan → {result.get('status', '?')}")
        return response

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def _poll_status(self):
        result = self._zmq_request({"action": "status"})
        msg = String()
        msg.data = json.dumps(result)
        self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        if self._cmd_socket is not None:
            self._cmd_socket.close()
        if self._ctx is not None:
            self._ctx.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ScanControlBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
