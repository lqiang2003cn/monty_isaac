#!/usr/bin/env python3
# coding: utf-8
"""
ROS2 bridge: subscribes to /x3plus/joint_commands, publishes /x3plus/joint_states.
Forwards commands to the real robot via ZMQ (radians on ROS side, degrees on HW side).

At startup: probes ZMQ service; optionally starts remote service via SSH if
ZMQ_REMOTE_SSH and ZMQ_REMOTE_START_CMD are set. Exits with clear logs if connection fails.

Mapping (verified against x3plus_isaac.urdf and x3plus_serial.py):
  deg = rad * 180/pi + 90       rad = (deg - 90) * pi/180

  Joint       URDF (rad)         -> Hardware (deg)
  ---------   ------------------    ----------------
  joint 1-4   [-pi/2,  pi/2]     -> [  0, 180]
  joint 5     [-pi/2,  pi  ]     -> [  0, 270]
  grip        [-1.54,  0   ]     -> [  2,  90]  (partial servo range)

  python3 zmq_bridge_node.py
"""

import json
import math
import os
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
import zmq

from lqtech_zmq_client import LqtechZMQClient
from zmq_protocol import GET_JOINT_POSITION_ARRAY

JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "grip_joint",
]

# Per-joint hardware degree limits (x3plus_serial.py)
JOINT_DEG_MAX = [180, 180, 180, 180, 270, 180]


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi + 90.0


def deg_to_rad(deg: float) -> float:
    return (deg - 90.0) * math.pi / 180.0


def clamp_deg(deg: float, joint_idx: int) -> int:
    return int(round(max(0.0, min(deg, float(JOINT_DEG_MAX[joint_idx])))))


def probe_connection(host: str, port: int, timeout_ms: int = 5000) -> bool:
    """Try one getJointPositionArray request; return True if service responds correctly."""
    ctx = None
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect("tcp://%s:%d" % (host, port))
        sock.send_string(json.dumps({"method": GET_JOINT_POSITION_ARRAY}))
        resp = sock.recv_string()
        data = json.loads(resp)
        if "error" in data:
            return False
        if "joint_array" not in data or len(data["joint_array"]) != 6:
            return False
        return True
    except Exception:
        return False
    finally:
        if ctx is not None:
            try:
                ctx.destroy(linger=0)
            except Exception:
                pass


def try_start_remote_service(ssh_target: str, start_cmd: str, log_fn, password: str = None) -> bool:
    """Run start_cmd on remote host via SSH. log_fn(msg) for logging. Returns True if SSH exit 0.
    If password is set, use sshpass for non-interactive auth (sshpass must be installed).
    """
    log_fn("Attempting to start remote ZMQ service: ssh %s '<cmd>'" % ssh_target)
    if password:
        log_fn("Using password authentication (ZMQ_REMOTE_SSH_PASSWORD).")
    try:
        ssh_opts = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no", ssh_target, start_cmd]
        if password:
            # sshpass -p <password> ssh ... (password never logged)
            proc = subprocess.run(
                ["sshpass", "-p", password, "ssh"] + ssh_opts,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ},  # pass through env
            )
        else:
            proc = subprocess.run(
                ["ssh"] + ssh_opts,
                capture_output=True,
                text=True,
                timeout=30,
            )
        out = proc
        if out.returncode != 0:
            log_fn("SSH command failed (exit %d). stderr: %s" % (out.returncode, out.stderr or "(none)"))
            return False
        if out.stdout or out.stderr:
            log_fn("SSH stdout: %s  stderr: %s" % (out.stdout or "", out.stderr or ""))
        return True
    except subprocess.TimeoutExpired:
        log_fn("SSH command timed out after 30s")
        return False
    except FileNotFoundError as e:
        if password:
            log_fn("'sshpass' or 'ssh' not found; install openssh-client and sshpass for password auth.")
        else:
            log_fn("'ssh' not found; cannot auto-start remote service.")
        return False
    except Exception as e:
        log_fn("SSH command error: %s" % e)
        return False


def ensure_zmq_connection(host: str, port: int, log_fn) -> None:
    """
    Probe ZMQ service; optionally try to start it via SSH. Exit process with clear message if failed.
    Env:
      ZMQ_REMOTE_SSH           e.g. wheeltec@192.168.31.142 (if set, try SSH start on first failure)
      ZMQ_REMOTE_SSH_PASSWORD  optional; if set, use sshpass for non-interactive SSH (install sshpass in image)
      ZMQ_REMOTE_ENV           remote conda/venv name (default gRPC); used in default ZMQ_REMOTE_START_CMD as %(env)s
      ZMQ_REMOTE_START_CMD     default runs script in conda env gRPC: "cd ... && nohup conda run -n %(env)s python ... &"
      ZMQ_PROBE_TIMEOUT_MS     default 5000
      ZMQ_RETRY_COUNT          default 5
      ZMQ_RETRY_DELAY_SEC      default 3
    """
    timeout_ms = int(os.environ.get("ZMQ_PROBE_TIMEOUT_MS", "5000"))
    retry_count = int(os.environ.get("ZMQ_RETRY_COUNT", "5"))
    retry_delay = float(os.environ.get("ZMQ_RETRY_DELAY_SEC", "3"))
    ssh_target = os.environ.get("ZMQ_REMOTE_SSH", "").strip()
    remote_env = os.environ.get("ZMQ_REMOTE_ENV", "gRPC").strip()
    # Default: source conda (common install paths), activate env, then run script (conda not in PATH over SSH)
    start_cmd_tpl = os.environ.get(
        "ZMQ_REMOTE_START_CMD",
        "bash -c 'source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null; conda activate %(env)s && cd /home/wheeltec/grpc/lqtech_grpc_x3plus/python/x3plus && nohup python lqtech_zmq_service.py --port %(port)s &'",
    ).strip()

    ssh_password = os.environ.get("ZMQ_REMOTE_SSH_PASSWORD", "").strip() or None

    for attempt in range(1, retry_count + 1):
        log_fn("[%d/%d] Probing ZMQ service at %s:%d ..." % (attempt, retry_count, host, port))
        if probe_connection(host, port, timeout_ms):
            log_fn("ZMQ service at %s:%d is reachable." % (host, port))
            return
        log_fn("Probe failed (connection refused or timeout).")
        if attempt == 1 and ssh_target and start_cmd_tpl:
            start_cmd = start_cmd_tpl % {"port": port, "env": remote_env}
            try_start_remote_service(ssh_target, start_cmd, log_fn, password=ssh_password)
            log_fn("Waiting %s s for remote service to start ..." % retry_delay)
            time.sleep(retry_delay)
        elif attempt < retry_count:
            log_fn("Retrying in %s s ..." % retry_delay)
            time.sleep(retry_delay)

    log_fn("")
    log_fn("FATAL: Could not reach ZMQ service at %s:%d after %d attempts." % (host, port, retry_count))
    log_fn("  - Ensure lqtech_zmq_service.py (or equivalent) is running on the robot.")
    log_fn("  - If using SSH auto-start, set ZMQ_REMOTE_SSH and optionally ZMQ_REMOTE_START_CMD.")
    log_fn("  - Check connectivity: nc -zv %s %d" % (host, port))
    log_fn("")
    sys.exit(1)


class ZMQBridgeNode(Node):
    def __init__(self):
        super().__init__("zmq_bridge_node")
        self._lock = threading.Lock()
        self._latest_positions = None  # list of 6 floats (rad) or None

        host = os.environ.get("ZMQ_HOST", "192.168.31.142")
        port = int(os.environ.get("ZMQ_PORT", "5555"))
        self.get_logger().info("ZMQ client connecting to %s:%d" % (host, port))
        self._zmq_client = LqtechZMQClient(host=host, port=port)

        self._sub = self.create_subscription(
            JointState,
            JOINT_COMMANDS_TOPIC,
            self._joint_commands_cb,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._pub = self.create_publisher(
            JointState,
            JOINT_STATES_TOPIC,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        self._state_timer = self.create_timer(0.02, self._publish_joint_state)

        self.get_logger().info(
            "Bridge: subscribe %s, publish %s" % (JOINT_COMMANDS_TOPIC, JOINT_STATES_TOPIC)
        )

    def _joint_commands_cb(self, msg: JointState) -> None:
        if not msg.name or len(msg.position) < len(msg.name):
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        positions_rad = []
        for name in JOINT_NAMES:
            if name in name_to_pos:
                positions_rad.append(float(name_to_pos[name]))
            else:
                self.get_logger().warn("Command missing joint %s, skipping" % name)
                return
        if len(positions_rad) != 6:
            return
        if any(math.isnan(r) or math.isinf(r) for r in positions_rad):
            return
        degs = [clamp_deg(rad_to_deg(r), i) for i, r in enumerate(positions_rad)]
        with self._lock:
            self._latest_positions = list(positions_rad)
        try:
            self._zmq_client.set_joint_position_array(degs)
        except Exception as e:
            self.get_logger().error("ZMQ set_joint_position_array failed: %s" % e)

    def _publish_joint_state(self) -> None:
        try:
            degs = self._zmq_client.get_joint_position_array()
        except Exception as e:
            self.get_logger().error("ZMQ get_joint_position_array failed: %s" % e)
            return
        if len(degs) != 6:
            return
        rads = [deg_to_rad(d) for d in degs]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = rads
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        try:
            self._zmq_client.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    host = os.environ.get("ZMQ_HOST", "192.168.31.142")
    port = int(os.environ.get("ZMQ_PORT", "5555"))
    ensure_zmq_connection(host, port, log_fn=print)

    rclpy.init(args=args)
    node = ZMQBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
