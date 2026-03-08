#!/usr/bin/env python3
"""
ZMQ X3plus robot bridge: subscribes to /x3plus/joint_commands, sends position
commands to hardware via ZMQ (remote_real_x3plus on Orin); reads positions and
publishes /x3plus/joint_states. Run with ros2_control bringup (mode:=zmq).

Uses opus_joint_config for rad/deg conversion (shared with real bridge).
"""

import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
import zmq

from monty_demo.opus_plan_and_imp.opus_joint_config import (
    ARM_GRIP_JOINTS,
    JOINT_NAMES,
    MIMIC_MAP,
    rad_to_deg,
    deg_to_rad,
)
from monty_demo.opus_plan_and_imp.lqtech_zmq_client import LqtechZMQClient
from monty_demo.opus_plan_and_imp.log_utils import make_file_logger
from monty_demo.opus_plan_and_imp.zmq_protocol import GET_JOINT_POSITION_ARRAY

JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

SERVO_ORDER = ARM_GRIP_JOINTS  # [arm_joint1, ..., arm_joint5, grip_joint]


def probe_connection(host: str, port: int, timeout_ms: int = 5000) -> bool:
    """Try one getJointPositionArray request; return True if service responds."""
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


def ensure_zmq_connection(host: str, port: int, log_fn) -> None:
    """Probe ZMQ service with retries. Exit process if connection fails."""
    timeout_ms = int(os.environ.get("ZMQ_PROBE_TIMEOUT_MS", "5000"))
    retry_count = int(os.environ.get("ZMQ_RETRY_COUNT", "5"))
    retry_delay = float(os.environ.get("ZMQ_RETRY_DELAY_SEC", "3"))

    for attempt in range(1, retry_count + 1):
        log_fn("[%d/%d] Probing ZMQ service at %s:%d ..." % (attempt, retry_count, host, port))
        if probe_connection(host, port, timeout_ms):
            log_fn("ZMQ service at %s:%d is reachable." % (host, port))
            return
        log_fn("Probe failed (connection refused or timeout).")
        if attempt < retry_count:
            log_fn("Retrying in %s s ..." % retry_delay)
            time.sleep(retry_delay)

    log_fn("")
    log_fn("FATAL: Could not reach ZMQ service at %s:%d after %d attempts." % (host, port, retry_count))
    log_fn("  - Ensure the ZMQ service (e.g. remote_real_x3plus container) is running on the robot.")
    log_fn("  - Check connectivity: nc -zv %s %d" % (host, port))
    log_fn("")
    sys.exit(1)


WATCHDOG_FAIL_THRESHOLD = 5
WATCHDOG_LOCKOUT_SEC = 10.0


class OpusX3PlusZMQBridge(Node):
    def __init__(self) -> None:
        super().__init__("opus_x3plus_zmq_bridge")
        self.declare_parameter("zmq_host", os.environ.get("ZMQ_HOST", "192.168.31.142"))
        self.declare_parameter("zmq_port", int(os.environ.get("ZMQ_PORT", "5555")))
        self.declare_parameter("state_publish_hz", 20.0)

        host = self.get_parameter("zmq_host").get_parameter_value().string_value
        port = self.get_parameter("zmq_port").get_parameter_value().integer_value

        self._flog = make_file_logger("zmq_bridge", "/tmp/x3plus_zmq_bridge.log")
        self._flog.info("=" * 60)
        self._flog.info("ZMQ bridge starting, host=%s:%d" % (host, port))

        self.get_logger().info("ZMQ bridge connecting to %s:%d" % (host, port))
        self._zmq_client = LqtechZMQClient(host=host, port=port)

        state_hz = self.get_parameter("state_publish_hz").get_parameter_value().double_value

        self._sub = self.create_subscription(
            JointState,
            JOINT_COMMANDS_TOPIC,
            self._joint_commands_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST),
        )
        self._pub = self.create_publisher(
            JointState,
            JOINT_STATES_TOPIC,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST),
        )
        self._state_timer = self.create_timer(1.0 / state_hz, self._publish_joint_state)
        self._cmd_count = 0
        self._state_log_count = 0

        self._consecutive_cmd_fails = 0
        self._consecutive_state_fails = 0
        self._cmd_lockout_until = 0.0

        self.get_logger().info(
            "ZMQ bridge: sub %s, pub %s, host=%s:%d" % (JOINT_COMMANDS_TOPIC, JOINT_STATES_TOPIC, host, port)
        )

    def _joint_commands_cb(self, msg: JointState) -> None:
        if not msg.name or len(msg.position) < len(msg.name):
            return

        now = time.time()
        if now < self._cmd_lockout_until:
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        angle_s = []
        for joint_name in SERVO_ORDER:
            if joint_name in name_to_pos:
                rad_val = float(name_to_pos[joint_name])
                if math.isnan(rad_val) or math.isinf(rad_val):
                    return
                deg = rad_to_deg(joint_name, rad_val)
                angle_s.append(deg)
            else:
                self.get_logger().warn("Command missing joint %s, skipping" % joint_name)
                return
        if len(angle_s) != 6:
            return
        try:
            self._zmq_client.set_joint_position_array(angle_s)
            self._consecutive_cmd_fails = 0
            self._cmd_count += 1
            if self._cmd_count == 1:
                self.get_logger().info("First command forwarded to hardware")
            rad_fmt = ", ".join(
                "%s=%.4f" % (n, name_to_pos[n]) for n in SERVO_ORDER if n in name_to_pos
            )
            deg_fmt = ", ".join("%.1f°" % d for d in angle_s)
            self._flog.debug(
                "CMD[%d]: rad=[%s] deg=[%s]" % (self._cmd_count, rad_fmt, deg_fmt)
            )
        except Exception as e:
            self._consecutive_cmd_fails += 1
            self._flog.error(
                "ZMQ set_joint_position_array failed (%d consecutive): %s"
                % (self._consecutive_cmd_fails, e)
            )
            if self._consecutive_cmd_fails >= WATCHDOG_FAIL_THRESHOLD:
                self._cmd_lockout_until = now + WATCHDOG_LOCKOUT_SEC
                msg = (
                    "WATCHDOG: %d consecutive command failures — "
                    "refusing new commands for %.0fs. "
                    "Check ZMQ service / Orin connectivity."
                    % (self._consecutive_cmd_fails, WATCHDOG_LOCKOUT_SEC)
                )
                self.get_logger().error(msg)
                self._flog.error(msg)

    def _publish_joint_state(self) -> None:
        try:
            degs = self._zmq_client.get_joint_position_array()
            self._consecutive_state_fails = 0
        except Exception as e:
            self._consecutive_state_fails += 1
            self._flog.error(
                "ZMQ get_joint_position_array failed (%d consecutive): %s"
                % (self._consecutive_state_fails, e)
            )
            if self._consecutive_state_fails == WATCHDOG_FAIL_THRESHOLD:
                msg = (
                    "WATCHDOG: %d consecutive state-read failures — "
                    "ZMQ service may be down. Joint states will be stale."
                    % self._consecutive_state_fails
                )
                self.get_logger().error(msg)
                self._flog.error(msg)
            return
        if len(degs) != 6:
            return
        n_bad = sum(1 for d in degs if d < 0)
        if n_bad > 0:
            self._flog.error("Invalid servo positions: %s" % degs)
            if not getattr(self, "_hw_warn_logged", False):
                self.get_logger().error(
                    "Hardware returning invalid servo positions "
                    "(negative = no servo response). Check Orin serial/power."
                )
                self._hw_warn_logged = True
        else:
            self._hw_warn_logged = False
        positions_rad_by_name = {}
        for i, joint_name in enumerate(SERVO_ORDER):
            d = degs[i] if degs[i] >= 0 else 90.0
            positions_rad_by_name[joint_name] = deg_to_rad(joint_name, d)
        grip_rad = positions_rad_by_name["grip_joint"]
        names_out = list(JOINT_NAMES)
        positions_out = []
        for name in JOINT_NAMES:
            if name in positions_rad_by_name:
                positions_out.append(positions_rad_by_name[name])
            else:
                mult = MIMIC_MAP[name][1]
                positions_out.append(mult * grip_rad)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names_out
        msg.position = positions_out
        self._pub.publish(msg)

        self._state_log_count += 1
        if self._state_log_count == 1 or self._state_log_count % 40 == 0:
            rad_fmt = ", ".join("%.4f" % p for p in positions_out[:6])
            deg_fmt = ", ".join("%.1f°" % d for d in degs)
            self._flog.debug(
                "STATE[%d]: hw_deg=[%s] pub_rad=[%s]"
                % (self._state_log_count, deg_fmt, rad_fmt)
            )

    def destroy_node(self) -> None:
        try:
            self._zmq_client.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> int:
    host = os.environ.get("ZMQ_HOST", "192.168.31.142")
    port = int(os.environ.get("ZMQ_PORT", "5555"))
    ensure_zmq_connection(host, port, log_fn=print)

    rclpy.init(args=args)
    node = OpusX3PlusZMQBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
