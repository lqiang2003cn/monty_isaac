#!/usr/bin/env python3
# coding: utf-8
"""
ROS2 bridge: subscribes to /x3plus/joint_commands, publishes /x3plus/joint_states.
Forwards commands to the real robot via ZMQ (radians <-> degrees conversion).

Same topic interface as isaac_comp's x3plus_isaac_arm_demo.py so ros2_control
joint_trajectory_controller works unchanged. Run with ROS2 environment sourced.

  python3 zmq_bridge_node.py
"""

import math
import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from lqtech_zmq_client import LqtechZMQClient

# Topic names (must match ros2_control_topic.xacro / JointStateTopicSystem)
JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

# Joint order for the 6 DOF (arm + grip); must match ZMQ servo order 1-6
JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "grip_joint",
]

# Radians (ROS, 0 = center) <-> degrees (hardware)
#
# Joints 2-6: serial uses 0-180° over pulse 900-3100.
#   rad_to_deg:  hw = rad * 180/pi + 90   => -pi/2 -> 0°, 0 -> 90°, pi/2 -> 180°
#   deg_to_rad:  rad = (hw - 90) * pi/180
#
# Joint 1:  serial uses 0-360° over the SAME pulse 900-3100 (artificial 2x labeling).
#   rad_to_deg_joint1:  hw = (rad + pi/2) * 360/pi  => -pi/2 -> 0°, 0 -> 180°, pi/2 -> 360°
#   deg_to_rad_joint1:  rad = hw * pi/360 - pi/2

def rad_to_deg(rad: float) -> int:
    return int(round(rad * 180.0 / math.pi + 90.0))


def rad_to_deg_joint1(rad: float) -> int:
    deg = (rad + math.pi / 2.0) * 360.0 / math.pi
    return int(round(max(0.0, min(360.0, deg))))


def deg_to_rad(deg: float) -> float:
    return (float(deg) - 90.0) * math.pi / 180.0


def deg_to_rad_joint1(deg: float) -> float:
    d = max(0.0, min(360.0, float(deg)))
    return d * math.pi / 360.0 - math.pi / 2.0


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

        # Publish joint state at ~50 Hz to match controller expectations
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
        # Skip command if any position is NaN (e.g. from controller before first state)
        if any(math.isnan(r) or math.isinf(r) for r in positions_rad):
            return
        degs = [
            rad_to_deg_joint1(positions_rad[0]),
            rad_to_deg(positions_rad[1]),
            rad_to_deg(positions_rad[2]),
            rad_to_deg(positions_rad[3]),
            rad_to_deg(positions_rad[4]),
            rad_to_deg(positions_rad[5]),
        ]
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
        rads = [
            deg_to_rad_joint1(degs[0]),
            deg_to_rad(degs[1]),
            deg_to_rad(degs[2]),
            deg_to_rad(degs[3]),
            deg_to_rad(degs[4]),
            deg_to_rad(degs[5]),
        ]
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
