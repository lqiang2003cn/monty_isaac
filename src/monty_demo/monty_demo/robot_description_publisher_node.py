#!/usr/bin/env python3
"""Publish robot_description from a parameter to a topic for controller_manager."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from std_msgs.msg import String


def main(args=None):
    rclpy.init(args=args)
    node = Node("robot_description_publisher")
    node.declare_parameter("robot_description", "")
    desc = node.get_parameter("robot_description").get_parameter_value().string_value
    if not desc:
        node.get_logger().error("Parameter 'robot_description' is empty")
        rclpy.shutdown()
        return 1
    # Both QoS profiles so late subscribers (controller_manager) get the message
    qos_tl = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(String, "robot_description_full", qos_tl)
    msg = String()
    msg.data = desc
    # Publish several times so controller_manager gets it even if it subscribes late
    for _ in range(10):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.2)
    node.get_logger().info("Published robot_description to robot_description_full")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    exit(main())
