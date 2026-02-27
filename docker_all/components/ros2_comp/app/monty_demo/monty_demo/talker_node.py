#!/usr/bin/env python3
"""Demo publisher node: publishes std_msgs/String on topic 'chatter'."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TalkerNode(Node):
    def __init__(self):
        super().__init__("talker")
        self.publisher_ = self.create_publisher(String, "chatter", 10)
        self.timer_ = self.create_timer(1.0, self.timer_callback)
        self.count_ = 0

    def timer_callback(self):
        msg = String()
        msg.data = f"Hello ROS2 from monty_demo #{self.count_}"
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: {msg.data}")
        self.count_ += 1


def main(args=None):
    rclpy.init(args=args)
    node = TalkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
