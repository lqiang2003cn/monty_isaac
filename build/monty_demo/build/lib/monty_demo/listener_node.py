#!/usr/bin/env python3
"""Demo subscriber node: subscribes to 'chatter' and logs messages."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ListenerNode(Node):
    def __init__(self):
        super().__init__("listener")
        self.subscription_ = self.create_subscription(
            String, "chatter", self.chatter_callback, 10
        )

    def chatter_callback(self, msg):
        self.get_logger().info(f"I heard: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
