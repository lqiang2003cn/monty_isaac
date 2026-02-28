#!/usr/bin/env python3
# coding: utf-8
"""
Move arm_joint1 to -pi/2 only. All other joints stay at their current positions.

Usage (from inside ros2_comp container):
  source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash
  python3 /scripts/send_first_joint_trajectory.py
"""

import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "grip_joint",
]
JOINT_STATES_TOPIC = "/x3plus/joint_states"
ACTION_NAME = "/joint_trajectory_controller/follow_joint_trajectory"
TIMEOUT_GET_STATE_SEC = 10.0
TIME_POINT_SEC = 5.0  # time to reach -pi/2
ARM_JOINT1_TARGET = -math.pi / 2


class SendFirstJointTrajectory(Node):
    def __init__(self):
        super().__init__("send_first_joint_trajectory")
        self._current_positions = None
        self._state_received = False
        self._sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self._joint_state_cb,
            1,
        )
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        if self._state_received or not msg.name or len(msg.position) < len(msg.name):
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        positions = []
        for name in JOINT_NAMES:
            if name not in name_to_pos:
                return
            positions.append(float(name_to_pos[name]))
        if len(positions) == 6 and not any(math.isnan(p) or math.isinf(p) for p in positions):
            self._current_positions = positions
            self._state_received = True

    def wait_for_current_state(self, timeout_sec: float = TIMEOUT_GET_STATE_SEC) -> bool:
        import time
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._current_positions is not None:
                return True
        return False

    def send_trajectory(self) -> bool:
        if self._current_positions is None:
            self.get_logger().error("No current joint state received from %s" % JOINT_STATES_TOPIC)
            return False

        self.get_logger().info(
            "Current positions (rad): %s" % [round(p, 4) for p in self._current_positions]
        )
        target_pos = [ARM_JOINT1_TARGET] + list(self._current_positions[1:])

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = list(JOINT_NAMES)
        goal_msg.trajectory.points = [
            JointTrajectoryPoint(
                positions=target_pos,
                time_from_start=Duration(sec=int(TIME_POINT_SEC), nanosec=0),
            ),
        ]

        self.get_logger().info("Waiting for action server %s ..." % ACTION_NAME)
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available")
            return False

        self.get_logger().info("Sending goal: joint1 -> -pi/2, others unchanged")
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            self.get_logger().error("Send goal timed out")
            return False
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        self.get_logger().info("Goal accepted, waiting for result...")
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=TIME_POINT_SEC + 5.0)
        if not result_future.done():
            self.get_logger().warn("Result wait timed out (trajectory may still be running)")
            return True
        result = result_future.result().result
        if result is not None and hasattr(result, "error_code"):
            self.get_logger().info("Result error_code: %s" % result.error_code)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = SendFirstJointTrajectory()
    try:
        if not node.wait_for_current_state():
            node.get_logger().error("Timeout waiting for joint state from %s" % JOINT_STATES_TOPIC)
            return 1
        if not node.send_trajectory():
            return 1
        node.get_logger().info("Done.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
