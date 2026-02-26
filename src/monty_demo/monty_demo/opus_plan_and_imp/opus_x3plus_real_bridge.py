#!/usr/bin/env python3
"""
Real X3plus robot bridge: subscribes to /x3plus/joint_commands, sends position
commands to hardware via Rosmaster_Lib; reads positions and publishes
/x3plus/joint_states. Position-only. Run with ros2_control bringup (mode:=real).
"""

import sys

try:
    from Rosmaster_Lib import Rosmaster
except ImportError as e:
    print("Rosmaster_Lib not found. Install with: pip install Rosmaster_Lib", file=sys.stderr)
    raise

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from monty_demo.opus_plan_and_imp.opus_joint_config import (
    ARM_GRIP_JOINTS,
    JOINT_NAMES,
    MIMIC_MAP,
    rad_to_deg,
    deg_to_rad,
)

JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

# Servo order for set_uart_servo_angle_array / get_uart_servo_angle_array: IDs 1-6
SERVO_ORDER = ARM_GRIP_JOINTS  # [arm_joint1, ..., arm_joint5, grip_joint]


class OpusX3PlusRealBridge(Node):
    def __init__(self) -> None:
        super().__init__("opus_x3plus_real_bridge")
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("state_publish_hz", 20.0)
        self.declare_parameter("servo_run_time_ms", 50)
        serial_port = self.get_parameter("serial_port").get_parameter_value().string_value
        state_hz = self.get_parameter("state_publish_hz").get_parameter_value().double_value
        self._run_time_ms = int(self.get_parameter("servo_run_time_ms").value)

        self._robot = Rosmaster(car_type=1, com=serial_port, delay=0.002, debug=False)
        try:
            self._robot.set_uart_servo_torque(1)
        except Exception as e:
            self.get_logger().warn(f"set_uart_servo_torque(1) failed: {e}")
        self._latest_positions_rad = None  # list of 6 floats in ARM_GRIP_JOINTS order, or None

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
        self._timer = self.create_timer(1.0 / state_hz, self._timer_cb)
        self.get_logger().info(
            f"Real bridge: sub {JOINT_COMMANDS_TOPIC}, pub {JOINT_STATES_TOPIC}, port={serial_port}"
        )

    def _joint_commands_cb(self, msg: JointState) -> None:
        if not msg.name or len(msg.position) < len(msg.name):
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        angle_s = []
        for joint_name in SERVO_ORDER:
            if joint_name in name_to_pos:
                deg = rad_to_deg(joint_name, float(name_to_pos[joint_name]))
                angle_s.append(deg)
            else:
                angle_s.append(90.0)
        if len(angle_s) != 6:
            return
        try:
            self._robot.set_uart_servo_angle_array(angle_s=angle_s, run_time=self._run_time_ms)
        except Exception as e:
            self.get_logger().warn(f"set_uart_servo_angle_array failed: {e}")

    def _timer_cb(self) -> None:
        try:
            raw = self._robot.get_uart_servo_angle_array()
        except Exception as e:
            self.get_logger().warn(f"get_uart_servo_angle_array failed: {e}")
            return
        if not raw or len(raw) != 6:
            return
        positions_rad_by_name = {}
        for i, joint_name in enumerate(SERVO_ORDER):
            deg = raw[i] if raw[i] >= 0 else 90.0
            positions_rad_by_name[joint_name] = deg_to_rad(joint_name, deg)
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

    def shutdown(self) -> None:
        try:
            self._robot.set_uart_servo_torque(0)
        except Exception:
            pass


def main(args=None) -> int:
    rclpy.init(args=args)
    node = OpusX3PlusRealBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
