#!/usr/bin/env python3
"""
One-shot joint state checker for the X3plus arm.

Reads the current /joint_states, computes FK to the wrist point, and compares
against the expected home position (all joints zero).

Usage:
  ros2 run monty_demo x3plus_joint_check
  ros2 run monty_demo x3plus_joint_check --expected 0.0 0.0 0.0 0.0 0.0
"""

import argparse
import math
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from monty_demo.opus_plan_and_imp.opus_joint_config import INIT_ARM_POSITIONS

ARM_JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5"]

HOME = list(INIT_ARM_POSITIONS)

L1 = 0.0829
L2 = 0.0829
L3 = 0.17455
L3X = 0.00215
BASE_XY = 0.09825
BASE_Z = 0.076 + 0.102 + 0.0405


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _tf(xyz, rpy=(0, 0, 0)):
    c, s = np.cos(rpy[1]), np.sin(rpy[1])
    ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    T = _rot_z(rpy[2]) @ ry @ _rot_x(rpy[0])
    T[:3, 3] = xyz
    return T


def _joint_tf(xyz, rpy, axis, q):
    T_origin = _tf(xyz, rpy)
    ax = np.array(axis, dtype=float)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    return T_origin @ T_rot


_T_BASE = _tf([0, 0, 0.076])
_CHAIN = [
    ([0.09825, 0, 0.102],     [0, 0, 0],       [0, 0, -1]),
    ([0, 0, 0.0405],          [-1.5708, 0, 0],  [0, 0, -1]),
    ([0, -0.0829, 0],         [0, 0, 0],        [0, 0, -1]),
    ([0, -0.0829, 0],         [0, 0, 0],        [0, 0, -1]),
    ([-0.00215, -0.17455, 0], [1.5708, 0, 0],   [0, 0, 1]),
]


def fk_wrist(qs):
    T = _T_BASE.copy()
    for i, (xyz, rpy, axis) in enumerate(_CHAIN):
        T = T @ _joint_tf(xyz, rpy, axis, qs[i])
    return T[:3, 3]


class JointCheck(Node):
    def __init__(self, expected_joints):
        super().__init__("x3plus_joint_check")
        self._expected = expected_joints
        self._done = False
        self.create_subscription(JointState, "/joint_states", self._cb, 1)
        self._timeout = self.create_timer(5.0, self._timeout_cb)

    def _timeout_cb(self):
        self.get_logger().error("No /joint_states received within 5s")
        self._done = True

    def _cb(self, msg: JointState):
        if self._done:
            return
        self._done = True
        self._timeout.cancel()

        name_to_pos = dict(zip(msg.name, msg.position))
        actual = []
        for j in ARM_JOINTS:
            if j not in name_to_pos:
                self.get_logger().error(f"Joint '{j}' not found in /joint_states")
                return
            actual.append(name_to_pos[j])

        actual_xyz = fk_wrist(actual)
        expected_xyz = fk_wrist(self._expected)

        print("\n" + "=" * 60)
        print("  X3plus Joint State Check")
        print("=" * 60)

        print(f"\n{'Joint':<14} {'Actual (rad)':>12} {'Expected (rad)':>14} {'Error (deg)':>12}  Status")
        print("-" * 68)
        all_ok = True
        for i, j in enumerate(ARM_JOINTS):
            err_rad = actual[i] - self._expected[i]
            err_deg = math.degrees(err_rad)
            ok = abs(err_deg) < 3.0
            status = "OK" if ok else "DRIFT"
            if not ok:
                all_ok = False
            print(f"  {j:<12} {actual[i]:>12.4f} {self._expected[i]:>14.4f} {err_deg:>+11.2f}°  {status}")

        pos_err = np.linalg.norm(actual_xyz - expected_xyz)
        print(f"\n  Wrist FK  actual:   ({actual_xyz[0]:.4f}, {actual_xyz[1]:.4f}, {actual_xyz[2]:.4f}) m")
        print(f"  Wrist FK  expected: ({expected_xyz[0]:.4f}, {expected_xyz[1]:.4f}, {expected_xyz[2]:.4f}) m")
        print(f"  Position error:     {pos_err * 1000:.1f} mm")

        if all_ok and pos_err < 0.005:
            print(f"\n  RESULT: ALL JOINTS MATCH (< 3° each, {pos_err*1000:.1f} mm wrist error)")
        else:
            print(f"\n  RESULT: MISMATCH DETECTED")
            if not all_ok:
                drifted = [ARM_JOINTS[i] for i in range(5)
                           if abs(math.degrees(actual[i] - self._expected[i])) >= 3.0]
                print(f"  Joints with > 3° drift: {', '.join(drifted)}")
            if pos_err >= 0.005:
                print(f"  Wrist position error: {pos_err*1000:.1f} mm (> 5 mm threshold)")

        print("=" * 60 + "\n")

    @property
    def done(self):
        return self._done


def main():
    parser = argparse.ArgumentParser(description="Check X3plus joint positions")
    parser.add_argument(
        "--expected", nargs=5, type=float, default=HOME,
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="Expected joint positions in radians (default: all zeros = home)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = JointCheck(args.expected)
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
