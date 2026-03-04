#!/usr/bin/env python3
"""
X3plus 5-DOF planning wrapper for MoveIt 2.

Exposes three ROS 2 services that accept progressively richer goal types:

  1. ~/plan_position        — (x, y, z) only; orientation is free
  2. ~/plan_position_orient — (x, y, z, pitch, roll); yaw derived from position
  3. ~/plan_pose            — full 6D Pose; rejects only if truly impossible

Each service validates the target against the workspace boundary, computes an
analytical IK solution when applicable, and delegates collision-free path
planning to MoveIt's move_group via the MoveGroupAction interface.

Works with both Isaac Sim and real robot (same controller stack).
"""

import math
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from std_srvs.srv import Trigger
from rcl_interfaces.msg import ParameterDescriptor

# ---------------------------------------------------------------------------
# Kinematic constants (from URDF)
# ---------------------------------------------------------------------------

L1 = 0.0829       # arm_link2 → arm_link3
L2 = 0.0829       # arm_link3 → arm_link4
L3 = 0.17455      # arm_link4 → wrist point (Y-component)
L3X = 0.00215     # arm_link4 → wrist point (X-component, perpendicular offset)

BASE_XY = 0.09825                      # joint1 X-offset from base_link
BASE_Z = 0.076 + 0.102 + 0.0405       # base_footprint + joint1_z + joint2_z

JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5",
]

JOINT_LIMITS = {
    "arm_joint1": (-1.5708, 1.5708),
    "arm_joint2": (-1.5708, 1.5708),
    "arm_joint3": (-1.5708, 1.5708),
    "arm_joint4": (-1.5708, 1.5708),
    "arm_joint5": (-1.5708, 3.14159),
}

PLANNING_GROUP = "arm"
PLANNING_FRAME = "base_link"
EE_LINK = "arm_link5"


class PlanResult(Enum):
    SUCCESS = auto()
    OUT_OF_WORKSPACE = auto()
    IK_FAILED = auto()
    YAW_IMPOSSIBLE = auto()
    PLANNING_FAILED = auto()


# ---------------------------------------------------------------------------
# Workspace validation
# ---------------------------------------------------------------------------

def is_in_workspace(x: float, y: float, z: float, margin: float = 0.01) -> bool:
    """Fast bounding check for the wrist point (arm_link5 origin).

    Joint 1 pivots at (BASE_XY, 0) and covers ±90°, so the target must be
    in front of that pivot (dx >= 0).  Radial and height bounds come from
    the planar 3-link chain reach.
    """
    dx = x - BASE_XY
    if dx < -margin:
        return False
    r = math.sqrt(dx * dx + y * y)
    max_reach = L1 + L2 + L3 + L3X
    if r > max_reach - margin:
        return False
    if r < 0.01:
        return False
    if z < 0.0 or z > BASE_Z + max_reach - margin:
        return False
    return True


# ---------------------------------------------------------------------------
# Analytical IK for the 5-DOF arm
# ---------------------------------------------------------------------------

def analytical_ik_5d(x, y, z, pitch, roll):
    """
    Closed-form IK for position (x,y,z) + gripper pitch + gripper roll.

    The target (x,y,z) is the wrist point (arm_link5 origin).

    pitch: angle of the gripper relative to horizontal in the arm's vertical
           plane.  0 = horizontal forward, -pi/2 = straight down, +pi/2 = up.
    roll:  rotation of the gripper around its own axis (= q5).

    Returns (q1, q2, q3, q4, q5) or None if unreachable.

    Planar FK (verified against full transformation matrices):
      r = BASE_XY - L1*sin(q2) - L2*sin(q2+q3) - L3*sin(phi)
      z = BASE_Z  + L1*cos(q2) + L2*cos(q2+q3) + L3*cos(phi)
    where phi = q2+q3+q4 is measured from vertical (0=up, -pi/2=forward, -pi=down)
    and pitch = phi + pi/2, so phi = pitch - pi/2.
    """
    # Joint 1 pivots at (BASE_XY, 0) in base_link and rotates around -Z.
    dx = x - BASE_XY
    q1 = -math.atan2(y, dx)
    if abs(q1) > math.pi / 2:
        return None

    r = math.sqrt(dx * dx + y * y)
    h = z - BASE_Z
    phi = pitch - math.pi / 2  # convert user pitch to angle-from-vertical

    # 2R sub-problem: subtract link-3 contribution (including perpendicular
    # offset L3X from joint-5 URDF origin xyz=(-0.00215, -0.17455, 0)).
    # Planar FK: r = ... - L3*sin(phi) - L3X*cos(phi)
    #            h = ... + L3*cos(phi) - L3X*sin(phi)
    tx = h - L3 * math.cos(phi) + L3X * math.sin(phi)
    ty = -(r + L3 * math.sin(phi) + L3X * math.cos(phi))

    d_sq = tx * tx + ty * ty
    cos_q3 = (d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(cos_q3) > 1.0 + 1e-6:
        return None

    for sign in (1.0, -1.0):
        q3 = sign * math.acos(max(-1.0, min(1.0, cos_q3)))
        alpha = math.atan2(ty, tx)
        beta = math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
        q2 = alpha - beta
        q4 = phi - q2 - q3

        if (_in_limits("arm_joint2", q2) and
                _in_limits("arm_joint3", q3) and
                _in_limits("arm_joint4", q4)):
            q5 = roll
            if not _in_limits("arm_joint5", q5):
                continue
            return (q1, q2, q3, q4, q5)

    return None


def _in_limits(name, val):
    lo, hi = JOINT_LIMITS[name]
    return lo - 1e-6 <= val <= hi + 1e-6


def _best_pitch_for_position(x, y, z):
    """Try a range of pitch angles and return the first that has a valid IK."""
    preferred = [-math.pi / 2, 0.0, -math.pi / 4, math.pi / 4, math.pi / 2]
    for p in preferred:
        result = analytical_ik_5d(x, y, z, p, 0.0)
        if result is not None:
            return p
    for p in np.linspace(-math.pi / 2, math.pi / 2, 36):
        result = analytical_ik_5d(x, y, z, float(p), 0.0)
        if result is not None:
            return float(p)
    return None


def _quaternion_to_rpy(qx, qy, qz, qw):
    """Convert quaternion to roll, pitch, yaw (intrinsic XYZ / extrinsic ZYX)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class X3plus5DofPlanner(Node):
    def __init__(self):
        super().__init__("x3plus_5dof_planner")

        self._action_cb_group = MutuallyExclusiveCallbackGroup()
        self._move_group_client = ActionClient(
            self, MoveGroup, "move_action",
            callback_group=self._action_cb_group,
        )

        # Service: plan to position (x, y, z)
        # Call with parameters: target_x, target_y, target_z
        self.create_service(
            Trigger, "~/plan_position", self._plan_position_cb
        )

        # Service: plan to position + orientation (x, y, z, pitch, roll)
        # Call with parameters: target_x, y, z, target_pitch, target_roll
        self.create_service(
            Trigger, "~/plan_position_orient", self._plan_position_orient_cb
        )

        # Service: plan to full pose
        # Call with parameters: target_x..z, target_qx..qw
        self.create_service(
            Trigger, "~/plan_pose", self._plan_pose_cb
        )

        # Declare parameters for goal specification
        self.declare_parameter("target_x", 0.2)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 0.2)
        self.declare_parameter("target_pitch", 0.0)
        self.declare_parameter("target_roll", 0.0)
        self.declare_parameter("target_qx", 0.0)
        self.declare_parameter("target_qy", 0.0)
        self.declare_parameter("target_qz", 0.0)
        self.declare_parameter("target_qw", 1.0)
        self.declare_parameter("execute", True,
                               ParameterDescriptor(description="Execute after planning"))

        self.get_logger().info("X3plus 5-DOF planner ready. Services: "
                               "~/plan_position, ~/plan_position_orient, ~/plan_pose")

    # ── Service callbacks ──

    def _plan_position_cb(self, request, response):
        """Plan to (x, y, z) with orientation free."""
        x = self.get_parameter("target_x").value
        y = self.get_parameter("target_y").value
        z = self.get_parameter("target_z").value
        execute = self.get_parameter("execute").value

        self.get_logger().info(f"plan_position: ({x:.3f}, {y:.3f}, {z:.3f})")

        if not is_in_workspace(x, y, z):
            response.success = False
            response.message = f"Target ({x:.3f},{y:.3f},{z:.3f}) is outside workspace"
            return response

        pitch = _best_pitch_for_position(x, y, z)
        if pitch is None:
            response.success = False
            response.message = "No valid IK found for any orientation at this position"
            return response

        joints = analytical_ik_5d(x, y, z, pitch, 0.0)
        if joints is None:
            response.success = False
            response.message = "Analytical IK failed"
            return response

        result = self._plan_joint_target(list(joints), execute)
        response.success = (result == PlanResult.SUCCESS)
        response.message = result.name
        return response

    def _plan_position_orient_cb(self, request, response):
        """Plan to (x, y, z) + pitch + roll."""
        x = self.get_parameter("target_x").value
        y = self.get_parameter("target_y").value
        z = self.get_parameter("target_z").value
        pitch = self.get_parameter("target_pitch").value
        roll = self.get_parameter("target_roll").value
        execute = self.get_parameter("execute").value

        self.get_logger().info(
            f"plan_position_orient: ({x:.3f},{y:.3f},{z:.3f}) "
            f"pitch={math.degrees(pitch):.1f}° roll={math.degrees(roll):.1f}°"
        )

        if not is_in_workspace(x, y, z):
            response.success = False
            response.message = f"Target ({x:.3f},{y:.3f},{z:.3f}) is outside workspace"
            return response

        joints = analytical_ik_5d(x, y, z, pitch, roll)
        if joints is None:
            response.success = False
            response.message = (
                f"IK failed: pitch={math.degrees(pitch):.1f}° not achievable at this position. "
                f"Try a different pitch or use plan_position for auto orientation."
            )
            return response

        result = self._plan_joint_target(list(joints), execute)
        response.success = (result == PlanResult.SUCCESS)
        response.message = result.name
        return response

    def _plan_pose_cb(self, request, response):
        """Plan to a full 6D pose. Decomposes into achievable 5-DOF components."""
        x = self.get_parameter("target_x").value
        y = self.get_parameter("target_y").value
        z = self.get_parameter("target_z").value
        qx = self.get_parameter("target_qx").value
        qy = self.get_parameter("target_qy").value
        qz = self.get_parameter("target_qz").value
        qw = self.get_parameter("target_qw").value
        execute = self.get_parameter("execute").value

        self.get_logger().info(
            f"plan_pose: ({x:.3f},{y:.3f},{z:.3f}) "
            f"q=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})"
        )

        if not is_in_workspace(x, y, z):
            response.success = False
            response.message = f"Target ({x:.3f},{y:.3f},{z:.3f}) is outside workspace"
            return response

        roll_d, pitch_d, yaw_d = _quaternion_to_rpy(qx, qy, qz, qw)
        arm_yaw = math.atan2(y, x)

        yaw_error = abs(yaw_d - arm_yaw)
        if yaw_error > math.pi:
            yaw_error = 2.0 * math.pi - yaw_error

        YAW_TOL = math.radians(10.0)

        if yaw_error > math.pi / 2 + YAW_TOL:
            response.success = False
            response.message = (
                f"Requested yaw ({math.degrees(yaw_d):.1f}°) differs from arm azimuth "
                f"({math.degrees(arm_yaw):.1f}°) by {math.degrees(yaw_error):.1f}°, "
                f"which exceeds joint1 range (±90°). Mobile base rotation required."
            )
            return response

        joints = analytical_ik_5d(x, y, z, pitch_d, roll_d)
        if joints is not None:
            result = self._plan_joint_target(list(joints), execute)
            response.success = (result == PlanResult.SUCCESS)
            response.message = result.name
            return response

        # Fallback: try nearby pitch values
        self.get_logger().warn(
            f"Exact pitch={math.degrees(pitch_d):.1f}° not achievable, "
            "searching nearest achievable orientation..."
        )
        best_joints = None
        best_err = float("inf")
        for p in np.linspace(-math.pi / 2, math.pi / 2, 72):
            j = analytical_ik_5d(x, y, z, float(p), roll_d)
            if j is not None:
                err = abs(float(p) - pitch_d)
                if err < best_err:
                    best_err = err
                    best_joints = j

        if best_joints is None:
            response.success = False
            response.message = "No valid IK solution found for any orientation at this position"
            return response

        self.get_logger().info(
            f"Using nearest achievable pitch (error: {math.degrees(best_err):.1f}°)"
        )
        result = self._plan_joint_target(list(best_joints), execute)
        response.success = (result == PlanResult.SUCCESS)
        response.message = result.name
        return response

    # ── MoveIt interface ──

    def _plan_joint_target(self, joint_values, execute=True):
        """Send a joint-space goal to move_group and optionally execute."""
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_group action server not available")
            return PlanResult.PLANNING_FAILED

        goal_msg = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0

        constraints = Constraints()
        for name, val in zip(JOINT_NAMES, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)
        goal_msg.request = req

        planning_opts = PlanningOptions()
        planning_opts.plan_only = not execute
        goal_msg.planning_options = planning_opts

        future = self._move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal rejected by move_group")
            return PlanResult.PLANNING_FAILED

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

        result = result_future.result()
        if result is None:
            self.get_logger().error("No result from move_group")
            return PlanResult.PLANNING_FAILED

        error_code = result.result.error_code.val
        if error_code == 1:  # MoveItErrorCodes.SUCCESS
            action = "executed" if execute else "planned"
            self.get_logger().info(f"Motion {action} successfully")
            return PlanResult.SUCCESS
        else:
            self.get_logger().error(f"MoveIt error code: {error_code}")
            return PlanResult.PLANNING_FAILED


def main(args=None):
    rclpy.init(args=args)
    node = X3plus5DofPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
