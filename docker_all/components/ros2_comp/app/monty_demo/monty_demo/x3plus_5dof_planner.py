#!/usr/bin/env python3
"""
X3plus 5-DOF planning wrapper for MoveIt 2.

Exposes ROS 2 services that accept progressively richer goal types:

  1. ~/plan_position        — (x, y, z) only; orientation is free
  2. ~/plan_position_orient — (x, y, z, pitch, roll); yaw derived from position
  3. ~/plan_pose            — full 6D Pose; rejects only if truly impossible
  4. ~/plan_straight_line   — straight-line Cartesian path from current pose
  5. ~/go_home              — return to all-zeros joint configuration
  6. ~/set_gripper          — move grip_joint to target_grip (arm stays fixed)

Services 1–3 validate the target against the workspace boundary, compute an
analytical IK solution when applicable, and delegate collision-free path
planning to MoveIt's move_group via the MoveGroupAction interface.

Service 4 bypasses MoveIt: it interpolates linearly in Cartesian space,
solves analytical IK at each waypoint, and sends the resulting joint
trajectory directly to the controller via FollowJointTrajectory.

Works with both Isaac Sim and real robot (same controller stack).
"""

import math
import time as _time
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rcl_interfaces.msg import ParameterDescriptor

from monty_demo.opus_plan_and_imp.log_utils import LOG_DIR, make_file_logger

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


def analytical_ik_5d_all(x, y, z, pitch, roll):
    """Like analytical_ik_5d but returns ALL valid (q1..q5) solutions.

    Used by the straight-line planner to pick the solution closest to the
    previous waypoint, preventing elbow-configuration flips that cause jerky
    motion.
    """
    dx = x - BASE_XY
    if dx * dx + y * y < 1e-8:
        return []
    q1 = -math.atan2(y, dx)
    if abs(q1) > math.pi / 2:
        return []

    r = math.sqrt(dx * dx + y * y)
    h = z - BASE_Z
    phi = pitch - math.pi / 2

    tx = h - L3 * math.cos(phi) + L3X * math.sin(phi)
    ty = -(r + L3 * math.sin(phi) + L3X * math.cos(phi))

    d_sq = tx * tx + ty * ty
    cos_q3 = (d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(cos_q3) > 1.0 + 1e-6:
        return []

    solutions = []
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
            solutions.append((q1, q2, q3, q4, q5))
    return solutions


def forward_kinematics_5d(q1, q2, q3, q4, q5):
    """Compute (x, y, z, pitch, roll) from joint angles.

    Inverse of analytical_ik_5d — uses the same kinematic model so FK→IK
    round-trips are exact (within floating-point precision).
    """
    phi = q2 + q3 + q4
    r = -(L1 * math.sin(q2) + L2 * math.sin(q2 + q3)) \
        - L3 * math.sin(phi) - L3X * math.cos(phi)
    h = L1 * math.cos(q2) + L2 * math.cos(q2 + q3) \
        + L3 * math.cos(phi) - L3X * math.sin(phi)
    x = BASE_XY + r * math.cos(q1)
    y = -r * math.sin(q1)
    z = BASE_Z + h
    pitch = phi + math.pi / 2
    roll = q5
    return x, y, z, pitch, roll


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
    _JOINT_STATES_TOPIC = "/x3plus/joint_states"
    _FJT_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
    _CONTROLLER_JOINTS = JOINT_NAMES + ["grip_joint"]

    _MAX_JOINT_VEL = 0.4   # rad/s — conservative limit for the real servos

    def __init__(self):
        super().__init__("x3plus_5dof_planner")

        self._action_cb_group = MutuallyExclusiveCallbackGroup()
        self._fjt_cb_group = MutuallyExclusiveCallbackGroup()
        self._service_cb_group = ReentrantCallbackGroup()

        self._move_group_client = ActionClient(
            self, MoveGroup, "move_action",
            callback_group=self._action_cb_group,
        )
        self._fjt_client = ActionClient(
            self, FollowJointTrajectory, self._FJT_ACTION,
            callback_group=self._fjt_cb_group,
        )

        # Joint state tracking for FK / straight-line planning.
        # Subscribe to both the bridge topic and the standard topic so the
        # straight-line planner works in all modes (real robot, sim, topic-based).
        self._current_arm_positions = None   # 5 arm joints (from hardware)
        self._current_grip_position = 0.0
        self._virtual_arm_positions = None  # overrides hardware state in dry-run
        self._virtual_grip_position = None
        self.create_subscription(
            JointState, self._JOINT_STATES_TOPIC,
            self._joint_state_cb, 1,
        )
        self.create_subscription(
            JointState, "/joint_states",
            self._joint_state_cb, 1,
        )

        self.create_service(
            Trigger, "~/plan_position", self._plan_position_cb,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger, "~/plan_position_orient", self._plan_position_orient_cb,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger, "~/plan_pose", self._plan_pose_cb,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger, "~/go_home", self._go_home_cb,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger, "~/plan_straight_line", self._plan_straight_line_cb,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger, "~/set_gripper", self._set_gripper_cb,
            callback_group=self._service_cb_group,
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
        self.declare_parameter("cartesian_speed", 0.03,
                               ParameterDescriptor(description="Straight-line speed (m/s)"))
        self.declare_parameter("cartesian_step", 0.01,
                               ParameterDescriptor(description="Waypoint spacing (m)"))
        self.declare_parameter("go_home_on_start", True,
                               ParameterDescriptor(description="Automatically go home when node starts"))
        self.declare_parameter("target_grip", -0.77,
                               ParameterDescriptor(description="Target grip_joint position for set_gripper"))
        self.declare_parameter("dry_run", False,
                               ParameterDescriptor(
                                   description="Safety lockout: when True, no trajectory "
                                               "is executed on the real robot"))

        _log_path = f"{LOG_DIR}/x3plus_planner.log"
        self._flog = make_file_logger("x3plus_planner", _log_path)
        self._flog.info("=" * 60)
        self._flog.info("X3plus 5-DOF planner node starting")
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.get_logger().info(
            "Planner ready (log: %s). Services: ~/plan_position, "
            "~/plan_position_orient, ~/plan_pose, ~/go_home, "
            "~/plan_straight_line, ~/set_gripper" % _log_path
        )

        self._js_log_count = 0

        self._startup_go_home_done = False
        if self.get_parameter("go_home_on_start").value:
            self.get_logger().info("go_home_on_start enabled — will go home once MoveIt is ready")
            self._startup_go_home_timer = self.create_timer(
                3.0, self._startup_go_home_cb,
                callback_group=self._service_cb_group,
            )

    # ── Parameter safety ──

    @property
    def _is_dry_run(self):
        return self.get_parameter("dry_run").value

    def _on_params_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name in ("execute", "dry_run"):
                self._flog.info(f"PARAM CHANGE: {p.name} = {p.value}")
        return SetParametersResult(successful=True)

    # ── Service callbacks ──

    def _format_result(self, result):
        """Return result name, appending the MoveIt error code on failure."""
        msg = result.name
        if result == PlanResult.PLANNING_FAILED:
            code = getattr(self, "_last_moveit_error", None)
            if code is not None:
                msg += f" (MoveIt error_code={code})"
        return msg

    def _apply_dry_run(self, execute, caller):
        """Defense-in-depth: override execute at the service-callback level."""
        if self._is_dry_run and execute:
            self._flog.warning(
                f"{caller}: dry_run OVERRIDE execute=True → False"
            )
            self.get_logger().warn(
                f"{caller}: dry_run active — forcing plan_only"
            )
            return False
        return execute

    def _plan_position_cb(self, request, response):
        """Plan to (x, y, z) with orientation free."""
        x = self.get_parameter("target_x").value
        y = self.get_parameter("target_y").value
        z = self.get_parameter("target_z").value
        execute = self._apply_dry_run(
            self.get_parameter("execute").value, "plan_position"
        )

        self._flog.info(
            f"plan_position: target=({x:.3f},{y:.3f},{z:.3f}) "
            f"execute={execute} dry_run={self._is_dry_run}"
        )
        self.get_logger().info("plan_position requested")

        if not is_in_workspace(x, y, z):
            self._flog.debug(f"plan_position: REJECTED — outside workspace")
            response.success = False
            response.message = f"Target ({x:.3f},{y:.3f},{z:.3f}) is outside workspace"
            return response

        pitch = _best_pitch_for_position(x, y, z)
        if pitch is None:
            self._flog.debug(f"plan_position: REJECTED — no valid pitch")
            response.success = False
            response.message = "No valid IK found for any orientation at this position"
            return response

        joints = analytical_ik_5d(x, y, z, pitch, 0.0)
        if joints is None:
            self._flog.debug(f"plan_position: REJECTED — IK failed at pitch={math.degrees(pitch):.1f}°")
            response.success = False
            response.message = "Analytical IK failed"
            return response

        jfmt = ", ".join(f"{j:.4f}" for j in joints)
        self._flog.debug(
            f"plan_position: pitch={math.degrees(pitch):.1f}° "
            f"joints=[{jfmt}] execute={execute}"
        )
        result = self._plan_joint_target(list(joints), execute)
        response.success = (result == PlanResult.SUCCESS)
        response.message = self._format_result(result)
        return response

    def _plan_position_orient_cb(self, request, response):
        """Plan to (x, y, z) + pitch + roll."""
        x = self.get_parameter("target_x").value
        y = self.get_parameter("target_y").value
        z = self.get_parameter("target_z").value
        pitch = self.get_parameter("target_pitch").value
        roll = self.get_parameter("target_roll").value
        execute = self._apply_dry_run(
            self.get_parameter("execute").value, "plan_position_orient"
        )

        self._flog.info(
            f"plan_position_orient: target=({x:.3f},{y:.3f},{z:.3f}) "
            f"pitch={math.degrees(pitch):.1f}° roll={math.degrees(roll):.1f}° "
            f"execute={execute} dry_run={self._is_dry_run}"
        )
        self.get_logger().info("plan_position_orient requested")

        if not is_in_workspace(x, y, z):
            self._flog.debug(f"plan_position_orient: REJECTED — outside workspace")
            response.success = False
            response.message = f"Target ({x:.3f},{y:.3f},{z:.3f}) is outside workspace"
            return response

        joints = analytical_ik_5d(x, y, z, pitch, roll)
        if joints is None:
            self._flog.debug(
                f"plan_position_orient: REJECTED — IK failed at "
                f"pitch={math.degrees(pitch):.1f}° roll={math.degrees(roll):.1f}°"
            )
            response.success = False
            response.message = (
                f"IK failed: pitch={math.degrees(pitch):.1f}° not achievable at this position. "
                f"Try a different pitch or use plan_position for auto orientation."
            )
            return response

        jfmt = ", ".join(f"{j:.4f}" for j in joints)
        self._flog.debug(
            f"plan_position_orient: joints=[{jfmt}] execute={execute}"
        )
        result = self._plan_joint_target(list(joints), execute)
        response.success = (result == PlanResult.SUCCESS)
        response.message = self._format_result(result)
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
        execute = self._apply_dry_run(
            self.get_parameter("execute").value, "plan_pose"
        )

        self._flog.info(
            f"plan_pose: target=({x:.3f},{y:.3f},{z:.3f}) "
            f"q=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f}) "
            f"execute={execute} dry_run={self._is_dry_run}"
        )
        self.get_logger().info("plan_pose requested")

        if not is_in_workspace(x, y, z):
            response.success = False
            response.message = f"Target ({x:.3f},{y:.3f},{z:.3f}) is outside workspace"
            return response

        roll_d, pitch_d, yaw_d = _quaternion_to_rpy(qx, qy, qz, qw)
        arm_yaw = math.atan2(y, x)

        self._flog.debug(
            f"plan_pose: RPY=({math.degrees(roll_d):.1f}°, "
            f"{math.degrees(pitch_d):.1f}°, {math.degrees(yaw_d):.1f}°) "
            f"arm_yaw={math.degrees(arm_yaw):.1f}°"
        )

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

        self._flog.info(
            f"plan_pose: using nearest achievable pitch (error: {math.degrees(best_err):.1f}°)"
        )
        result = self._plan_joint_target(list(best_joints), execute)
        response.success = (result == PlanResult.SUCCESS)
        response.message = self._format_result(result)
        return response

    def _go_home_cb(self, request, response):
        """Return to init pose via joint-space planning."""
        from monty_demo.opus_plan_and_imp.opus_joint_config import INIT_ARM_POSITIONS
        execute = self._apply_dry_run(
            self.get_parameter("execute").value, "go_home"
        )
        cur = self._virtual_arm_positions or self._current_arm_positions
        cur_fmt = ", ".join(f"{j:.4f}" for j in cur) if cur else "None"
        self._flog.info(
            f"go_home: current=[{cur_fmt}] target={list(INIT_ARM_POSITIONS)} "
            f"execute={execute} dry_run={self._is_dry_run}"
        )
        self.get_logger().info("go_home requested")
        self._flog.debug(
            f"go_home: current=[{cur_fmt}] target={list(INIT_ARM_POSITIONS)} execute={execute}"
        )
        home = list(INIT_ARM_POSITIONS)
        result = self._plan_joint_target(
            home, execute, vel_scale=0.15, accel_scale=0.15,
        )
        response.success = (result == PlanResult.SUCCESS)
        response.message = self._format_result(result)
        self._flog.debug(f"go_home: result={result.name}")
        return response

    def _startup_go_home_cb(self):
        """Retry timer: wait for joint states + MoveIt, then go home once."""
        if self._startup_go_home_done:
            self._startup_go_home_timer.cancel()
            return

        if self._current_arm_positions is None:
            self._flog.debug("go_home_on_start: waiting for joint states…")
            return

        self._startup_go_home_done = True
        self._startup_go_home_timer.cancel()

        from monty_demo.opus_plan_and_imp.opus_joint_config import INIT_ARM_POSITIONS
        home = list(INIT_ARM_POSITIONS)
        cur_fmt = ", ".join(f"{j:.4f}" for j in self._current_arm_positions)
        self._flog.debug(
            f"go_home_on_start: current_joints=[{cur_fmt}] "
            f"grip={self._current_grip_position:.4f}"
        )
        if self._is_dry_run:
            self._flog.warning(
                "go_home_on_start: SKIPPED — dry_run=True is active"
            )
            self.get_logger().warn(
                "go_home_on_start: SKIPPED (dry_run=True safety lockout)"
            )
            return

        execute = self.get_parameter("execute").value
        self._flog.info(
            f"go_home_on_start: executing go_home → {home} execute={execute}"
        )
        self.get_logger().info("go_home_on_start: executing")
        result = self._plan_joint_target(
            home, execute=execute, vel_scale=0.15, accel_scale=0.15,
        )
        if result == PlanResult.SUCCESS:
            self.get_logger().info("go_home_on_start: completed successfully")
            self._flog.info("go_home_on_start: completed successfully")
        else:
            self.get_logger().warn(f"go_home_on_start: {self._format_result(result)}")
            self._flog.warning(f"go_home_on_start: {self._format_result(result)}")

    def _joint_state_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        positions = []
        for name in JOINT_NAMES:
            if name not in name_to_pos:
                return
            positions.append(float(name_to_pos[name]))
        if any(math.isnan(p) or math.isinf(p) for p in positions):
            return
        first_receipt = self._current_arm_positions is None
        self._current_arm_positions = positions
        self._current_grip_position = float(name_to_pos.get("grip_joint", 0.0))
        self._js_log_count += 1
        if first_receipt or self._js_log_count % 40 == 0:
            fmt = ", ".join(f"{p:.4f}" for p in positions)
            self._flog.debug(
                f"joint_state{'[FIRST]' if first_receipt else ''}: "
                f"arm=[{fmt}] grip={self._current_grip_position:.4f}"
            )

    def _plan_straight_line_cb(self, request, response):
        """Plan and execute a straight-line Cartesian path to the target.

        Validates the full path upfront before execution:
          1. Reject immediately if the target has no valid IK.
          2. Interpolate pitch smoothly from current to target (no mid-path
             discontinuities).
          3. Validate every waypoint — any IK failure rejects the request.
          4. Execute only after the entire trajectory is validated.
        """
        q = self._virtual_arm_positions or self._current_arm_positions
        if q is None:
            response.success = False
            response.message = (
                f"No joint state received on {self._JOINT_STATES_TOPIC} yet"
            )
            return response

        target_x = self.get_parameter("target_x").value
        target_y = self.get_parameter("target_y").value
        target_z = self.get_parameter("target_z").value
        execute = self._apply_dry_run(
            self.get_parameter("execute").value, "plan_straight_line"
        )
        self._flog.info(
            f"plan_straight_line: target=({target_x:.3f},{target_y:.3f},{target_z:.3f}) "
            f"execute={execute} dry_run={self._is_dry_run}"
        )
        speed = self.get_parameter("cartesian_speed").value
        step = self.get_parameter("cartesian_step").value
        cur_x, cur_y, cur_z, cur_pitch, cur_roll = forward_kinematics_5d(*q)

        self._flog.info(
            f"plan_straight_line: ({cur_x:.4f},{cur_y:.4f},{cur_z:.4f}) -> "
            f"({target_x:.4f},{target_y:.4f},{target_z:.4f})"
        )
        qfmt = ", ".join(f"{v:.4f}" for v in q)
        self._flog.debug(
            f"plan_straight_line: cur_joints=[{qfmt}] "
            f"cur_pitch={math.degrees(cur_pitch):.1f}° cur_roll={math.degrees(cur_roll):.1f}° "
            f"speed={speed:.3f}m/s step={step:.3f}m execute={execute}"
        )

        # --- Phase 1: Validate target upfront ---
        if not is_in_workspace(target_x, target_y, target_z):
            response.success = False
            response.message = (
                f"Target ({target_x:.3f},{target_y:.3f},{target_z:.3f}) "
                "is outside workspace"
            )
            return response

        target_joints = analytical_ik_5d(
            target_x, target_y, target_z, cur_pitch, cur_roll,
        )
        if target_joints is not None:
            target_pitch = cur_pitch
        else:
            target_pitch = _best_pitch_for_position(target_x, target_y, target_z)
            if target_pitch is None:
                response.success = False
                response.message = (
                    f"Target ({target_x:.3f},{target_y:.3f},{target_z:.3f}) "
                    "has no valid IK for any orientation"
                )
                return response
            target_joints = analytical_ik_5d(
                target_x, target_y, target_z, target_pitch, cur_roll,
            )
            if target_joints is None:
                response.success = False
                response.message = (
                    f"Target ({target_x:.3f},{target_y:.3f},{target_z:.3f}) "
                    f"IK failed at best pitch={math.degrees(target_pitch):.1f}°"
                )
                return response

        tjfmt = ", ".join(f"{j:.4f}" for j in target_joints)
        self._flog.debug(
            f"plan_straight_line: target_pitch={math.degrees(target_pitch):.1f}° "
            f"target_joints=[{tjfmt}]"
        )

        dx = target_x - cur_x
        dy = target_y - cur_y
        dz = target_z - cur_z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 1e-4:
            response.success = True
            response.message = "Already at target (distance < 0.1mm)"
            return response

        n_points = max(2, int(math.ceil(distance / step)) + 1)
        total_time = distance / speed

        self._flog.debug(
            f"plan_straight_line: distance={distance*100:.2f}cm "
            f"n_points={n_points} nominal_time={total_time:.2f}s"
        )

        # --- Phase 2: Generate and validate full path ---
        # Pitch is interpolated smoothly from cur_pitch to target_pitch so the
        # gripper orientation transitions gradually (no mid-path jumps).
        grip = self._current_grip_position
        waypoint_positions = [list(q) + [grip]]

        n_flips_avoided = 0
        for i in range(1, n_points):
            t = i / (n_points - 1)
            wx = cur_x + t * dx
            wy = cur_y + t * dy
            wz = cur_z + t * dz
            wp_pitch = cur_pitch + t * (target_pitch - cur_pitch)

            solutions = analytical_ik_5d_all(wx, wy, wz, wp_pitch, cur_roll)
            if not solutions:
                response.success = False
                response.message = (
                    f"Straight-line path infeasible: IK failed at waypoint "
                    f"{i}/{n_points - 1} ({wx:.3f}, {wy:.3f}, {wz:.3f}), "
                    f"pitch={math.degrees(wp_pitch):.1f}°"
                )
                return response
            prev = waypoint_positions[-1][:5]
            joints = min(
                solutions,
                key=lambda s: sum((a - b) ** 2 for a, b in zip(s, prev)),
            )
            if len(solutions) > 1 and solutions[0] != joints:
                n_flips_avoided += 1
            waypoint_positions.append(list(joints) + [grip])

        if n_flips_avoided:
            self._flog.info(
                f"Continuity: avoided {n_flips_avoided} IK config flip(s) "
                f"along {n_points} waypoints"
            )

        # Per-segment velocity limiting: stretch only the segments where a
        # joint would exceed _MAX_JOINT_VEL, leaving the rest at nominal speed.
        # This prevents a single waypoint with a large pitch change from
        # slowing the entire trajectory by 10-20x.
        dt_nominal = total_time / (n_points - 1)
        segment_dts = []
        for i in range(1, n_points):
            max_seg_vel = 0.0
            for j in range(5):
                v = abs(waypoint_positions[i][j] - waypoint_positions[i - 1][j]) / dt_nominal
                if v > max_seg_vel:
                    max_seg_vel = v
            if max_seg_vel > self._MAX_JOINT_VEL:
                segment_dts.append(dt_nominal * max_seg_vel / self._MAX_JOINT_VEL)
            else:
                segment_dts.append(dt_nominal)

        total_time = sum(segment_dts)
        n_stretched = sum(1 for sdt in segment_dts if sdt > dt_nominal * 1.01)
        if n_stretched:
            self._flog.info(
                f"Stretched {n_stretched}/{len(segment_dts)} segments "
                f"for joint velocity limit ({self._MAX_JOINT_VEL} rad/s), "
                f"total_time={total_time:.2f}s"
            )

        timestamps = [0.0]
        for sdt in segment_dts:
            timestamps.append(timestamps[-1] + sdt)

        n_joints = len(self._CONTROLLER_JOINTS)
        zero_vel = [0.0] * n_joints

        points = []
        for i in range(n_points):
            pt = JointTrajectoryPoint()
            pt.positions = waypoint_positions[i]

            t_sec = timestamps[i]
            pt.time_from_start = Duration(
                sec=int(t_sec), nanosec=int((t_sec - int(t_sec)) * 1e9),
            )

            if i == 0 or i == n_points - 1:
                pt.velocities = zero_vel
            else:
                dt_span = timestamps[i + 1] - timestamps[i - 1]
                pt.velocities = [
                    (waypoint_positions[i + 1][j] - waypoint_positions[i - 1][j])
                    / dt_span
                    for j in range(n_joints)
                ]
            points.append(pt)

        self.get_logger().info(
            f"Straight-line path: {n_points} waypoints, "
            f"{distance * 100:.1f} cm, {total_time:.2f}s"
        )

        wp0 = ", ".join(f"{v:.4f}" for v in waypoint_positions[0])
        wpN = ", ".join(f"{v:.4f}" for v in waypoint_positions[-1])
        dt_min = min(segment_dts) if segment_dts else 0
        dt_max = max(segment_dts) if segment_dts else 0
        self._flog.debug(
            f"plan_straight_line: first_wp=[{wp0}] last_wp=[{wpN}] "
            f"dt_range=[{dt_min:.4f}s, {dt_max:.4f}s] total_time={total_time:.2f}s"
        )

        if not execute:
            # Advance virtual position so the next dry-run call starts from
            # the end of this path (separate from hardware joint_state_cb).
            final = waypoint_positions[-1]
            self._virtual_arm_positions = list(final[:5])
            self._virtual_grip_position = final[5]
            response.success = True
            response.message = (
                f"Plan OK: {n_points} waypoints, "
                f"{distance * 100:.1f} cm, {total_time:.2f}s"
            )
            return response

        self._virtual_arm_positions = None
        self._virtual_grip_position = None
        result = self._execute_fjt(points, total_time)
        if result == PlanResult.SUCCESS:
            final = waypoint_positions[-1]
            self._virtual_arm_positions = list(final[:5])
            self._virtual_grip_position = final[5]
        response.success = (result == PlanResult.SUCCESS)
        response.message = self._format_result(result)
        return response

    def _set_gripper_cb(self, request, response):
        """Move grip_joint to target_grip while keeping all arm joints fixed."""
        target_grip = self.get_parameter("target_grip").value
        execute = self._apply_dry_run(
            self.get_parameter("execute").value, "set_gripper"
        )
        self._flog.info(
            f"set_gripper: target_grip={target_grip:.3f} "
            f"execute={execute} dry_run={self._is_dry_run}"
        )

        q = self._virtual_arm_positions or self._current_arm_positions
        if q is None:
            response.success = False
            response.message = (
                f"No joint state received on {self._JOINT_STATES_TOPIC} yet"
            )
            return response

        grip = (self._virtual_grip_position
                if self._virtual_grip_position is not None
                else self._current_grip_position)

        self._flog.info(f"set_gripper: {grip:.3f} -> {target_grip:.3f}")
        self.get_logger().info("set_gripper requested")

        delta = abs(target_grip - grip)
        self._flog.debug(
            f"set_gripper: delta={delta:.4f} execute={execute}"
        )
        if delta < 1e-4:
            response.success = True
            response.message = "Gripper already at target"
            return response

        duration = max(0.5, delta / self._MAX_JOINT_VEL)
        self._flog.debug(
            f"set_gripper: duration={duration:.2f}s "
            f"(delta={delta:.3f} / max_vel={self._MAX_JOINT_VEL})"
        )

        n_joints = len(self._CONTROLLER_JOINTS)
        zero_vel = [0.0] * n_joints

        start_pos = list(q) + [grip]
        end_pos = list(q) + [target_grip]

        pt0 = JointTrajectoryPoint()
        pt0.positions = start_pos
        pt0.velocities = zero_vel
        pt0.time_from_start = Duration(sec=0, nanosec=0)

        pt1 = JointTrajectoryPoint()
        pt1.positions = end_pos
        pt1.velocities = zero_vel
        pt1.time_from_start = Duration(
            sec=int(duration),
            nanosec=int((duration - int(duration)) * 1e9),
        )

        if not execute:
            self._virtual_grip_position = target_grip
            response.success = True
            response.message = (
                f"Plan OK: grip {grip:.3f} -> {target_grip:.3f} "
                f"({duration:.2f}s)"
            )
            return response

        self._virtual_arm_positions = None
        self._virtual_grip_position = None
        result = self._execute_fjt([pt0, pt1], duration)
        if result == PlanResult.SUCCESS:
            self._virtual_arm_positions = list(q)
            self._virtual_grip_position = target_grip
        response.success = (result == PlanResult.SUCCESS)
        response.message = self._format_result(result)
        return response

    # ── Helpers ──

    @staticmethod
    def _wait_for_future(future, timeout_sec):
        """Poll until *future* completes. Returns True if done, False on timeout."""
        deadline = _time.monotonic() + timeout_sec
        while not future.done():
            if _time.monotonic() > deadline:
                return False
            _time.sleep(0.01)
        return True

    # ── FollowJointTrajectory execution ──

    def _execute_fjt(self, points, total_time):
        """Send a pre-computed trajectory to the controller."""
        if self._is_dry_run:
            self._flog.critical(
                "SAFETY GATE: _execute_fjt called while dry_run=True! "
                "Refusing to send trajectory to controller."
            )
            self.get_logger().error(
                "SAFETY: dry_run=True — trajectory NOT sent to controller"
            )
            return PlanResult.SUCCESS

        if not self._fjt_client.wait_for_server(timeout_sec=5.0):
            msg = f"FollowJointTrajectory server {self._FJT_ACTION} not available"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return PlanResult.PLANNING_FAILED

        p0_fmt = ", ".join(f"{v:.4f}" for v in points[0].positions)
        pN_fmt = ", ".join(f"{v:.4f}" for v in points[-1].positions)
        last_t = points[-1].time_from_start
        dur_s = last_t.sec + last_t.nanosec * 1e-9
        self._flog.debug(
            f"_execute_fjt: {len(points)} points, duration={dur_s:.2f}s "
            f"first=[{p0_fmt}] last=[{pN_fmt}]"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(self._CONTROLLER_JOINTS)
        goal.trajectory.points = points

        future = self._fjt_client.send_goal_async(goal)
        if not self._wait_for_future(future, timeout_sec=10.0):
            msg = "Timed out waiting for trajectory goal acceptance"
            self.get_logger().error(msg)
            self._flog.error(f"FJT: {msg}")
            return PlanResult.PLANNING_FAILED

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            msg = "Trajectory goal rejected by controller"
            self.get_logger().error(msg)
            self._flog.error(f"FJT: {msg}")
            return PlanResult.PLANNING_FAILED

        self._flog.info(
            f"FJT: trajectory ACCEPTED — {len(points)} pts, "
            f"{total_time:.2f}s — ARM IS MOVING"
        )
        result_future = goal_handle.get_result_async()
        timeout = total_time + 10.0
        if not self._wait_for_future(result_future, timeout_sec=timeout):
            msg = "Timed out waiting for trajectory execution"
            self.get_logger().error(msg)
            self._flog.error(f"FJT: {msg}")
            return PlanResult.PLANNING_FAILED

        result = result_future.result()
        if result is None:
            msg = "No result from FollowJointTrajectory"
            self.get_logger().error(msg)
            self._flog.error(f"FJT: {msg}")
            return PlanResult.PLANNING_FAILED

        error_code = result.result.error_code
        if error_code == 0:  # SUCCESSFUL
            self._flog.info("FJT: trajectory executed successfully")
            return PlanResult.SUCCESS

        self.get_logger().error(f"FollowJointTrajectory error code: {error_code}")
        self._flog.error(f"FJT: error code {error_code}")
        self._last_moveit_error = error_code
        return PlanResult.PLANNING_FAILED

    # ── MoveIt interface ──

    def _plan_joint_target(self, joint_values, execute=True,
                           vel_scale=0.3, accel_scale=0.3):
        """Send a joint-space goal to move_group and optionally execute."""
        if self._is_dry_run and execute:
            self._flog.warning(
                "_plan_joint_target: dry_run OVERRIDE execute=True → False"
            )
            self.get_logger().warn(
                "dry_run active — forcing plan_only for MoveIt goal"
            )
            execute = False
        jfmt = ", ".join(f"{v:.4f}" for v in joint_values)
        self._flog.debug(
            f"_plan_joint_target: joints=[{jfmt}] execute={execute} "
            f"dry_run={self._is_dry_run}"
        )
        self._flog.debug(
            f"_plan_joint_target: joints=[{jfmt}] execute={execute}"
        )
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            msg = "move_group action server not available"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return PlanResult.PLANNING_FAILED

        goal_msg = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.planner_id = "RRTConnect"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 2.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = accel_scale
        self._flog.debug(
            f"_plan_joint_target: vel_scale={req.max_velocity_scaling_factor} "
            f"accel_scale={req.max_acceleration_scaling_factor} "
            f"plan_only={not execute}"
        )

        req.start_state.is_diff = True

        req.workspace_parameters.header.frame_id = PLANNING_FRAME
        req.workspace_parameters.min_corner.x = -1.0
        req.workspace_parameters.min_corner.y = -1.0
        req.workspace_parameters.min_corner.z = -1.0
        req.workspace_parameters.max_corner.x = 1.0
        req.workspace_parameters.max_corner.y = 1.0
        req.workspace_parameters.max_corner.z = 1.0

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
        if not self._wait_for_future(future, timeout_sec=10.0):
            msg = "Timed out waiting for MoveIt goal acceptance"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return PlanResult.PLANNING_FAILED

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            msg = "Goal rejected by move_group"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return PlanResult.PLANNING_FAILED

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, timeout_sec=30.0):
            msg = "Timed out waiting for MoveIt planning result"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return PlanResult.PLANNING_FAILED

        result = result_future.result()
        if result is None:
            msg = "No result from move_group"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return PlanResult.PLANNING_FAILED

        error_code = result.result.error_code.val
        if error_code == 1:  # MoveItErrorCodes.SUCCESS
            self._virtual_arm_positions = list(joint_values[:5])
            self._virtual_grip_position = 0.0
            action = "executed" if execute else "planned"
            self._flog.info(
                f"MoveIt: motion {action} successfully "
                f"(plan_only={not execute})"
            )
            self.get_logger().info(f"Motion {action} successfully")
            return PlanResult.SUCCESS
        else:
            self._flog.error(f"MoveIt error code: {error_code}")
            self.get_logger().error(f"MoveIt error code: {error_code}")
            self._last_moveit_error = error_code
            return PlanResult.PLANNING_FAILED


def main(args=None):
    rclpy.init(args=args)
    node = X3plus5DofPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
