#!/usr/bin/env python3
"""
Pick-and-place orchestrator for the X3plus arm.

Exposes ~/pick and ~/place services that coordinate the planner's
low-level services (set_gripper, plan_position_orient, plan_straight_line)
into complete grasp/release sequences.

The gripper always faces straight down (pitch = -pi/2).  The block's
6DOF pose (position + quaternion of the upper surface) is read from
node parameters; the yaw component maps to the gripper roll (J5).

Prerequisites — the planner must be running:
  Terminal 1 (from docker_all/):  ./scripts/real_up.sh

Usage:
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo x3plus_pick_place"
"""

import math
import time as _time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType, ParameterDescriptor
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger

# ---------------------------------------------------------------------------
# Kinematic helpers (imported from planner module, no ROS needed)
# ---------------------------------------------------------------------------

from monty_demo.opus_plan_and_imp.log_utils import LOG_DIR, make_file_logger
from monty_demo.x3plus_5dof_planner import (
    analytical_ik_5d,
    is_in_workspace,
    is_safe_from_base_collision,
    JOINT_LIMITS,
    BASE_XY,
)

# ---------------------------------------------------------------------------
# Pick-and-place constants
# ---------------------------------------------------------------------------

PITCH_DOWN = -math.pi / 2

FINGERTIP_BELOW_WRIST = 0.012   # worst case (fully open), measured from STL meshes
MIN_FLOOR_CLEARANCE = 0.005     # 5 mm safety margin above floor
BLOCK_HEIGHT = 0.03             # 3 cm cube default
GRASP_DEPTH = 0.015             # wrist goes this far below block surface
APPROACH_HEIGHT = 0.04          # clearance above grasp height
GRIP_OPEN = 0.0                 # grip_joint fully open
GRIP_CLOSED = -1.3              # grip_joint closed on 3 cm block (tune on robot)

J5_LO, J5_HI = JOINT_LIMITS["arm_joint5"]  # -pi/2 .. pi
MAX_TILT_RAD = math.radians(5.0)

PLANNER_NODE = "/x3plus_5dof_planner"

# ---------------------------------------------------------------------------
# Standalone helpers (no ROS dependencies — importable by test script)
# ---------------------------------------------------------------------------


def quaternion_to_rpy(qx, qy, qz, qw):
    """Quaternion → (roll, pitch, yaw) using intrinsic XYZ convention."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def yaw_to_quaternion(yaw):
    """Pure Z-rotation yaw → quaternion (qx, qy, qz, qw)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def map_yaw_to_j5(yaw):
    """Map block yaw to J5 range, exploiting cube symmetry (try yaw ± pi)."""
    for candidate in [yaw, yaw - math.pi, yaw + math.pi]:
        if J5_LO - 1e-6 <= candidate <= J5_HI + 1e-6:
            return candidate
    return None


def parse_block_pose(bx, by, bz, qx, qy, qz, qw):
    """Parse a 6DOF block surface pose. Returns (roll_j5, reason) or (None, reason)."""
    roll_b, pitch_b, yaw_b = quaternion_to_rpy(qx, qy, qz, qw)

    if abs(roll_b) > MAX_TILT_RAD or abs(pitch_b) > MAX_TILT_RAD:
        return None, (
            f"Block tilt too large: roll={math.degrees(roll_b):.1f}° "
            f"pitch={math.degrees(pitch_b):.1f}° (max {math.degrees(MAX_TILT_RAD):.0f}°)"
        )

    j5 = map_yaw_to_j5(yaw_b)
    if j5 is None:
        return None, (
            f"Block yaw={math.degrees(yaw_b):.1f}° cannot be mapped to J5 "
            f"range [{math.degrees(J5_LO):.0f}°, {math.degrees(J5_HI):.0f}°]"
        )

    return j5, "ok"


def is_reachable(x, y, z_wrist, roll, approach_height=APPROACH_HEIGHT,
                 grip_q=0.0):
    """Check if a gripper-down target is reachable, floor-safe, and base-safe.

    Returns (True, "") or (False, reason).
    grip_q defaults to 0.0 (open) for worst-case base collision checking.
    Validates intermediate waypoints along the vertical descent to ensure
    the full straight-line path is feasible.
    """
    z_fingertip = z_wrist - FINGERTIP_BELOW_WRIST
    if z_fingertip < MIN_FLOOR_CLEARANCE:
        return False, (
            f"Floor collision: fingertip z={z_fingertip * 100:.1f} cm "
            f"< min {MIN_FLOOR_CLEARANCE * 100:.1f} cm"
        )

    dx = x - BASE_XY
    if dx * dx + y * y < 1e-8:
        return False, "Target too close to J1 axis"

    q1 = -math.atan2(y, dx)
    if abs(q1) > math.pi / 2:
        return False, (
            f"Azimuth q1={math.degrees(q1):.1f}° outside ±90° range"
        )

    if not is_in_workspace(x, y, z_wrist):
        return False, (
            f"Wrist ({x:.3f},{y:.3f},{z_wrist:.3f}) outside workspace"
        )

    if not is_safe_from_base_collision(x, y, z_wrist, grip_q):
        return False, (
            f"Base collision: wrist ({x:.3f},{y:.3f},{z_wrist:.3f}) "
            "too close to chassis/wheels"
        )

    ik = analytical_ik_5d(x, y, z_wrist, PITCH_DOWN, roll)
    if ik is None:
        return False, (
            f"IK failed at grasp height z_wrist={z_wrist:.4f}, roll={math.degrees(roll):.1f}°"
        )

    z_approach = z_wrist + approach_height
    if not is_in_workspace(x, y, z_approach):
        return False, (
            f"Approach ({x:.3f},{y:.3f},{z_approach:.3f}) outside workspace"
        )

    if not is_safe_from_base_collision(x, y, z_approach, grip_q):
        return False, (
            f"Base collision at approach: ({x:.3f},{y:.3f},{z_approach:.3f}) "
            "too close to chassis/wheels"
        )

    ik_a = analytical_ik_5d(x, y, z_approach, PITCH_DOWN, roll)
    if ik_a is None:
        return False, (
            f"IK failed at approach height z={z_approach:.4f}"
        )

    cartesian_step = 0.01
    n_pts = max(2, int(math.ceil(approach_height / cartesian_step)) + 1)
    for i in range(1, n_pts - 1):
        t = i / (n_pts - 1)
        z_mid = z_approach + t * (z_wrist - z_approach)
        if analytical_ik_5d(x, y, z_mid, PITCH_DOWN, roll) is None:
            return False, (
                f"IK failed at descent height z={z_mid:.4f}, "
                f"roll={math.degrees(roll):.1f}°"
            )

    return True, ""


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------


class X3plusPickPlace(Node):

    def __init__(self):
        super().__init__("x3plus_pick_place")

        self._srv_cb_group = ReentrantCallbackGroup()
        self._cli_cb_group = MutuallyExclusiveCallbackGroup()

        # --- Planner service clients ---
        self._set_params_cli = self.create_client(
            SetParameters, f"{PLANNER_NODE}/set_parameters",
            callback_group=self._cli_cb_group,
        )
        self._set_gripper_cli = self.create_client(
            Trigger, f"{PLANNER_NODE}/set_gripper",
            callback_group=self._cli_cb_group,
        )
        self._plan_pos_orient_cli = self.create_client(
            Trigger, f"{PLANNER_NODE}/plan_position_orient",
            callback_group=self._cli_cb_group,
        )
        self._straight_line_cli = self.create_client(
            Trigger, f"{PLANNER_NODE}/plan_straight_line",
            callback_group=self._cli_cb_group,
        )
        self._go_home_cli = self.create_client(
            Trigger, f"{PLANNER_NODE}/go_home",
            callback_group=self._cli_cb_group,
        )

        # --- Own services ---
        self.create_service(
            Trigger, "~/pick", self._pick_cb,
            callback_group=self._srv_cb_group,
        )
        self.create_service(
            Trigger, "~/place", self._place_cb,
            callback_group=self._srv_cb_group,
        )

        # --- Parameters ---
        self.declare_parameter("block_x", 0.2)
        self.declare_parameter("block_y", 0.0)
        self.declare_parameter("block_z", BLOCK_HEIGHT)
        self.declare_parameter("block_qx", 0.0)
        self.declare_parameter("block_qy", 0.0)
        self.declare_parameter("block_qz", 0.0)
        self.declare_parameter("block_qw", 1.0)
        self.declare_parameter("place_x", 0.2)
        self.declare_parameter("place_y", 0.03)
        self.declare_parameter("place_z", BLOCK_HEIGHT)
        self.declare_parameter("place_qx", 0.0)
        self.declare_parameter("place_qy", 0.0)
        self.declare_parameter("place_qz", 0.0)
        self.declare_parameter("place_qw", 1.0)
        self.declare_parameter("approach_height", APPROACH_HEIGHT,
                               ParameterDescriptor(description="Clearance above grasp (m)"))
        self.declare_parameter("grasp_depth", GRASP_DEPTH,
                               ParameterDescriptor(description="How far below block surface the wrist descends (m)"))
        self.declare_parameter("grip_open", GRIP_OPEN)
        self.declare_parameter("grip_closed", GRIP_CLOSED)
        self.declare_parameter("execute", True)
        self.declare_parameter("dry_run", False,
                               ParameterDescriptor(
                                   description="Safety lockout: when True, forces "
                                               "execute=False on all planner calls"))

        _log_path = f"{LOG_DIR}/x3plus_pick_place.log"
        self._flog = make_file_logger("x3plus_pick_place", _log_path)
        self._flog.info("=" * 60)
        self._flog.info("X3plus pick-place node starting")

        self.get_logger().info(
            "Pick-place ready (log: %s). Services: ~/pick, ~/place" % _log_path
        )

    # ── Planner RPC helpers ──

    @property
    def _is_dry_run(self):
        return self.get_parameter("dry_run").value

    def _resolve_execute(self, caller):
        """Read execute param, apply defense-in-depth dry_run override."""
        execute = self.get_parameter("execute").value
        if self._is_dry_run and execute:
            self._flog.warning(
                f"{caller}: dry_run OVERRIDE execute=True → False"
            )
            self.get_logger().warn(
                f"{caller}: dry_run active — forcing plan_only"
            )
            return False
        self._flog.info(
            f"{caller}: execute={execute} dry_run={self._is_dry_run}"
        )
        return execute

    def _wait(self, future, timeout=30.0):
        deadline = _time.monotonic() + timeout
        while not future.done():
            if _time.monotonic() > deadline:
                return None
            _time.sleep(0.05)
        return future.result()

    def _set_planner_params(self, params_dict):
        req = SetParameters.Request()
        for name, value in params_dict.items():
            p = Parameter()
            p.name = name
            pv = ParameterValue()
            if isinstance(value, bool):
                pv.type = ParameterType.PARAMETER_BOOL
                pv.bool_value = value
            else:
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = float(value)
            p.value = pv
            req.parameters.append(p)
        resp = self._wait(self._set_params_cli.call_async(req), timeout=5.0)
        if resp is None:
            msg = "_set_planner_params: timed out"
            self.get_logger().error(msg)
            self._flog.error(msg)
            return None
        for i, r in enumerate(resp.results):
            if not r.successful:
                names = list(params_dict.keys())
                msg = f"_set_planner_params: '{names[i]}' failed: {r.reason}"
                self.get_logger().error(msg)
                self._flog.error(msg)
        self._flog.debug(f"_set_planner_params: {params_dict}")
        return resp

    def _call_trigger(self, client, timeout=60.0):
        resp = self._wait(client.call_async(Trigger.Request()), timeout)
        if resp is None:
            return False, "Service call timed out"
        return resp.success, resp.message

    # ── Sequence steps ──

    def _step_set_gripper(self, grip_val, execute):
        self._set_planner_params({
            "target_grip": grip_val,
            "execute": execute,
        })
        ok, msg = self._call_trigger(self._set_gripper_cli, timeout=15.0)
        if not ok:
            return False, f"set_gripper({grip_val:.2f}) failed: {msg}"
        return True, msg

    def _step_move_above(self, x, y, z, roll, execute):
        self._set_planner_params({
            "target_x": x,
            "target_y": y,
            "target_z": z,
            "target_pitch": PITCH_DOWN,
            "target_roll": roll,
            "execute": execute,
        })
        ok, msg = self._call_trigger(self._plan_pos_orient_cli, timeout=30.0)
        if not ok:
            return False, f"plan_position_orient failed: {msg}"
        return True, msg

    def _step_straight_line(self, x, y, z, execute):
        self._set_planner_params({
            "target_x": x,
            "target_y": y,
            "target_z": z,
            "execute": execute,
        })
        ok, msg = self._call_trigger(self._straight_line_cli, timeout=60.0)
        if not ok:
            return False, f"plan_straight_line failed: {msg}"
        return True, msg

    # ── Service callbacks ──

    def _pick_cb(self, request, response):
        bx = self.get_parameter("block_x").value
        by = self.get_parameter("block_y").value
        bz = self.get_parameter("block_z").value
        qx = self.get_parameter("block_qx").value
        qy = self.get_parameter("block_qy").value
        qz = self.get_parameter("block_qz").value
        qw = self.get_parameter("block_qw").value
        ah = self.get_parameter("approach_height").value
        gd = self.get_parameter("grasp_depth").value
        g_open = self.get_parameter("grip_open").value
        g_closed = self.get_parameter("grip_closed").value
        execute = self._resolve_execute("pick")

        self._flog.info(
            f"pick: block=({bx:.3f},{by:.3f},{bz:.3f}) "
            f"q=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f}) execute={execute}"
        )
        self.get_logger().info("pick requested")

        j5, reason = parse_block_pose(bx, by, bz, qx, qy, qz, qw)
        if j5 is None:
            response.success = False
            response.message = f"Pose rejected: {reason}"
            return response

        z_grasp = bz - gd
        z_approach = z_grasp + ah

        ok, reason = is_reachable(bx, by, z_grasp, j5, ah)
        if not ok:
            response.success = False
            response.message = f"Not reachable: {reason}"
            return response

        self._flog.info(
            f"pick: yaw={math.degrees(j5):.1f}° z_grasp={z_grasp:.4f} "
            f"z_approach={z_approach:.4f}"
        )

        steps = [
            ("open gripper", lambda: self._step_set_gripper(g_open, execute)),
            ("move above", lambda: self._step_move_above(bx, by, z_approach, j5, execute)),
            ("descend", lambda: self._step_straight_line(bx, by, z_grasp, execute)),
            ("close gripper", lambda: self._step_set_gripper(g_closed, execute)),
            ("ascend", lambda: self._step_straight_line(bx, by, z_approach, execute)),
        ]
        for i, (name, fn) in enumerate(steps, 1):
            self._flog.info(f"pick step {i}/{len(steps)}: {name}")
            ok, msg = fn()
            self._flog.info(f"  result: ok={ok} msg={msg}")
            if not ok:
                self.get_logger().error(f"pick failed at step '{name}'")
                response.success = False
                response.message = f"Pick failed at '{name}': {msg}"
                return response

        self.get_logger().info("pick complete")
        response.success = True
        response.message = "Pick complete"
        return response

    def _place_cb(self, request, response):
        px = self.get_parameter("place_x").value
        py = self.get_parameter("place_y").value
        pz = self.get_parameter("place_z").value
        qx = self.get_parameter("place_qx").value
        qy = self.get_parameter("place_qy").value
        qz = self.get_parameter("place_qz").value
        qw = self.get_parameter("place_qw").value
        ah = self.get_parameter("approach_height").value
        gd = self.get_parameter("grasp_depth").value
        g_open = self.get_parameter("grip_open").value
        execute = self._resolve_execute("place")

        self._flog.info(
            f"place: pos=({px:.3f},{py:.3f},{pz:.3f}) "
            f"q=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f}) execute={execute}"
        )
        self.get_logger().info("place requested")

        j5, reason = parse_block_pose(px, py, pz, qx, qy, qz, qw)
        if j5 is None:
            response.success = False
            response.message = f"Pose rejected: {reason}"
            return response

        z_place = pz - gd
        z_approach = z_place + ah

        ok, reason = is_reachable(px, py, z_place, j5, ah)
        if not ok:
            response.success = False
            response.message = f"Not reachable: {reason}"
            return response

        self._flog.info(
            f"place: yaw={math.degrees(j5):.1f}° z_place={z_place:.4f} "
            f"z_approach={z_approach:.4f}"
        )

        steps = [
            ("move above place", lambda: self._step_move_above(px, py, z_approach, j5, execute)),
            ("descend", lambda: self._step_straight_line(px, py, z_place, execute)),
            ("open gripper", lambda: self._step_set_gripper(g_open, execute)),
            ("ascend", lambda: self._step_straight_line(px, py, z_approach, execute)),
        ]
        for i, (name, fn) in enumerate(steps, 1):
            self._flog.info(f"place step {i}/{len(steps)}: {name}")
            ok, msg = fn()
            self._flog.info(f"  result: ok={ok} msg={msg}")
            if not ok:
                self.get_logger().error(f"place failed at step '{name}'")
                response.success = False
                response.message = f"Place failed at '{name}': {msg}"
                return response

        self.get_logger().info("place complete")
        response.success = True
        response.message = "Place complete"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = X3plusPickPlace()

    for cli in [node._set_params_cli, node._set_gripper_cli,
                node._plan_pos_orient_cli, node._straight_line_cli,
                node._go_home_cli]:
        if not cli.wait_for_service(timeout_sec=15.0):
            msg = (
                f"Planner service {cli.srv_name} not available after 15 s. "
                "Is the planner running?"
            )
            node.get_logger().error(msg)
            node._flog.error(msg)
            node.destroy_node()
            rclpy.shutdown()
            return

    node.get_logger().info("Connected to all planner services.")

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
