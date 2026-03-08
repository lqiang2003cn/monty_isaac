#!/usr/bin/env python3
"""
100-point reachability test for the X3plus 5-DOF arm on a real robot.

Generates 100 random kinematically-reachable wrist points with good spatial
diversity, then commands the real robot to each via the x3plus_5dof_planner
node (MoveIt collision-free planning).

Prerequisites:
  Terminal 1 (from docker_all/):  ./scripts/real_up.sh
    — starts bringup, MoveIt, and the planner automatically.

Usage (Terminal 2, from docker_all/):
  # Dry-run: plan all motions, arm does NOT move (safe first check)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo x3plus_reachability_test --dry-run"

  # Full test on real robot
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo x3plus_reachability_test"

  # Test 5 points first to validate the full chain
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo x3plus_reachability_test -n 5"
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Kinematic constants (from URDF, matching x3plus_5dof_planner.py)
# ---------------------------------------------------------------------------

L1 = 0.0829
L2 = 0.0829
L3 = 0.17455
L3X = 0.00215
BASE_XY = 0.09825
BASE_Z = 0.076 + 0.102 + 0.0405  # 0.2185 m

JOINT_LIMITS = [
    (-1.5708, 1.5708),   # arm_joint1
    (-1.5708, 1.5708),   # arm_joint2
    (-1.5708, 1.5708),   # arm_joint3
    (-1.5708, 1.5708),   # arm_joint4
    (-1.5708, 3.14159),  # arm_joint5
]

PLANNER_NODE = "/x3plus_5dof_planner"

# ---------------------------------------------------------------------------
# Workspace boundary (must match x3plus_5dof_planner.is_in_workspace exactly)
# ---------------------------------------------------------------------------

_MAX_REACH = L1 + L2 + L3 + L3X


def is_in_workspace(x: float, y: float, z: float, margin: float = 0.01) -> bool:
    dx = x - BASE_XY
    if dx < -margin:
        return False
    r = math.sqrt(dx * dx + y * y)
    if r > _MAX_REACH - margin:
        return False
    if r < 0.01:
        return False
    if z < 0.0 or z > BASE_Z + _MAX_REACH - margin:
        return False
    return True


# ---------------------------------------------------------------------------
# Forward kinematics to arm_link5 origin (wrist point)
# ---------------------------------------------------------------------------

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
    """FK to arm_link5 origin (the wrist point the planner targets)."""
    T = _T_BASE.copy()
    for i, (xyz, rpy, axis) in enumerate(_CHAIN):
        T = T @ _joint_tf(xyz, rpy, axis, qs[i])
    return T[:3, 3]


from monty_demo.opus_plan_and_imp.opus_joint_config import INIT_ARM_POSITIONS
HOME_JOINTS = list(INIT_ARM_POSITIONS)
HOME_XYZ = fk_wrist(HOME_JOINTS)

# ---------------------------------------------------------------------------
# Analytical IK (mirror of x3plus_5dof_planner.py)
# ---------------------------------------------------------------------------


def _in_limits(joint_idx, val):
    lo, hi = JOINT_LIMITS[joint_idx]
    return lo - 1e-6 <= val <= hi + 1e-6


def analytical_ik(x, y, z, pitch, roll=0.0):
    """Returns (q1..q5) or None."""
    dx = x - BASE_XY
    if dx * dx + y * y < 1e-8:
        return None
    q1 = -math.atan2(y, dx)
    if abs(q1) > math.pi / 2:
        return None

    r = math.sqrt(dx * dx + y * y)
    h = z - BASE_Z
    phi = pitch - math.pi / 2

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
        if _in_limits(1, q2) and _in_limits(2, q3) and _in_limits(3, q4):
            q5 = roll
            if _in_limits(4, q5):
                return (q1, q2, q3, q4, q5)
    return None


def _best_pitch(x, y, z):
    for p in [-math.pi / 2, 0.0, -math.pi / 4, math.pi / 4, math.pi / 2]:
        if analytical_ik(x, y, z, p) is not None:
            return p
    for p in np.linspace(-math.pi / 2, math.pi / 2, 36):
        if analytical_ik(x, y, z, float(p)) is not None:
            return float(p)
    return None


# ---------------------------------------------------------------------------
# Test point generation
# ---------------------------------------------------------------------------


def generate_test_points(n=100, seed=42, min_dist=0.018):
    """Sample n random, IK-verified wrist points with spatial diversity.

    Samples uniformly in joint space (q1..q4), computes FK, verifies IK
    round-trip, and enforces a minimum distance between accepted points
    for good spatial coverage.
    """
    rng = np.random.default_rng(seed)
    points = []
    joint_configs = []
    max_attempts = n * 500

    for _ in range(max_attempts):
        if len(points) >= n:
            break

        qs = [float(rng.uniform(lo, hi)) for lo, hi in JOINT_LIMITS[:4]] + [0.0]
        xyz = fk_wrist(qs)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        if z < 0.05:
            continue

        if not is_in_workspace(x, y, z):
            continue

        pitch = _best_pitch(x, y, z)
        if pitch is None:
            continue

        too_close = any(
            math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) < min_dist
            for px, py, pz in points
        )
        if too_close:
            continue

        points.append((x, y, z))
        joint_configs.append(qs)

    if len(points) < n:
        print(f"WARNING: Only generated {len(points)}/{n} points "
              f"(try lowering min_dist or raising max_attempts)")

    return points, joint_configs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_points(points, results=None, out_path="x3plus_test_points.png"):
    """3D + projected views of the test points, colored by result if available."""
    import matplotlib.pyplot as plt

    pts = np.array(points)

    if results is not None:
        colors = ["#2ecc71" if r["success"] else "#e74c3c" for r in results]
        labels_ok = sum(1 for r in results if r["success"])
        labels_fail = len(results) - labels_ok
        title_suffix = f"  ({labels_ok} passed, {labels_fail} failed)"
    else:
        colors = "#3498db"
        title_suffix = ""

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"X3plus 100-Point Reachability Test{title_suffix}",
                 fontsize=14, fontweight="bold")

    # 3D scatter
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D view")

    # Top (XY)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(pts[:, 0], pts[:, 1], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top view (XY)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Side (R vs Z)
    r_xy = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(r_xy, pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    ax3.set_xlabel("Radial distance (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("Side cross-section (R vs Z)")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # Front (YZ)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(pts[:, 1], pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    ax4.set_xlabel("Y (m)")
    ax4.set_ylabel("Z (m)")
    ax4.set_title("Front view (YZ)")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    if results is not None:
        for ax in [ax2, ax3, ax4]:
            ax.plot([], [], "o", color="#2ecc71", label="Pass")
            ax.plot([], [], "o", color="#e74c3c", label="Fail")
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization -> {out_path}")


# ---------------------------------------------------------------------------
# ROS 2 test execution
# ---------------------------------------------------------------------------


def _make_param(name, value):
    """Build a rcl_interfaces Parameter message."""
    from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

    p = Parameter()
    p.name = name
    pv = ParameterValue()
    pv.type = ParameterType.PARAMETER_DOUBLE
    pv.double_value = float(value)
    p.value = pv
    return p


def _publish_ground_plane(node, logger):
    """Add a large ground-plane collision box to the MoveIt planning scene.

    Protects against the arm hitting the iron base plate that extends beyond
    the robot's base_link collision mesh.  The box sits at z=0 in
    base_footprint frame (= ground level).
    """
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    from std_msgs.msg import Header

    scene_pub = node.create_publisher(PlanningScene, "/planning_scene", 1)
    time.sleep(1.0)

    obj = CollisionObject()
    obj.header = Header()
    obj.header.frame_id = "base_footprint"
    obj.id = "ground_plate"
    obj.operation = CollisionObject.ADD

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [1.0, 1.0, 0.02]

    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = -0.01
    pose.orientation.w = 1.0

    obj.primitives.append(box)
    obj.primitive_poses.append(pose)

    scene = PlanningScene()
    scene.world.collision_objects.append(obj)
    scene.is_diff = True

    scene_pub.publish(scene)
    time.sleep(0.5)
    scene_pub.publish(scene)
    logger.info("Published ground-plate collision object to MoveIt planning scene "
                "(1m x 1m box at z=0)")


def run_test(points, dry_run=False, go_home=True, pause=1.0,
             results_path="x3plus_test_results.json",
             viz_path="x3plus_test_results.png"):
    """Execute the reachability test on a real robot via ROS 2."""
    import rclpy
    from rclpy.node import Node
    from rcl_interfaces.srv import SetParameters
    from std_srvs.srv import Trigger

    rclpy.init()
    node = Node("reachability_test")
    logger = node.get_logger()

    set_params_cli = node.create_client(
        SetParameters, f"{PLANNER_NODE}/set_parameters"
    )
    plan_pos_cli = node.create_client(
        Trigger, f"{PLANNER_NODE}/plan_position"
    )
    go_home_cli = node.create_client(
        Trigger, f"{PLANNER_NODE}/go_home"
    )

    logger.info("Waiting for planner services...")
    if not set_params_cli.wait_for_service(timeout_sec=15.0):
        logger.error(f"Service {PLANNER_NODE}/set_parameters not available. "
                     "Is the planner node running?")
        rclpy.shutdown()
        return
    if not plan_pos_cli.wait_for_service(timeout_sec=5.0):
        logger.error(f"Service {PLANNER_NODE}/plan_position not available.")
        rclpy.shutdown()
        return
    if not go_home_cli.wait_for_service(timeout_sec=5.0):
        logger.error(f"Service {PLANNER_NODE}/go_home not available.")
        rclpy.shutdown()
        return
    logger.info("Planner services connected.")

    _publish_ground_plane(node, logger)

    from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType

    def _verified_set_params(params_list, label=""):
        req = SetParameters.Request()
        req.parameters = params_list
        future = set_params_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if not future.done():
            logger.error(f"set_params{label}: timed out")
            return False
        resp = future.result()
        if resp is None:
            logger.error(f"set_params{label}: no response from planner")
            return False
        for i, r in enumerate(resp.results):
            if not r.successful:
                logger.error(
                    f"set_params{label}: '{params_list[i].name}' failed: {r.reason}"
                )
                return False
        return True

    def _make_bool_param(name, value):
        p = ParamMsg()
        p.name = name
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = bool(value)
        p.value = pv
        return p

    def set_target(x, y, z):
        return _verified_set_params([
            _make_param("target_x", x),
            _make_param("target_y", y),
            _make_param("target_z", z),
        ], " target")

    def _spin_with_progress(future, timeout_sec, label=""):
        deadline = time.time() + timeout_sec
        next_dot = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
            if time.time() >= next_dot:
                elapsed = time.time() - (deadline - timeout_sec)
                print(f"[{elapsed:.0f}s{label}]", end="", flush=True)
                next_dot = time.time() + 5.0
        return future.result()

    def call_plan_position():
        req = Trigger.Request()
        future = plan_pos_cli.call_async(req)
        return _spin_with_progress(future, 60.0, " plan")

    def call_go_home():
        req = Trigger.Request()
        future = go_home_cli.call_async(req)
        return _spin_with_progress(future, 60.0, " home")

    # Safety lockout: set dry_run=True FIRST, before any motion service call
    if dry_run:
        logger.info("Setting dry_run=True on planner (safety lockout)...")
        if not _verified_set_params(
            [_make_bool_param("dry_run", True)], " dry_run"
        ):
            logger.error(
                "CRITICAL: Could not set dry_run on planner! Aborting for safety."
            )
            rclpy.shutdown()
            return
        _verified_set_params(
            [_make_bool_param("execute", False)], " execute"
        )
        logger.info("DRY RUN: dry_run=True + execute=False — arm WILL NOT move")

    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"Starting {mode} test: {len(points)} points, "
                f"go_home={go_home}, pause={pause}s")
    print(f"\n{'='*60}")
    print(f"  X3plus Reachability Test — {mode}")
    print(f"  {len(points)} points, go_home={go_home}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    try:
        for i, (x, y, z) in enumerate(points):
            tag = f"[{i + 1:3d}/{len(points)}]"
            logger.info(f"{tag} Target: ({x:.4f}, {y:.4f}, {z:.4f})")
            print(f"{tag} ({x:.4f}, {y:.4f}, {z:.4f})", end="  ", flush=True)

            set_target(x, y, z)
            t0 = time.time()
            resp = call_plan_position()
            dt = time.time() - t0

            ok = resp is not None and resp.success
            msg = resp.message if resp else "Service call timed out"

            result = {
                "index": i,
                "target_xyz": [round(x, 5), round(y, 5), round(z, 5)],
                "success": ok,
                "message": msg,
                "duration_s": round(dt, 2),
            }
            results.append(result)

            if ok:
                print(f"PASS  ({dt:.1f}s)")
            else:
                print(f"FAIL  ({dt:.1f}s)  {msg}")

            if go_home and not dry_run:
                home_resp = call_go_home()
                if home_resp is None or not home_resp.success:
                    logger.warn(f"  Home return failed: "
                                f"{home_resp.message if home_resp else 'timeout'}")

            if pause > 0:
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")

    elapsed = time.time() - start_time

    if dry_run:
        try:
            _verified_set_params(
                [_make_bool_param("dry_run", False)], " dry_run_cleanup"
            )
            _verified_set_params(
                [_make_bool_param("execute", True)], " execute_cleanup"
            )
            logger.info("dry_run=False, execute=True restored on planner")
        except Exception:
            logger.warning("Could not reset dry_run/execute on planner")

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    # Summary
    n_done = len(results)
    n_pass = sum(1 for r in results if r["success"])
    n_fail = n_done - n_pass

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run" if dry_run else "live",
        "total_points": len(points),
        "completed": n_done,
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": round(n_pass / n_done * 100, 1) if n_done else 0,
        "elapsed_s": round(elapsed, 1),
        "go_home": go_home,
    }

    # Failure breakdown
    fail_reasons = {}
    for r in results:
        if not r["success"]:
            fail_reasons[r["message"]] = fail_reasons.get(r["message"], 0) + 1
    summary["failure_reasons"] = fail_reasons

    report = {"summary": summary, "results": results}

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Completed: {n_done}/{len(points)}")
    print(f"  Passed:    {n_pass}/{n_done}  ({summary['pass_rate']}%)")
    print(f"  Failed:    {n_fail}/{n_done}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/max(n_done,1):.1f}s/point)")
    if fail_reasons:
        print(f"  Failure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:3d}x  {reason}")
    print(f"{'='*60}\n")

    # Visualization with results
    visualize_points(points[:n_done], results, viz_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="X3plus 100-point reachability test"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Generate and visualize test points only (no ROS / robot needed)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only — do not execute motions on the real arm"
    )
    parser.add_argument(
        "--no-home", action="store_true",
        help="Skip returning to home between points (faster, less reproducible)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for point generation (default: 42)"
    )
    parser.add_argument(
        "--num-points", "-n", type=int, default=100,
        help="Number of test points (default: 100)"
    )
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="Seconds to pause between points (default: 1.0)"
    )
    parser.add_argument(
        "--points-file", type=str, default=None,
        help="Load points from a JSON file instead of generating"
    )
    args = parser.parse_args()

    if args.points_file and os.path.exists(args.points_file):
        print(f"Loading points from {args.points_file}")
        with open(args.points_file) as f:
            data = json.load(f)
        points = [tuple(p) for p in data["points"]]
        print(f"Loaded {len(points)} points")
    else:
        print(f"Generating {args.num_points} test points (seed={args.seed})...")
        points, joint_configs = generate_test_points(
            n=args.num_points, seed=args.seed
        )
        print(f"Generated {len(points)} points")

        pts_arr = np.array(points)
        r_xy = np.sqrt(pts_arr[:, 0] ** 2 + pts_arr[:, 1] ** 2)
        print(f"  X: [{pts_arr[:,0].min():.4f}, {pts_arr[:,0].max():.4f}] m")
        print(f"  Y: [{pts_arr[:,1].min():.4f}, {pts_arr[:,1].max():.4f}] m")
        print(f"  Z: [{pts_arr[:,2].min():.4f}, {pts_arr[:,2].max():.4f}] m")
        print(f"  R: [{r_xy.min():.4f}, {r_xy.max():.4f}] m")

        # Save points for reproducibility
        pts_path = "x3plus_test_points.json"
        with open(pts_path, "w") as f:
            json.dump({
                "seed": args.seed,
                "n": len(points),
                "points": [[round(x, 5), round(y, 5), round(z, 5)]
                           for x, y, z in points],
            }, f, indent=2)
        print(f"Points saved -> {pts_path}")

    if args.generate_only:
        visualize_points(points, out_path="x3plus_test_points.png")
        return

    run_test(
        points,
        dry_run=args.dry_run,
        go_home=not args.no_home,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
