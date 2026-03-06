#!/usr/bin/env python3
"""
Straight-line test for the X3plus 5-DOF arm on a real robot.

Generates random kinematically-reachable candidate wrist points, validates
them as a chain using the planner's own dry-run mode, then commands the
real robot to traverse the validated chain sequentially in straight lines:

  home -> point_1 -> point_2 -> ... -> point_N

Validation uses the planner's plan_straight_line service with execute=false,
ensuring every segment is feasible with the planner's exact FK/IK — no
offline approximation that can diverge from the live planner state.

Prerequisites:
  Terminal 1 (from docker_all/):  ./scripts/real_up.sh
    — starts bringup, MoveIt, and the planner automatically.

Usage (Terminal 2, from docker_all/):
  # Full test on real robot (5 segments to start)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo straight_line_test -n 5"

  # Validate only — plans all motions, arm does NOT move
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo straight_line_test --dry-run -n 5"

  # Generate candidate points only (no ROS / robot needed)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo straight_line_test --generate-only"
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


HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0]
HOME_XYZ = fk_wrist(HOME_JOINTS)

# ---------------------------------------------------------------------------
# Analytical IK (cheap pre-filter for candidate generation)
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
# Candidate point generation (offline, no ROS needed)
# ---------------------------------------------------------------------------


def generate_candidates(n=300, seed=42, min_dist=0.018):
    """Sample n random, IK-reachable wrist points with spatial diversity.

    These are *candidates* — chain feasibility is validated later by
    the planner's dry-run mode.  The local IK check is a cheap pre-filter
    so we don't waste planner round-trips on obviously unreachable points.
    """
    rng = np.random.default_rng(seed)
    points = []
    max_attempts = n * 1000

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
        if _best_pitch(x, y, z) is None:
            continue

        too_close = any(
            math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) < min_dist
            for px, py, pz in points
        )
        if too_close:
            continue

        points.append((x, y, z))

    if len(points) < n:
        print(f"WARNING: Only generated {len(points)}/{n} candidates "
              f"(try lowering min_dist or raising max_attempts)")

    return points


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_points(points, results=None, out_path="straight_line_test_points.png"):
    """3D + projected views of the test points with path lines."""
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
    fig.suptitle(f"X3plus Straight-Line Test{title_suffix}",
                 fontsize=14, fontweight="bold")

    def _draw_path(ax, xs, ys, zs=None, result_list=None):
        """Draw path segments colored by result between consecutive points."""
        if result_list is None:
            if zs is not None:
                ax.plot(xs, ys, zs, "-", color="#bdc3c7", linewidth=0.6, alpha=0.5)
            else:
                ax.plot(xs, ys, "-", color="#bdc3c7", linewidth=0.6, alpha=0.5)
        else:
            for i in range(len(result_list)):
                c = "#2ecc71" if result_list[i]["success"] else "#e74c3c"
                a = 0.6 if result_list[i]["success"] else 0.9
                i0, i1 = i, i + 1
                if zs is not None:
                    ax.plot(xs[i0:i1 + 1], ys[i0:i1 + 1], zs[i0:i1 + 1],
                            "-", color=c, linewidth=0.8, alpha=a)
                else:
                    ax.plot(xs[i0:i1 + 1], ys[i0:i1 + 1],
                            "-", color=c, linewidth=0.8, alpha=a)

    # 3D scatter + path
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_path(ax1, pts[:, 0], pts[:, 1], pts[:, 2], results)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    ax1.scatter(*HOME_XYZ, c="gold", s=80, marker="*", edgecolors="k",
                linewidths=0.5, zorder=10, label="Home")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D view")

    # Top (XY)
    ax2 = fig.add_subplot(2, 2, 2)
    _draw_path(ax2, pts[:, 0], pts[:, 1], result_list=results)
    ax2.scatter(pts[:, 0], pts[:, 1], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    ax2.scatter(HOME_XYZ[0], HOME_XYZ[1], c="gold", s=80, marker="*",
                edgecolors="k", linewidths=0.5, zorder=10, label="Home")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top view (XY)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Side (R vs Z)
    r_xy = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    r_home = np.sqrt(HOME_XYZ[0] ** 2 + HOME_XYZ[1] ** 2)
    ax3 = fig.add_subplot(2, 2, 3)
    _draw_path(ax3, r_xy, pts[:, 2], result_list=results)
    ax3.scatter(r_xy, pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    ax3.scatter(r_home, HOME_XYZ[2], c="gold", s=80, marker="*",
                edgecolors="k", linewidths=0.5, zorder=10, label="Home")
    ax3.set_xlabel("Radial distance (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("Side cross-section (R vs Z)")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # Segment distance histogram
    ax4 = fig.add_subplot(2, 2, 4)
    if len(pts) > 1:
        diffs = np.diff(pts, axis=0)
        seg_dists = np.sqrt((diffs ** 2).sum(axis=1)) * 100  # cm
        bar_colors = None
        if results is not None:
            bar_colors = ["#2ecc71" if r["success"] else "#e74c3c"
                          for r in results]
        ax4.bar(range(1, len(seg_dists) + 1), seg_dists, color=bar_colors or "#3498db",
                alpha=0.7, edgecolor="k", linewidth=0.2)
        ax4.set_xlabel("Segment index")
        ax4.set_ylabel("Segment length (cm)")
        ax4.set_title(f"Segment distances (mean {seg_dists.mean():.1f} cm)")
        ax4.grid(True, alpha=0.3, axis="y")
    else:
        ax4.text(0.5, 0.5, "Need ≥ 2 points", ha="center", va="center")

    for ax in [ax2, ax3]:
        if results is not None:
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


def run_test(candidates, num_points, dry_run=False, pause=0.5,
             results_path="straight_line_test_results.json",
             viz_path="straight_line_test_results.png",
             skip_validation=False):
    """Validate candidate points via planner dry-run, then execute."""
    import rclpy
    from rclpy.node import Node
    from rcl_interfaces.srv import SetParameters
    from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
    from std_srvs.srv import Trigger

    rclpy.init()
    node = Node("straight_line_test")
    logger = node.get_logger()

    set_params_cli = node.create_client(
        SetParameters, f"{PLANNER_NODE}/set_parameters"
    )
    plan_pos_cli = node.create_client(
        Trigger, f"{PLANNER_NODE}/plan_position"
    )
    straight_line_cli = node.create_client(
        Trigger, f"{PLANNER_NODE}/plan_straight_line"
    )
    go_home_cli = node.create_client(
        Trigger, f"{PLANNER_NODE}/go_home"
    )

    logger.info("Waiting for planner services...")
    for name, cli in [("set_parameters", set_params_cli),
                      ("plan_position", plan_pos_cli),
                      ("plan_straight_line", straight_line_cli),
                      ("go_home", go_home_cli)]:
        if not cli.wait_for_service(timeout_sec=15.0):
            logger.error(f"Service {PLANNER_NODE}/{name} not available. "
                         "Is the planner node running?")
            rclpy.shutdown()
            return
    logger.info("Planner services connected.")

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

    def set_execute(value):
        req = SetParameters.Request()
        p = ParamMsg()
        p.name = "execute"
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = bool(value)
        p.value = pv
        req.parameters = [p]
        future = set_params_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    def set_target(x, y, z):
        req = SetParameters.Request()
        req.parameters = [
            _make_param("target_x", x),
            _make_param("target_y", y),
            _make_param("target_z", z),
        ]
        future = set_params_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        return future.result()

    def call_plan_position(timeout=60.0):
        future = plan_pos_cli.call_async(Trigger.Request())
        return _spin_with_progress(future, timeout, " plan")

    def call_straight_line(timeout=120.0):
        future = straight_line_cli.call_async(Trigger.Request())
        return _spin_with_progress(future, timeout, " line")

    def call_go_home(timeout=60.0):
        future = go_home_cli.call_async(Trigger.Request())
        return _spin_with_progress(future, timeout, " home")

    # ==================================================================
    # Phase 1: Validate chain via planner dry-run
    # ==================================================================

    if skip_validation:
        validated = list(candidates[:num_points])
        print(f"Loaded {len(validated)} pre-validated points (skipping validation)")
    else:
        print(f"\nValidating chain via planner dry-run "
              f"(target: {num_points} points from {len(candidates)} candidates)...")
        set_execute(False)

        print("  go_home (dry-run)...", end="  ", flush=True)
        home_resp = call_go_home(timeout=30.0)
        if not (home_resp and home_resp.success):
            msg = home_resp.message if home_resp else "timeout"
            print(f"FAILED ({msg})")
            logger.error(f"Dry-run go_home failed: {msg}")
            rclpy.shutdown()
            return
        print("OK")

        validated = []
        skipped = 0
        val_start = time.time()

        for ci, (x, y, z) in enumerate(candidates):
            if len(validated) >= num_points:
                break

            set_target(x, y, z)
            if len(validated) == 0:
                resp = call_plan_position(timeout=30.0)
            else:
                resp = call_straight_line(timeout=10.0)

            if resp and resp.success:
                validated.append((x, y, z))
                tag = f"[{len(validated):3d}/{num_points}]"
                print(f"  {tag} ({x:.4f}, {y:.4f}, {z:.4f})  OK  "
                      f"(candidate {ci + 1})")
            else:
                skipped += 1

        val_elapsed = time.time() - val_start
        print(f"Validated {len(validated)}/{num_points} points in {val_elapsed:.1f}s "
              f"({skipped} candidates skipped out of {ci + 1} tried)")

        if len(validated) < num_points:
            print(f"WARNING: Only validated {len(validated)}/{num_points} points")

    if not validated:
        print("No validated points — nothing to do.")
        rclpy.shutdown()
        return

    # Print stats for the validated set
    pts_arr = np.array(validated)
    r_xy = np.sqrt(pts_arr[:, 0] ** 2 + pts_arr[:, 1] ** 2)
    print(f"  X: [{pts_arr[:,0].min():.4f}, {pts_arr[:,0].max():.4f}] m")
    print(f"  Y: [{pts_arr[:,1].min():.4f}, {pts_arr[:,1].max():.4f}] m")
    print(f"  Z: [{pts_arr[:,2].min():.4f}, {pts_arr[:,2].max():.4f}] m")
    print(f"  R: [{r_xy.min():.4f}, {r_xy.max():.4f}] m")

    if len(pts_arr) > 1:
        diffs = np.diff(pts_arr, axis=0)
        seg_dists_cm = np.sqrt((diffs ** 2).sum(axis=1)) * 100
        print(f"  Segment lengths: mean={seg_dists_cm.mean():.1f} cm, "
              f"min={seg_dists_cm.min():.1f} cm, max={seg_dists_cm.max():.1f} cm")

    # Save validated points
    pts_path = "straight_line_test_points.json"
    with open(pts_path, "w") as f:
        json.dump({
            "n": len(validated),
            "points": [[round(x, 5), round(y, 5), round(z, 5)]
                       for x, y, z in validated],
        }, f, indent=2)
    print(f"Points saved -> {pts_path}")

    if dry_run:
        print("\nDRY RUN complete — all segments validated, arm did not move.")
        visualize_points(validated, out_path=viz_path)
        try:
            set_execute(True)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        return

    # ==================================================================
    # Phase 2: Execute validated chain on real robot
    # ==================================================================

    set_execute(True)
    logger.info("Going home before execution...")
    print("\nGoing home...", end="  ", flush=True)
    home_resp = call_go_home()
    if home_resp and home_resp.success:
        print("OK")
    else:
        msg = home_resp.message if home_resp else "timeout"
        print(f"FAILED ({msg})")
        logger.warn(f"go_home failed: {msg} — continuing anyway")
    time.sleep(1.0)

    n_pts = len(validated)
    logger.info(f"Starting LIVE straight-line test: {n_pts} points")
    print(f"\n{'='*60}")
    print(f"  X3plus Straight-Line Test — LIVE")
    print(f"  {n_pts} points, sequential straight-line moves")
    print(f"{'='*60}\n")

    seg_dists = []
    for i in range(1, n_pts):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(validated[i], validated[i - 1])))
        seg_dists.append(d)

    results = []
    start_time = time.time()
    consecutive_fails = 0
    max_consecutive_fails = 5

    try:
        for i, (x, y, z) in enumerate(validated):
            tag = f"[{i + 1:3d}/{n_pts}]"

            if i == 0:
                src = "home"
                method = "moveit"
                dist_cm = math.sqrt(
                    sum((a - b) ** 2 for a, b
                        in zip(validated[0], (float(HOME_XYZ[0]),
                                              float(HOME_XYZ[1]),
                                              float(HOME_XYZ[2]))))
                ) * 100
            else:
                src = f"pt{i}"
                method = "line"
                dist_cm = seg_dists[i - 1] * 100

            logger.info(f"{tag} [{method}] {src} -> ({x:.4f}, {y:.4f}, {z:.4f})  "
                        f"[{dist_cm:.1f} cm]")
            print(f"{tag} [{method}] {src} -> ({x:.4f}, {y:.4f}, {z:.4f})  "
                  f"[{dist_cm:.1f} cm]", end="  ", flush=True)

            set_target(x, y, z)
            t0 = time.time()
            if i == 0:
                resp = call_plan_position()
            else:
                resp = call_straight_line()
            dt = time.time() - t0

            ok = resp is not None and resp.success
            msg = resp.message if resp else "Service call timed out"

            result = {
                "index": i,
                "from": "home" if i == 0 else [round(c, 5) for c in validated[i - 1]],
                "target_xyz": [round(x, 5), round(y, 5), round(z, 5)],
                "distance_cm": round(dist_cm, 2),
                "success": ok,
                "message": msg,
                "duration_s": round(dt, 2),
            }
            results.append(result)

            if ok:
                print(f"PASS  ({dt:.1f}s)")
                consecutive_fails = 0
            else:
                print(f"FAIL  ({dt:.1f}s)  {msg}")
                consecutive_fails += 1

            if consecutive_fails >= max_consecutive_fails:
                print(f"\n{max_consecutive_fails} consecutive failures — "
                      "aborting to prevent damage.")
                logger.error(f"Aborting after {max_consecutive_fails} "
                             "consecutive failures")
                break

            if pause > 0:
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")

    elapsed = time.time() - start_time

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
    total_dist = sum(r["distance_cm"] for r in results)
    pass_dist = sum(r["distance_cm"] for r in results if r["success"])

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "total_points": n_pts,
        "completed": n_done,
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": round(n_pass / n_done * 100, 1) if n_done else 0,
        "total_distance_cm": round(total_dist, 1),
        "passed_distance_cm": round(pass_dist, 1),
        "elapsed_s": round(elapsed, 1),
    }

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
    print(f"  Completed: {n_done}/{n_pts} segments")
    print(f"  Passed:    {n_pass}/{n_done}  ({summary['pass_rate']}%)")
    print(f"  Failed:    {n_fail}/{n_done}")
    print(f"  Distance:  {total_dist:.1f} cm total, {pass_dist:.1f} cm passed")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/max(n_done,1):.1f}s/segment)")
    if fail_reasons:
        print(f"  Failure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:3d}x  {reason}")
    print(f"{'='*60}\n")

    visualize_points(validated[:n_done], results, viz_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="X3plus straight-line test"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Generate and visualize candidate points only (no ROS / robot needed)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate via planner dry-run only — arm does not move"
    )
    parser.add_argument(
        "--seed", type=int, default=99,
        help="Random seed for candidate generation (default: 99)"
    )
    parser.add_argument(
        "--num-points", "-n", type=int, default=100,
        help="Number of test points (default: 100)"
    )
    parser.add_argument(
        "--pause", type=float, default=0.5,
        help="Seconds to pause between segments during execution (default: 0.5)"
    )
    parser.add_argument(
        "--points-file", type=str, default=None,
        help="Load pre-validated points from a JSON file (skips generation + validation)"
    )
    args = parser.parse_args()

    if args.points_file and os.path.exists(args.points_file):
        print(f"Loading points from {args.points_file}")
        with open(args.points_file) as f:
            data = json.load(f)
        points = [tuple(p) for p in data["points"]]
        print(f"Loaded {len(points)} points")
        run_test(
            points, len(points),
            dry_run=args.dry_run,
            pause=args.pause,
            skip_validation=True,
        )
        return

    n_candidates = max(args.num_points * 5, args.num_points + 50)
    print(f"Generating {n_candidates} candidate points (seed={args.seed})...")
    candidates = generate_candidates(n=n_candidates, seed=args.seed)
    print(f"Generated {len(candidates)} candidates")

    if args.generate_only:
        pts_arr = np.array(candidates)
        r_xy = np.sqrt(pts_arr[:, 0] ** 2 + pts_arr[:, 1] ** 2)
        print(f"  X: [{pts_arr[:,0].min():.4f}, {pts_arr[:,0].max():.4f}] m")
        print(f"  Y: [{pts_arr[:,1].min():.4f}, {pts_arr[:,1].max():.4f}] m")
        print(f"  Z: [{pts_arr[:,2].min():.4f}, {pts_arr[:,2].max():.4f}] m")
        print(f"  R: [{r_xy.min():.4f}, {r_xy.max():.4f}] m")
        visualize_points(candidates, out_path="straight_line_test_points.png")
        return

    run_test(
        candidates, args.num_points,
        dry_run=args.dry_run,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
