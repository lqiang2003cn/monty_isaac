#!/usr/bin/env python3
"""
100-point pick-and-place test for the X3plus 5-DOF arm on a real robot.

Generates 6DOF block surface poses (position + yaw) within the gripper-down
reachable workspace, validates each one, then runs the full pick → place → home
cycle at every pose — exercising gripper open/close/rotate even without a real
block.

Prerequisites:
  Terminal 1 (from docker_all/):  ./scripts/real_up.sh

Usage (Terminal 2, from docker_all/):
  # Full test on real robot (100 cycles)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo pick_place_test -n 100"

  # Dry-run (plans everything, arm does NOT move)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo pick_place_test --dry-run -n 10"

  # Generate & visualize candidate poses only (no ROS / robot needed)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo pick_place_test --generate-only"
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
# Kinematic constants (matching x3plus_5dof_planner.py)
# ---------------------------------------------------------------------------

L1 = 0.0829
L2 = 0.0829
L3 = 0.17455
L3X = 0.00215
BASE_XY = 0.09825
BASE_Z = 0.076 + 0.102 + 0.0405

JOINT_LIMITS = [
    (-1.5708, 1.5708),   # arm_joint1
    (-1.5708, 1.5708),   # arm_joint2
    (-1.5708, 1.5708),   # arm_joint3
    (-1.5708, 1.5708),   # arm_joint4
    (-1.5708, 3.14159),  # arm_joint5
]

PITCH_DOWN = -math.pi / 2
J5_LO, J5_HI = JOINT_LIMITS[4]

FINGERTIP_BELOW_WRIST = 0.012
MIN_FLOOR_CLEARANCE = 0.005
BLOCK_HEIGHT = 0.03
GRASP_DEPTH = 0.015
APPROACH_HEIGHT = 0.04
GRIP_OPEN = 0.0
GRIP_CLOSED = -1.3
PLACE_Y_OFFSET = 0.03

PLANNER_NODE = "/x3plus_5dof_planner"

# ---------------------------------------------------------------------------
# Analytical IK (local copy for offline candidate generation)
# ---------------------------------------------------------------------------


def _in_limits(joint_idx, val):
    lo, hi = JOINT_LIMITS[joint_idx]
    return lo - 1e-6 <= val <= hi + 1e-6


def analytical_ik(x, y, z, pitch, roll=0.0):
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


def forward_kin(qs):
    """FK to wrist point (arm_link5 origin) — 4x4 matrix via chain."""
    def _tf(xyz, rpy=(0, 0, 0)):
        c, s = np.cos(rpy[1]), np.sin(rpy[1])
        ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        rx = np.array([[1, 0, 0, 0], [0, cr, -sr, 0], [0, sr, cr, 0], [0, 0, 0, 1]])
        cz, sz = np.cos(rpy[2]), np.sin(rpy[2])
        rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T = rz @ ry @ rx
        T[:3, 3] = xyz
        return T

    def _joint(xyz, rpy, axis, q):
        T = _tf(xyz, rpy)
        ax = np.array(axis, dtype=float)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
        Tr = np.eye(4); Tr[:3, :3] = R
        return T @ Tr

    chain = [
        ([0.09825, 0, 0.102],     [0, 0, 0],       [0, 0, -1]),
        ([0, 0, 0.0405],          [-1.5708, 0, 0],  [0, 0, -1]),
        ([0, -0.0829, 0],         [0, 0, 0],        [0, 0, -1]),
        ([0, -0.0829, 0],         [0, 0, 0],        [0, 0, -1]),
        ([-0.00215, -0.17455, 0], [1.5708, 0, 0],   [0, 0, 1]),
    ]
    T = _tf([0, 0, 0.076])
    for i, (xyz, rpy, axis) in enumerate(chain):
        T = T @ _joint(xyz, rpy, axis, qs[i])
    return T[:3, 3]


from monty_demo.opus_plan_and_imp.opus_joint_config import INIT_ARM_POSITIONS
HOME_JOINTS = list(INIT_ARM_POSITIONS)
HOME_XYZ = forward_kin(HOME_JOINTS)


def _is_reachable_offline(x, y, z_wrist, roll, approach_height=APPROACH_HEIGHT):
    """Offline reachability check (no ROS)."""
    z_fingertip = z_wrist - FINGERTIP_BELOW_WRIST
    if z_fingertip < MIN_FLOOR_CLEARANCE:
        return False, "floor collision"
    if analytical_ik(x, y, z_wrist, PITCH_DOWN, roll) is None:
        return False, "IK fail at grasp"
    z_app = z_wrist + approach_height
    if analytical_ik(x, y, z_app, PITCH_DOWN, roll) is None:
        return False, "IK fail at approach"
    return True, ""


# ---------------------------------------------------------------------------
# Candidate generation (offline, no ROS)
# ---------------------------------------------------------------------------


def generate_candidates(n=100, seed=42, min_dist=0.015):
    """Sample n 6DOF block poses within the gripper-down pickable workspace.

    Each candidate is (x, y, z_block, yaw) where z_block = BLOCK_HEIGHT.
    """
    rng = np.random.default_rng(seed)
    z_block = BLOCK_HEIGHT
    z_wrist_grasp = z_block - GRASP_DEPTH + FINGERTIP_BELOW_WRIST

    candidates = []
    max_attempts = n * 2000

    for _ in range(max_attempts):
        if len(candidates) >= n:
            break

        # Random arm config with gripper-down constraint
        q2 = float(rng.uniform(*JOINT_LIMITS[1]))
        q3 = float(rng.uniform(*JOINT_LIMITS[2]))
        q4 = -math.pi - q2 - q3
        if abs(q4) > JOINT_LIMITS[3][1] + 1e-6:
            continue

        q1 = float(rng.uniform(*JOINT_LIMITS[0]))
        yaw = float(rng.uniform(J5_LO, J5_HI))

        qs = [q1, q2, q3, q4, yaw]
        wrist = forward_kin(qs)
        x, y, z_w = float(wrist[0]), float(wrist[1]), float(wrist[2])

        # Accept only if the wrist height roughly matches the needed grasp height
        if abs(z_w - z_wrist_grasp) > 0.02:
            continue

        ok, _ = _is_reachable_offline(x, y, z_wrist_grasp, yaw)
        if not ok:
            continue

        # Also verify the place position is reachable
        ok_p, _ = _is_reachable_offline(x, y + PLACE_Y_OFFSET, z_wrist_grasp, yaw)
        if not ok_p:
            continue

        too_close = any(
            math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < min_dist
            for cx, cy, _, _ in candidates
        )
        if too_close:
            continue

        candidates.append((x, y, z_block, yaw))

    if len(candidates) < n:
        print(f"WARNING: Only generated {len(candidates)}/{n} candidates "
              f"(try lowering min_dist or raising max_attempts)")

    return candidates


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_points(candidates, results=None, out_path="pick_place_test_results.png"):
    if not candidates:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.array([(x, y, z) for x, y, z, _ in candidates])
    yaws = np.array([yaw for _, _, _, yaw in candidates])

    if results is not None:
        colors = ["#2ecc71" if r["success"] else "#e74c3c" for r in results]
        n_ok = sum(1 for r in results if r["success"])
        title_suffix = f"  ({n_ok} passed, {len(results) - n_ok} failed)"
    else:
        colors = "#3498db"
        title_suffix = ""

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"X3plus Pick-Place Test{title_suffix}",
                 fontsize=14, fontweight="bold")

    # 3D view with yaw arrows
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    arrow_len = 0.01
    for i, (x, y, z, yaw) in enumerate(candidates):
        c = colors[i] if isinstance(colors, list) else colors
        ax1.quiver(x, y, z, arrow_len * math.cos(yaw), arrow_len * math.sin(yaw), 0,
                   color=c, alpha=0.6, arrow_length_ratio=0.3)
    ax1.scatter(*HOME_XYZ, c="gold", s=80, marker="*", edgecolors="k",
                linewidths=0.5, zorder=10, label="Home")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D view (arrows = block yaw)")

    # Top XY
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(pts[:, 0], pts[:, 1], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    for i, (x, y, z, yaw) in enumerate(candidates):
        c = colors[i] if isinstance(colors, list) else colors
        ax2.arrow(x, y, arrow_len * math.cos(yaw), arrow_len * math.sin(yaw),
                  head_width=0.002, color=c, alpha=0.6)
    ax2.scatter(HOME_XYZ[0], HOME_XYZ[1], c="gold", s=80, marker="*",
                edgecolors="k", linewidths=0.5, zorder=10, label="Home")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top view (XY)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Side R vs Z
    r_xy = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    r_home = np.sqrt(HOME_XYZ[0] ** 2 + HOME_XYZ[1] ** 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(r_xy, pts[:, 2], c=colors, s=25, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    ax3.scatter(r_home, HOME_XYZ[2], c="gold", s=80, marker="*",
                edgecolors="k", linewidths=0.5, zorder=10, label="Home")
    ax3.set_xlabel("Radial distance (m)")
    ax3.set_ylabel("Z block surface (m)")
    ax3.set_title("Side cross-section (R vs Z)")
    ax3.grid(True, alpha=0.3)

    # Yaw histogram
    ax4 = fig.add_subplot(2, 2, 4)
    yaws_deg = np.degrees(yaws[:len(results) if results else len(yaws)])
    if results is not None:
        ok_yaws = [math.degrees(candidates[i][3]) for i, r in enumerate(results) if r["success"]]
        fail_yaws = [math.degrees(candidates[i][3]) for i, r in enumerate(results) if not r["success"]]
        bins = np.linspace(-90, 180, 28)
        if ok_yaws:
            ax4.hist(ok_yaws, bins=bins, color="#2ecc71", alpha=0.7, label="Pass")
        if fail_yaws:
            ax4.hist(fail_yaws, bins=bins, color="#e74c3c", alpha=0.7, label="Fail")
        ax4.legend()
    else:
        ax4.hist(yaws_deg, bins=28, color="#3498db", alpha=0.7)
    ax4.set_xlabel("Block yaw (°)")
    ax4.set_ylabel("Count")
    ax4.set_title("Yaw distribution")
    ax4.grid(True, alpha=0.3, axis="y")

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


def _make_double_param(name, value):
    from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
    p = ParamMsg()
    p.name = name
    pv = ParameterValue()
    pv.type = ParameterType.PARAMETER_DOUBLE
    pv.double_value = float(value)
    p.value = pv
    return p


def _make_bool_param(name, value):
    from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
    p = ParamMsg()
    p.name = name
    pv = ParameterValue()
    pv.type = ParameterType.PARAMETER_BOOL
    pv.bool_value = bool(value)
    p.value = pv
    return p


def run_test(candidates, num_poses, dry_run=False, pause=0.5,
             results_path="pick_place_test_results.json",
             viz_path="pick_place_test_results.png",
             verbose=False):
    """Run the full pick-place-home cycle for each candidate pose."""
    import rclpy
    from rclpy.node import Node as RclNode
    from rcl_interfaces.srv import SetParameters
    from std_srvs.srv import Trigger
    import logging

    rclpy.init()
    node = RclNode("pick_place_test")
    logger = node.get_logger()
    if verbose:
        logger.set_level(logging.DEBUG)

    set_params_cli = node.create_client(SetParameters, f"{PLANNER_NODE}/set_parameters")
    set_gripper_cli = node.create_client(Trigger, f"{PLANNER_NODE}/set_gripper")
    plan_orient_cli = node.create_client(Trigger, f"{PLANNER_NODE}/plan_position_orient")
    straight_line_cli = node.create_client(Trigger, f"{PLANNER_NODE}/plan_straight_line")
    go_home_cli = node.create_client(Trigger, f"{PLANNER_NODE}/go_home")

    def set_params(params_list):
        from rcl_interfaces.srv import SetParameters as SP
        req = SP.Request()
        req.parameters = params_list
        future = set_params_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if not future.done():
            logger.error("set_params: timed out — parameters may not be set!")
            return False
        resp = future.result()
        if resp is None:
            logger.error("set_params: no response from planner")
            return False
        for i, r in enumerate(resp.results):
            if not r.successful:
                logger.error(
                    f"set_params: '{params_list[i].name}' failed: {r.reason}"
                )
                return False
        return True

    logger.info("Waiting for planner services...")
    if not set_params_cli.wait_for_service(timeout_sec=15.0):
        logger.error(f"Service {PLANNER_NODE}/set_parameters not available")
        rclpy.shutdown()
        return

    if dry_run:
        logger.info("Setting dry_run=True on planner (safety lockout)...")
        ok = set_params([_make_bool_param("dry_run", True)])
        if not ok:
            logger.error(
                "CRITICAL: Could not set dry_run on planner! Aborting for safety."
            )
            rclpy.shutdown()
            return
        logger.info("dry_run=True confirmed on planner — arm WILL NOT move")

    for name, cli in [("set_gripper", set_gripper_cli),
                      ("plan_position_orient", plan_orient_cli),
                      ("plan_straight_line", straight_line_cli),
                      ("go_home", go_home_cli)]:
        if not cli.wait_for_service(timeout_sec=15.0):
            logger.error(f"Service {PLANNER_NODE}/{name} not available")
            rclpy.shutdown()
            return
    logger.info("All planner services connected.")

    def spin_call(future, timeout_sec=60.0, label=""):
        deadline = time.time() + timeout_sec
        next_dot = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
            if time.time() >= next_dot:
                elapsed = time.time() - (deadline - timeout_sec)
                print(f"[{elapsed:.0f}s{label}]", end="", flush=True)
                next_dot = time.time() + 5.0
        return future.result()

    def call_trigger(cli, timeout=60.0, label=""):
        future = cli.call_async(Trigger.Request())
        resp = spin_call(future, timeout, label)
        if resp is None:
            logger.debug(f"call_trigger{label}: timeout")
            return False, "timeout"
        logger.debug(f"call_trigger{label}: ok={resp.success} msg={resp.message}")
        return resp.success, resp.message

    def step_set_gripper(grip_val, execute):
        logger.debug(f"step_set_gripper: grip={grip_val:.3f} execute={execute}")
        set_params([_make_double_param("target_grip", grip_val),
                    _make_bool_param("execute", execute)])
        return call_trigger(set_gripper_cli, timeout=15.0, label=" grip")

    def step_move_above(x, y, z, roll, execute):
        logger.debug(
            f"step_move_above: ({x:.4f},{y:.4f},{z:.4f}) "
            f"pitch={math.degrees(PITCH_DOWN):.1f}° roll={math.degrees(roll):.1f}° "
            f"execute={execute}"
        )
        set_params([_make_double_param("target_x", x),
                    _make_double_param("target_y", y),
                    _make_double_param("target_z", z),
                    _make_double_param("target_pitch", PITCH_DOWN),
                    _make_double_param("target_roll", roll),
                    _make_bool_param("execute", execute)])
        return call_trigger(plan_orient_cli, timeout=30.0, label=" move")

    def step_straight_line(x, y, z, execute):
        logger.debug(f"step_straight_line: ({x:.4f},{y:.4f},{z:.4f}) execute={execute}")
        set_params([_make_double_param("target_x", x),
                    _make_double_param("target_y", y),
                    _make_double_param("target_z", z),
                    _make_bool_param("execute", execute)])
        return call_trigger(straight_line_cli, timeout=60.0, label=" line")

    def step_go_home(execute):
        logger.debug(f"step_go_home: execute={execute}")
        set_params([_make_bool_param("execute", execute)])
        return call_trigger(go_home_cli, timeout=30.0, label=" home")

    # ── Execution ──

    do_exec = not dry_run
    mode_label = "DRY-RUN" if dry_run else "LIVE"

    # Go home first
    print(f"\n{'='*60}")
    print(f"  X3plus Pick-Place Test — {mode_label}")
    print(f"  {num_poses} poses, full pick-place-home cycle")
    print(f"{'='*60}\n")

    print("Going home...", end="  ", flush=True)
    ok, msg = step_go_home(do_exec)
    print("OK" if ok else f"FAIL ({msg})")
    if not ok and not dry_run:
        logger.error(f"Initial go_home failed: {msg}")
    time.sleep(0.5)

    results = []
    start_time = time.time()
    consecutive_fails = 0
    max_consecutive_fails = 5

    try:
        for i, (bx, by, bz, yaw) in enumerate(candidates[:num_poses]):
            tag = f"[{i + 1:3d}/{num_poses}]"
            yaw_deg = math.degrees(yaw)
            z_wrist = bz - GRASP_DEPTH + FINGERTIP_BELOW_WRIST
            z_approach = z_wrist + APPROACH_HEIGHT
            px, py = bx, by + PLACE_Y_OFFSET

            logger.info(f"{tag} pick ({bx:.4f},{by:.4f},{bz:.3f}) yaw={yaw_deg:.1f}°")
            print(f"{tag} pick ({bx:.4f},{by:.4f}) yaw={yaw_deg:+6.1f}°", end="  ", flush=True)

            t0 = time.time()
            cycle_ok = True
            fail_step = ""
            fail_msg = ""

            # -- PICK sequence --
            pick_steps = [
                ("open",   lambda: step_set_gripper(GRIP_OPEN, do_exec)),
                ("above",  lambda: step_move_above(bx, by, z_approach, yaw, do_exec)),
                ("down",   lambda: step_straight_line(bx, by, z_wrist, do_exec)),
                ("close",  lambda: step_set_gripper(GRIP_CLOSED, do_exec)),
                ("up",     lambda: step_straight_line(bx, by, z_approach, do_exec)),
            ]
            for sname, sfn in pick_steps:
                ok, msg = sfn()
                if not ok:
                    cycle_ok = False
                    fail_step = f"pick/{sname}"
                    fail_msg = msg
                    break

            # -- PLACE sequence (only if pick succeeded) --
            if cycle_ok:
                place_steps = [
                    ("above",  lambda: step_move_above(px, py, z_approach, yaw, do_exec)),
                    ("down",   lambda: step_straight_line(px, py, z_wrist, do_exec)),
                    ("open",   lambda: step_set_gripper(GRIP_OPEN, do_exec)),
                    ("up",     lambda: step_straight_line(px, py, z_approach, do_exec)),
                ]
                for sname, sfn in place_steps:
                    ok, msg = sfn()
                    if not ok:
                        cycle_ok = False
                        fail_step = f"place/{sname}"
                        fail_msg = msg
                        break

            # -- GO HOME --
            if cycle_ok:
                ok, msg = step_go_home(do_exec)
                if not ok:
                    cycle_ok = False
                    fail_step = "go_home"
                    fail_msg = msg

            dt = time.time() - t0
            result = {
                "index": i,
                "block_xyz": [round(bx, 5), round(by, 5), round(bz, 4)],
                "yaw_deg": round(yaw_deg, 1),
                "place_xy": [round(px, 5), round(py, 5)],
                "success": cycle_ok,
                "fail_step": fail_step,
                "fail_msg": fail_msg,
                "duration_s": round(dt, 2),
            }
            results.append(result)

            if cycle_ok:
                print(f"PASS  ({dt:.1f}s)")
                consecutive_fails = 0
            else:
                print(f"FAIL  ({dt:.1f}s)  [{fail_step}] {fail_msg}")
                consecutive_fails += 1
                # Try to go home after failure
                step_go_home(do_exec)

            if consecutive_fails >= max_consecutive_fails:
                print(f"\n{max_consecutive_fails} consecutive failures — aborting.")
                break

            if pause > 0:
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")

    if dry_run:
        try:
            set_params([_make_bool_param("dry_run", False)])
            logger.info("dry_run=False restored on planner")
        except Exception:
            logger.warning("Could not reset dry_run on planner (node may be gone)")

    elapsed = time.time() - start_time

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    # ── Summary ──
    n_done = len(results)
    n_pass = sum(1 for r in results if r["success"])
    n_fail = n_done - n_pass

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode_label.lower(),
        "total_poses": num_poses,
        "completed": n_done,
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": round(n_pass / n_done * 100, 1) if n_done else 0,
        "elapsed_s": round(elapsed, 1),
    }

    fail_reasons = {}
    for r in results:
        if not r["success"]:
            key = f"{r['fail_step']}: {r['fail_msg']}"
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    summary["failure_reasons"] = fail_reasons

    report = {"summary": summary, "results": results}
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Completed: {n_done}/{num_poses} cycles")
    print(f"  Passed:    {n_pass}/{n_done}  ({summary['pass_rate']}%)")
    print(f"  Failed:    {n_fail}/{n_done}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed / max(n_done, 1):.1f}s/cycle)")
    if fail_reasons:
        print(f"  Failure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:3d}x  {reason}")
    print(f"{'='*60}\n")

    visualize_points(candidates[:n_done], results, viz_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="X3plus pick-place 100-point test")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate and visualize poses only (no ROS)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only — arm does not move")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--num-poses", "-n", type=int, default=100,
                        help="Number of test poses (default: 100)")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Pause between cycles (s, default: 0.5)")
    parser.add_argument("--points-file", type=str, default=None,
                        help="Load pre-generated poses from JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug-level logging for service calls and responses")
    args = parser.parse_args()

    if args.points_file and os.path.exists(args.points_file):
        print(f"Loading poses from {args.points_file}")
        with open(args.points_file) as f:
            data = json.load(f)
        candidates = [(p[0], p[1], p[2], p[3]) for p in data["poses"]]
        print(f"Loaded {len(candidates)} poses")
        if args.generate_only:
            visualize_points(candidates, out_path="pick_place_test_points.png")
            return
        run_test(candidates, min(args.num_poses, len(candidates)),
                 dry_run=args.dry_run, pause=args.pause, verbose=args.verbose)
        return

    n_candidates = max(args.num_poses * 3, args.num_poses + 50)
    print(f"Generating {n_candidates} candidate poses (seed={args.seed})...")
    candidates = generate_candidates(n=n_candidates, seed=args.seed)
    print(f"Generated {len(candidates)} valid candidates")

    if not candidates:
        print("ERROR: No valid candidates generated.")
        return

    pts = np.array([(x, y) for x, y, _, _ in candidates])
    yaws = np.array([yaw for _, _, _, yaw in candidates])
    print(f"  X: [{pts[:, 0].min():.4f}, {pts[:, 0].max():.4f}] m")
    print(f"  Y: [{pts[:, 1].min():.4f}, {pts[:, 1].max():.4f}] m")
    print(f"  Yaw: [{np.degrees(yaws.min()):.1f}°, {np.degrees(yaws.max()):.1f}°]")

    # Save candidates
    poses_path = "pick_place_test_points.json"
    with open(poses_path, "w") as f:
        json.dump({
            "n": len(candidates),
            "poses": [[round(x, 5), round(y, 5), round(z, 4), round(yaw, 5)]
                      for x, y, z, yaw in candidates],
        }, f, indent=2)
    print(f"Poses saved -> {poses_path}")

    if args.generate_only:
        visualize_points(candidates, out_path="pick_place_test_points.png")
        return

    run_test(candidates, min(args.num_poses, len(candidates)),
             dry_run=args.dry_run, pause=args.pause, verbose=args.verbose)


if __name__ == "__main__":
    main()
