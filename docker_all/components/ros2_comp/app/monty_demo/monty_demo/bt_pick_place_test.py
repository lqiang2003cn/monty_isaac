#!/usr/bin/env python3
"""
BT-based pick-and-place test for the X3plus 5-DOF arm.

Mirrors pick_place_test.py but drives the BehaviorTree executor node
(bt_pick_place) instead of calling planner services directly.  Each cycle
sets block/place parameters on the BT node and triggers run_pick_and_place,
which ticks: Pick → go_home transit → Place.

Prerequisites:
  Terminal 1 (from docker_all/):  ./scripts/real_up.sh
  (bt_pick_place_node must be included in the launch)

Usage (Terminal 2, from docker_all/):
  # Full test (100 cycles)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo bt_pick_place_test -n 100"

  # Dry-run (plans everything, arm does NOT move)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo bt_pick_place_test --dry-run -n 10"

  # Generate & visualize candidate poses only (no ROS / robot needed)
  docker compose exec ros2_comp bash -l -c "ros2 run monty_demo bt_pick_place_test --generate-only"
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np

from monty_demo.pick_place_test import (
    generate_candidates,
    visualize_points,
    PLACE_Y_OFFSET,
)

BT_NODE = "/bt_pick_place"
PLANNER_NODE = "/x3plus_5dof_planner"


def _make_double_param(name, value):
    from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
    p = Parameter()
    p.name = name
    pv = ParameterValue()
    pv.type = ParameterType.PARAMETER_DOUBLE
    pv.double_value = float(value)
    p.value = pv
    return p


def _make_bool_param(name, value):
    from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
    p = Parameter()
    p.name = name
    pv = ParameterValue()
    pv.type = ParameterType.PARAMETER_BOOL
    pv.bool_value = bool(value)
    p.value = pv
    return p


def _yaw_to_quat(yaw):
    """Convert yaw (rad) to (qx, qy, qz, qw) quaternion (Z-rotation only)."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def run_test(candidates, num_poses, dry_run=False, pause=0.5,
             results_path="bt_pick_place_test_results.json",
             viz_path="bt_pick_place_test_results.png",
             verbose=False):
    import rclpy
    from rclpy.node import Node
    from rcl_interfaces.srv import SetParameters
    from std_srvs.srv import Trigger
    import logging

    rclpy.init()
    node = Node("bt_pick_place_test")
    logger = node.get_logger()
    if verbose:
        logger.set_level(logging.DEBUG)

    bt_params_cli = node.create_client(
        SetParameters, f"{BT_NODE}/set_parameters")
    bt_run_cli = node.create_client(
        Trigger, f"{BT_NODE}/run_pick_and_place")
    planner_params_cli = node.create_client(
        SetParameters, f"{PLANNER_NODE}/set_parameters")
    go_home_cli = node.create_client(
        Trigger, f"{PLANNER_NODE}/go_home")

    def set_params(cli, params_list):
        req = SetParameters.Request()
        req.parameters = params_list
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return False
        return all(r.successful for r in future.result().results)

    logger.info("Waiting for BT and planner services...")
    for name, cli in [
        (f"{BT_NODE}/set_parameters", bt_params_cli),
        (f"{BT_NODE}/run_pick_and_place", bt_run_cli),
        (f"{PLANNER_NODE}/set_parameters", planner_params_cli),
        (f"{PLANNER_NODE}/go_home", go_home_cli),
    ]:
        if not cli.wait_for_service(timeout_sec=15.0):
            logger.error(f"Service {name} not available")
            rclpy.shutdown()
            return
    logger.info("All services connected.")

    if dry_run:
        logger.info("Setting dry_run=True on BT node and planner...")
        ok_bt = set_params(bt_params_cli, [_make_bool_param("dry_run", True)])
        ok_pl = set_params(
            planner_params_cli, [_make_bool_param("dry_run", True)])
        if not (ok_bt and ok_pl):
            logger.error(
                "CRITICAL: Could not set dry_run! Aborting for safety.")
            rclpy.shutdown()
            return
        logger.info("dry_run=True confirmed — arm WILL NOT move")

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

    def go_home(execute):
        set_params(planner_params_cli, [_make_bool_param("execute", execute)])
        return call_trigger(go_home_cli, timeout=30.0, label=" home")

    do_exec = not dry_run
    mode_label = "DRY-RUN" if dry_run else "LIVE"

    print(f"\n{'='*60}")
    print(f"  X3plus BT Pick-Place Test — {mode_label}")
    print(f"  {num_poses} poses, full pick-place-home cycle via BT")
    print(f"{'='*60}\n")

    print("Going home...", end="  ", flush=True)
    ok, msg = go_home(do_exec)
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
            px, py = bx, by + PLACE_Y_OFFSET

            qx, qy, qz_val, qw_val = _yaw_to_quat(yaw)

            ok_params = set_params(bt_params_cli, [
                _make_double_param("block_x", bx),
                _make_double_param("block_y", by),
                _make_double_param("block_z", bz),
                _make_double_param("block_qx", qx),
                _make_double_param("block_qy", qy),
                _make_double_param("block_qz", qz_val),
                _make_double_param("block_qw", qw_val),
                _make_double_param("place_x", px),
                _make_double_param("place_y", py),
                _make_double_param("place_z", bz),
                _make_double_param("place_qx", qx),
                _make_double_param("place_qy", qy),
                _make_double_param("place_qz", qz_val),
                _make_double_param("place_qw", qw_val),
                _make_bool_param("execute", do_exec),
            ])
            if not ok_params:
                logger.error(f"{tag} Failed to set BT parameters — skipping")

            logger.info(
                f"{tag} pick ({bx:.4f},{by:.4f},{bz:.3f}) yaw={yaw_deg:.1f}°")
            print(
                f"{tag} pick ({bx:.4f},{by:.4f}) yaw={yaw_deg:+6.1f}°",
                end="  ", flush=True)

            t0 = time.time()

            ok, msg = call_trigger(bt_run_cli, timeout=120.0, label=" bt")

            fail_step = "" if ok else "bt_pick_and_place"
            fail_msg = "" if ok else msg

            if ok:
                home_ok, home_msg = go_home(do_exec)
                if not home_ok:
                    ok = False
                    fail_step = "go_home"
                    fail_msg = home_msg

            dt = time.time() - t0
            result = {
                "index": i,
                "block_xyz": [round(bx, 5), round(by, 5), round(bz, 4)],
                "yaw_deg": round(yaw_deg, 1),
                "place_xy": [round(px, 5), round(py, 5)],
                "success": ok,
                "fail_step": fail_step,
                "fail_msg": fail_msg,
                "duration_s": round(dt, 2),
            }
            results.append(result)

            if ok:
                print(f"PASS  ({dt:.1f}s)")
                consecutive_fails = 0
            else:
                print(f"FAIL  ({dt:.1f}s)  [{fail_step}] {fail_msg}")
                consecutive_fails += 1
                go_home(do_exec)

            if consecutive_fails >= max_consecutive_fails:
                print(
                    f"\n{max_consecutive_fails} consecutive failures — aborting.")
                break

            if pause > 0:
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")

    if dry_run:
        try:
            set_params(bt_params_cli, [_make_bool_param("dry_run", False)])
            set_params(
                planner_params_cli, [_make_bool_param("dry_run", False)])
            logger.info("dry_run=False restored")
        except Exception:
            logger.warning("Could not reset dry_run (node may be gone)")

    elapsed = time.time() - start_time

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    n_done = len(results)
    n_pass = sum(1 for r in results if r["success"])
    n_fail = n_done - n_pass

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode_label.lower(),
        "executor": "behavior_tree",
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
    print(f"  Executor:  BehaviorTree (bt_pick_place)")
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


def main():
    parser = argparse.ArgumentParser(
        description="X3plus BT pick-place 100-point test")
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Generate and visualize poses only (no ROS)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only — arm does not move")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    parser.add_argument(
        "--num-poses", "-n", type=int, default=100,
        help="Number of test poses (default: 100)")
    parser.add_argument(
        "--pause", type=float, default=0.5,
        help="Pause between cycles (s, default: 0.5)")
    parser.add_argument(
        "--points-file", type=str, default=None,
        help="Load pre-generated poses from JSON")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging")
    args = parser.parse_args()

    if args.points_file and os.path.exists(args.points_file):
        print(f"Loading poses from {args.points_file}")
        with open(args.points_file) as f:
            data = json.load(f)
        candidates = [(p[0], p[1], p[2], p[3]) for p in data["poses"]]
        print(f"Loaded {len(candidates)} poses")
        if args.generate_only:
            visualize_points(
                candidates, out_path="bt_pick_place_test_points.png")
            return
        run_test(
            candidates, min(args.num_poses, len(candidates)),
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
    print(f"  Yaw: [{np.degrees(yaws.min()):.1f}°, "
          f"{np.degrees(yaws.max()):.1f}°]")

    poses_path = "bt_pick_place_test_points.json"
    with open(poses_path, "w") as f:
        json.dump({
            "n": len(candidates),
            "poses": [[round(x, 5), round(y, 5), round(z, 4), round(yaw, 5)]
                      for x, y, z, yaw in candidates],
        }, f, indent=2)
    print(f"Poses saved -> {poses_path}")

    if args.generate_only:
        visualize_points(
            candidates, out_path="bt_pick_place_test_points.png")
        return

    run_test(
        candidates, min(args.num_poses, len(candidates)),
        dry_run=args.dry_run, pause=args.pause, verbose=args.verbose)


if __name__ == "__main__":
    main()
