#!/usr/bin/env python3
"""
ZMQ X3plus robot bridge: subscribes to /x3plus/joint_commands, sends position
commands to hardware via ZMQ (remote_real_x3plus on Orin); reads positions and
publishes /x3plus/joint_states. Run with ros2_control bringup (mode:=zmq).

Uses opus_joint_config for rad/deg conversion (shared with real bridge).
"""

import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
import zmq

from monty_demo.opus_plan_and_imp.opus_joint_config import (
    ARM_GRIP_JOINTS,
    JOINT_NAMES,
    MIMIC_MAP,
    SERVO_MAP,
    rad_to_deg,
    deg_to_rad,
)
from monty_demo.opus_plan_and_imp.lqtech_zmq_client import LqtechZMQClient
from monty_demo.opus_plan_and_imp.log_utils import LOG_DIR, make_file_logger
from monty_demo.opus_plan_and_imp.zmq_protocol import GET_JOINT_POSITION_ARRAY

JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

SERVO_ORDER = ARM_GRIP_JOINTS  # [arm_joint1, ..., arm_joint5, grip_joint]


def probe_connection(host: str, port: int, timeout_ms: int = 5000) -> bool:
    """Try one getJointPositionArray request; return True if service responds."""
    ctx = None
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect("tcp://%s:%d" % (host, port))
        sock.send_string(json.dumps({"method": GET_JOINT_POSITION_ARRAY}))
        resp = sock.recv_string()
        data = json.loads(resp)
        if "error" in data:
            return False
        if "joint_array" not in data or len(data["joint_array"]) != 6:
            return False
        return True
    except Exception:
        return False
    finally:
        if ctx is not None:
            try:
                ctx.destroy(linger=0)
            except Exception:
                pass


def ensure_zmq_connection(host: str, port: int, log_fn) -> None:
    """Probe ZMQ service with retries. Exit process if connection fails."""
    timeout_ms = int(os.environ.get("ZMQ_PROBE_TIMEOUT_MS", "5000"))
    retry_count = int(os.environ.get("ZMQ_RETRY_COUNT", "5"))
    retry_delay = float(os.environ.get("ZMQ_RETRY_DELAY_SEC", "3"))

    for attempt in range(1, retry_count + 1):
        log_fn("[%d/%d] Probing ZMQ service at %s:%d ..." % (attempt, retry_count, host, port))
        if probe_connection(host, port, timeout_ms):
            log_fn("ZMQ service at %s:%d is reachable." % (host, port))
            return
        log_fn("Probe failed (connection refused or timeout).")
        if attempt < retry_count:
            log_fn("Retrying in %s s ..." % retry_delay)
            time.sleep(retry_delay)

    log_fn("")
    log_fn("FATAL: Could not reach ZMQ service at %s:%d after %d attempts." % (host, port, retry_count))
    log_fn("  - Ensure the ZMQ service (e.g. remote_real_x3plus container) is running on the robot.")
    log_fn("  - Check connectivity: nc -zv %s %d" % (host, port))
    log_fn("")
    sys.exit(1)


WATCHDOG_FAIL_THRESHOLD = 5
WATCHDOG_LOCKOUT_SEC = 10.0

DEVIATION_WARN_DEG = 5.0
DEVIATION_CRIT_DEG = 10.0
DEVIATION_LOCKOUT_DEG = 30.0
DEVIATION_LOCKOUT_COUNT = 3

# During active trajectory execution, servos naturally lag behind the rapidly
# advancing command target.  Use relaxed thresholds so normal lag doesn't spam
# the console; the LOCKOUT threshold (30°) is unchanged and still catches
# genuinely stuck servos.
DEVIATION_ACTIVE_WARN_DEG = 15.0
DEVIATION_ACTIVE_CRIT_DEG = 25.0
TRAJECTORY_ACTIVE_TIMEOUT_SEC = 1.0

INVALID_POS_LOCKOUT_COUNT = 5
MAX_PLAUSIBLE_JUMP_DEG = 30.0
STARTUP_CONFIRM_READS = 2

# Grace period after the first command is sent.  Servos need time to travel from
# their power-on position to the init/home position; deviation lockout during
# this window would false-trigger every time the robot starts far from home.
STARTUP_GRACE_SEC = 8.0

BATTERY_POLL_SEC = 30.0
BATTERY_WARN_V = 6.5
BATTERY_CRIT_V = 6.0


class OpusX3PlusZMQBridge(Node):
    def __init__(self) -> None:
        super().__init__("opus_x3plus_zmq_bridge")
        self.declare_parameter("zmq_host", os.environ.get("ZMQ_HOST", "192.168.31.142"))
        self.declare_parameter("zmq_port", int(os.environ.get("ZMQ_PORT", "5555")))
        self.declare_parameter("state_publish_hz", 20.0)

        host = self.get_parameter("zmq_host").get_parameter_value().string_value
        port = self.get_parameter("zmq_port").get_parameter_value().integer_value

        self._flog = make_file_logger(
            "zmq_bridge", f"{LOG_DIR}/x3plus_zmq_bridge.log", backup_count=19,
        )
        self._flog.info("=" * 60)
        self._flog.info("ZMQ bridge starting, host=%s:%d" % (host, port))

        self.get_logger().info("ZMQ bridge connecting to %s:%d" % (host, port))
        self._zmq_client = LqtechZMQClient(host=host, port=port)

        state_hz = self.get_parameter("state_publish_hz").get_parameter_value().double_value

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
        self._state_timer = self.create_timer(1.0 / state_hz, self._control_loop)
        self._cmd_count = 0
        self._state_log_count = 0

        self._pending_cmd_degs = None
        self._last_cmd_send_time = 0.0
        self._consecutive_cmd_fails = 0
        self._consecutive_state_fails = 0
        self._cmd_lockout_until = 0.0

        self._last_cmd_deg = None
        self._first_cmd_time = 0.0
        self._deviation_console_count = 0
        self._in_deviation = False
        self._consecutive_deviation_lockout = 0

        self._safety_lockout = False
        self._last_good_hw_deg = None
        self._consecutive_invalid_reads = 0
        self._startup_candidate = None
        self._startup_confirms = 0

        self._battery_timer = self.create_timer(BATTERY_POLL_SEC, self._poll_battery)
        self._last_battery_v = None

        self.get_logger().info(
            "ZMQ bridge: sub %s, pub %s, host=%s:%d" % (JOINT_COMMANDS_TOPIC, JOINT_STATES_TOPIC, host, port)
        )

    def _joint_commands_cb(self, msg: JointState) -> None:
        """Validate and store the latest command. No ZMQ call here — the command
        is forwarded to hardware in the next _control_loop tick, which ensures
        commands are evenly spaced at the state-publish rate (~20 Hz) instead of
        arriving in bursts that overwhelm the servo firmware."""
        if self._safety_lockout or self._last_good_hw_deg is None:
            return

        if not msg.name or len(msg.position) < len(msg.name):
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        angle_s = []
        for joint_name in SERVO_ORDER:
            if joint_name in name_to_pos:
                rad_val = float(name_to_pos[joint_name])
                if math.isnan(rad_val) or math.isinf(rad_val):
                    return
                deg = rad_to_deg(joint_name, rad_val)
                angle_s.append(deg)
            else:
                self.get_logger().warn("Command missing joint %s, skipping" % joint_name)
                return
        if len(angle_s) != 6:
            return

        self._pending_cmd_degs = list(angle_s)

    def _forward_pending_command(self) -> None:
        """Forward the latest stored command to hardware via ZMQ."""
        if self._pending_cmd_degs is None:
            return

        now = time.time()
        if now < self._cmd_lockout_until:
            return

        cmd = self._pending_cmd_degs
        self._pending_cmd_degs = None

        try:
            self._zmq_client.set_joint_position_array(cmd)
            self._consecutive_cmd_fails = 0
            self._last_cmd_deg = list(cmd)
            self._last_cmd_send_time = now
            self._cmd_count += 1
            if self._cmd_count == 1:
                self._first_cmd_time = now
                self.get_logger().info("First command forwarded to hardware")
            deg_fmt = ", ".join("%.1f°" % d for d in cmd)
            self._flog.debug("CMD_FWD[%d]: deg=[%s]" % (self._cmd_count, deg_fmt))
        except Exception as e:
            self._consecutive_cmd_fails += 1
            self._flog.error(
                "ZMQ set_joint_position_array failed (%d consecutive): %s"
                % (self._consecutive_cmd_fails, e)
            )
            if self._consecutive_cmd_fails >= WATCHDOG_FAIL_THRESHOLD:
                self._cmd_lockout_until = now + WATCHDOG_LOCKOUT_SEC
                err = (
                    "WATCHDOG: %d consecutive command failures — "
                    "refusing new commands for %.0fs. "
                    "Check ZMQ service / Orin connectivity."
                    % (self._consecutive_cmd_fails, WATCHDOG_LOCKOUT_SEC)
                )
                self.get_logger().error(err)
                self._flog.error(err)

    def _trigger_safety_lockout(self, reason: str) -> None:
        """Hard-stop: refuse all further commands and state publishing."""
        if self._safety_lockout:
            return
        self._safety_lockout = True
        self.get_logger().error("SAFETY LOCKOUT: %s — all motion halted, restart node to resume" % reason)
        self._flog.critical("SAFETY LOCKOUT TRIGGERED: %s" % reason)

    def _publish_degs(self, degs: list) -> None:
        """Convert hardware degree values to radians and publish JointState."""
        positions_rad_by_name = {}
        for i, joint_name in enumerate(SERVO_ORDER):
            positions_rad_by_name[joint_name] = deg_to_rad(joint_name, degs[i])
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

        self._state_log_count += 1
        if self._state_log_count == 1 or self._state_log_count % 40 == 0:
            rad_fmt = ", ".join("%.4f" % p for p in positions_out[:6])
            deg_fmt = ", ".join("%.1f°" % d for d in degs)
            self._flog.debug(
                "STATE[%d]: hw_deg=[%s] pub_rad=[%s]"
                % (self._state_log_count, deg_fmt, rad_fmt)
            )

    def _validate_startup_reading(self, degs: list) -> bool:
        """Require multiple consistent reads before accepting the first baseline.
        Returns True when baseline is established (and sets _last_good_hw_deg).
        Returns False while still collecting confirmation reads."""
        if self._startup_candidate is None:
            self._startup_candidate = list(degs)
            self._startup_confirms = 0
            self._flog.info(
                "Startup: candidate baseline %s" % [round(d, 1) for d in degs]
            )
            return False

        worst_delta = 0.0
        worst_idx = 0
        for i in range(len(degs)):
            delta = abs(degs[i] - self._startup_candidate[i])
            if delta > worst_delta:
                worst_delta = delta
                worst_idx = i

        if worst_delta <= MAX_PLAUSIBLE_JUMP_DEG:
            self._startup_confirms += 1
            if self._startup_confirms >= STARTUP_CONFIRM_READS:
                self._last_good_hw_deg = list(degs)
                self._consecutive_invalid_reads = 0
                self.get_logger().info(
                    "Servo baseline locked (%d consistent reads)"
                    % (self._startup_confirms + 1)
                )
                self._flog.info(
                    "Startup: baseline confirmed: %s"
                    % [round(d, 1) for d in degs]
                )
                return True
            return False

        self._consecutive_invalid_reads += 1
        self._flog.warning(
            "Startup: candidate rejected (%.1f° delta on %s), "
            "new candidate: %s"
            % (worst_delta, SERVO_ORDER[worst_idx],
               [round(d, 1) for d in degs])
        )
        if self._consecutive_invalid_reads <= 3:
            self.get_logger().warn(
                "Startup: unstable servo readings (%.1f° jitter), recalibrating"
                % worst_delta
            )
        self._startup_candidate = list(degs)
        self._startup_confirms = 0
        if self._consecutive_invalid_reads >= INVALID_POS_LOCKOUT_COUNT * 2:
            self._trigger_safety_lockout(
                "Cannot establish baseline: %d inconsistent startup reads"
                % self._consecutive_invalid_reads
            )
        return False

    def _control_loop(self) -> None:
        """Combined command-forward + state-read cycle.

        Runs at state_publish_hz (~20 Hz).  Each tick:
          1. Forward pending command to hardware (fast ZMQ write, ~5 ms)
          2. Read actual servo positions (ZMQ + serial read, ~35 ms)
          3. Publish joint state

        This ensures commands are evenly spaced at the control loop rate,
        giving the servo firmware time to complete each interpolation before
        receiving the next target.
        """
        if self._safety_lockout:
            if self._last_good_hw_deg is not None:
                self._publish_degs(self._last_good_hw_deg)
            return

        self._forward_pending_command()

        try:
            degs = self._zmq_client.get_joint_position_array()
            self._consecutive_state_fails = 0
        except Exception as e:
            self._consecutive_state_fails += 1
            self._flog.error(
                "ZMQ get_joint_position_array failed (%d consecutive): %s"
                % (self._consecutive_state_fails, e)
            )
            if self._consecutive_state_fails == WATCHDOG_FAIL_THRESHOLD:
                msg = (
                    "WATCHDOG: %d consecutive state-read failures — "
                    "ZMQ service may be down. Joint states will be stale."
                    % self._consecutive_state_fails
                )
                self.get_logger().error(msg)
                self._flog.error(msg)
            return
        if len(degs) != 6:
            return

        reject_reason = None
        n_neg = sum(1 for d in degs if d < 0)
        if n_neg > 0:
            reject_reason = "negative servo values (no servo response)"

        if reject_reason is None:
            for i, jn in enumerate(SERVO_ORDER):
                m = SERVO_MAP.get(jn)
                if m and not (m["min_deg"] - 5.0 <= degs[i] <= m["max_deg"] + 5.0):
                    reject_reason = (
                        "%s=%.1f° outside servo range [%.0f, %.0f]"
                        % (jn, degs[i], m["min_deg"], m["max_deg"])
                    )
                    break

        if reject_reason is None and self._last_good_hw_deg is None:
            if not self._validate_startup_reading(degs):
                return

        if reject_reason is None and self._last_good_hw_deg is not None:
            worst_jump = 0.0
            worst_idx = 0
            for i in range(len(degs)):
                jump = abs(degs[i] - self._last_good_hw_deg[i])
                if jump > worst_jump:
                    worst_jump = jump
                    worst_idx = i
            if worst_jump > MAX_PLAUSIBLE_JUMP_DEG:
                # When the control loop is blocked by a serial read timeout,
                # servos continue moving toward the last commanded position.
                # A large jump from the last-good reading is legitimate if
                # the new reading is close to where we told the servo to go.
                near_cmd = False
                if self._last_cmd_deg is not None:
                    near_cmd = True
                    for i in range(len(degs)):
                        if abs(degs[i] - self._last_good_hw_deg[i]) > MAX_PLAUSIBLE_JUMP_DEG:
                            if abs(degs[i] - self._last_cmd_deg[i]) > MAX_PLAUSIBLE_JUMP_DEG:
                                near_cmd = False
                                break
                if near_cmd:
                    self._flog.info(
                        "Large move on %s (%.1f°) accepted — near command target "
                        "(hw=%.1f°, cmd=%.1f°, prev=%.1f°)"
                        % (SERVO_ORDER[worst_idx], worst_jump,
                           degs[worst_idx], self._last_cmd_deg[worst_idx],
                           self._last_good_hw_deg[worst_idx])
                    )
                else:
                    reject_reason = (
                        "implausible %.1f° jump on %s (hw=%.1f°, prev=%.1f°)"
                        % (worst_jump, SERVO_ORDER[worst_idx],
                           degs[worst_idx], self._last_good_hw_deg[worst_idx])
                    )

        if reject_reason is not None:
            self._consecutive_invalid_reads += 1
            self._flog.error(
                "Bad HW read #%d: %s | raw=%s"
                % (self._consecutive_invalid_reads, reject_reason,
                   [round(d, 1) for d in degs])
            )
            if self._consecutive_invalid_reads <= 3:
                self.get_logger().error("Bad servo read: %s" % reject_reason)
            if self._consecutive_invalid_reads >= INVALID_POS_LOCKOUT_COUNT:
                self._trigger_safety_lockout(
                    "%d consecutive bad servo reads" % self._consecutive_invalid_reads
                )
            if self._last_good_hw_deg is not None:
                degs = list(self._last_good_hw_deg)
            else:
                return
        else:
            self._consecutive_invalid_reads = 0
            self._last_good_hw_deg = list(degs)

        self._publish_degs(degs)
        self._check_position_deviation(degs)

    def _check_position_deviation(self, hw_degs) -> None:
        """Compare actual servo positions against last commanded position.

        During active trajectory execution, servos naturally lag behind the
        advancing command target.  Relaxed thresholds avoid false alarms while
        the safety lockout (30°) still catches genuinely stuck servos.
        """
        if self._last_cmd_deg is None:
            return

        traj_active = (time.time() - self._last_cmd_send_time) < TRAJECTORY_ACTIVE_TIMEOUT_SEC
        warn_deg = DEVIATION_ACTIVE_WARN_DEG if traj_active else DEVIATION_WARN_DEG
        crit_deg = DEVIATION_ACTIVE_CRIT_DEG if traj_active else DEVIATION_CRIT_DEG

        deviations = []
        max_dev = 0.0
        worst_idx = 0
        for i, joint_name in enumerate(SERVO_ORDER):
            dev = abs(hw_degs[i] - self._last_cmd_deg[i])
            deviations.append(dev)
            if dev > max_dev:
                max_dev = dev
                worst_idx = i
        if max_dev < warn_deg:
            if self._in_deviation:
                self._in_deviation = False
                self._deviation_console_count = 0
                self._consecutive_deviation_lockout = 0
                self._flog.info("DEVIATION CLEARED: servos tracking normally again")
            return

        self._in_deviation = True
        bad = [
            "%s: hw=%.1f° cmd=%.1f° (Δ%.1f°)"
            % (SERVO_ORDER[i], hw_degs[i], self._last_cmd_deg[i], deviations[i])
            for i in range(len(SERVO_ORDER))
            if deviations[i] >= warn_deg
        ]
        detail = "; ".join(bad)
        if max_dev >= crit_deg:
            self._flog.critical(
                "SERVO DEVIATION [%.1f°]: %s | full_hw=[%s] full_cmd=[%s]"
                % (
                    max_dev, detail,
                    ", ".join("%.1f°" % d for d in hw_degs),
                    ", ".join("%.1f°" % d for d in self._last_cmd_deg),
                )
            )
        else:
            self._flog.warning("SERVO DEVIATION [%.1f°]: %s" % (max_dev, detail))

        self._deviation_console_count += 1
        if self._deviation_console_count <= 3:
            self.get_logger().error(
                "SERVO DEVIATION: %s off by %.1f° — check servo power/mechanical binding"
                % (SERVO_ORDER[worst_idx], max_dev)
            )

        in_grace = (
            self._first_cmd_time > 0
            and (time.time() - self._first_cmd_time) < STARTUP_GRACE_SEC
        )
        if max_dev >= DEVIATION_LOCKOUT_DEG:
            if in_grace:
                self._flog.warning(
                    "STARTUP GRACE: ignoring %.1f° deviation on %s (%.1fs into grace period)"
                    % (max_dev, SERVO_ORDER[worst_idx],
                       time.time() - self._first_cmd_time)
                )
            else:
                self._consecutive_deviation_lockout += 1
                if self._consecutive_deviation_lockout >= DEVIATION_LOCKOUT_COUNT:
                    self._trigger_safety_lockout(
                        "%s deviation %.1f° for %d consecutive reads"
                        % (SERVO_ORDER[worst_idx], max_dev, self._consecutive_deviation_lockout)
                    )
        else:
            self._consecutive_deviation_lockout = 0

    def _poll_battery(self) -> None:
        """Low-frequency battery voltage check (every BATTERY_POLL_SEC)."""
        try:
            v = self._zmq_client.get_battery_voltage()
        except Exception:
            self._flog.debug("Battery poll failed (ZMQ error)")
            return
        if v < 0.1:
            return
        self._last_battery_v = v
        self._flog.debug("BATTERY: %.1fV" % v)
        if v <= BATTERY_CRIT_V:
            self.get_logger().error(
                "BATTERY CRITICAL: %.1fV — risk of servo brownout, power off or connect charger" % v
            )
            self._flog.critical("BATTERY CRITICAL: %.1fV" % v)
        elif v <= BATTERY_WARN_V:
            self.get_logger().warn("BATTERY LOW: %.1fV — consider charging" % v)
            self._flog.warning("BATTERY LOW: %.1fV" % v)

    def destroy_node(self) -> None:
        try:
            self._zmq_client.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> int:
    host = os.environ.get("ZMQ_HOST", "192.168.31.142")
    port = int(os.environ.get("ZMQ_PORT", "5555"))
    ensure_zmq_connection(host, port, log_fn=print)

    rclpy.init(args=args)
    node = OpusX3PlusZMQBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
