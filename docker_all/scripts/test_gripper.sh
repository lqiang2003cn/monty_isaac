#!/usr/bin/env bash
# Diagnostic script: cycle the gripper through several positions and report
# the actual joint state at each step.  Run from docker_all/:
#   bash scripts/test_gripper.sh
#
# Prerequisites: real_up.sh is already running in another terminal.
set -euo pipefail

COMPOSE="docker compose"
EXEC="$COMPOSE exec ros2_comp bash -l -c"

echo "=== Gripper Diagnostic ==="
echo ""

read_grip() {
    $EXEC "ros2 topic echo /x3plus/joint_states --once 2>/dev/null" \
      | awk '/^- grip_joint/{found=1} found && /^- /{idx++} /^position:/{pos=1; next} pos && /^- /{i++; if(i==6) print \$2; if(i>=6) pos=0}'
}

# Simpler: just grab the 6th position value (grip_joint is 6th in the name list)
read_grip_rad() {
    $EXEC "python3 -c \"
import rclpy, sys
from sensor_msgs.msg import JointState
rclpy.init()
node = rclpy.create_node('_grip_reader')
msg_holder = [None]
def cb(m): msg_holder[0] = m
node.create_subscription(JointState, '/x3plus/joint_states', cb, 1)
import time; t0 = time.time()
while msg_holder[0] is None and time.time() - t0 < 5.0:
    rclpy.spin_once(node, timeout_sec=0.1)
if msg_holder[0]:
    idx = msg_holder[0].name.index('grip_joint')
    print(f'{msg_holder[0].position[idx]:.6f}')
else:
    print('TIMEOUT')
node.destroy_node(); rclpy.shutdown()
\""
}

set_and_call() {
    local target=$1
    local label=$2
    echo "--- Step: $label (target_grip=$target rad) ---"
    $EXEC "ros2 param set /x3plus_5dof_planner target_grip $target" 2>&1
    $EXEC "ros2 param set /x3plus_5dof_planner execute true" 2>&1
    $EXEC "ros2 service call /x3plus_5dof_planner/set_gripper std_srvs/srv/Trigger" 2>&1
    sleep 2
    local actual
    actual=$(read_grip_rad)
    echo "  Actual grip_joint after move: $actual rad"
    echo ""
}

echo "1) Reading current gripper position..."
CURRENT=$(read_grip_rad)
echo "   Current grip_joint = $CURRENT rad"
echo ""

echo "2) Closing gripper to -1.0 rad (mostly closed)..."
set_and_call "-1.0" "close to -1.0"

echo "3) Opening gripper to -0.5 rad (halfway)..."
set_and_call "-0.5" "open to -0.5"

echo "4) Opening gripper fully to 0.0 rad..."
set_and_call "0.0" "fully open 0.0"

echo "=== Done. Observe the physical gripper at each step. ==="
echo "If the gripper did not move between steps, check:"
echo "  - ZMQ bridge logs:  cat logs/x3plus_zmq_bridge.log | tail -50"
echo "  - Planner logs:     cat logs/x3plus_planner.log | tail -50"
echo "  - Servo 6 power/wiring on the Orin"
