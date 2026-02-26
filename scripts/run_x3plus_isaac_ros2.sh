#!/usr/bin/env bash
# (Optional) Start Isaac Sim in background, wait for bridge, then start bringup in one script.
# Prefer two terminals:
#   Terminal 1: ./scripts/start_isaac.sh
#   Terminal 2: ./scripts/start_ros2_bringup.sh
#
# This script: Isaac in background -> wait for "ROS2 Bridge extension enabled" -> bringup.
# Usage: source scripts/rebuild_and_source.sh  then  ./scripts/run_x3plus_isaac_ros2.sh
# Ctrl+C stops bringup and the Isaac Sim process.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ISAAC_VENV="${ISAAC_VENV:-$HOME/isaacsim_venv}"
WAIT_MARKER="ROS2 Bridge extension enabled"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
LOG_FILE="${LOG_FILE:-/tmp/x3plus_isaac_ros2_demo.log}"

# Cleanup: kill Isaac Sim process we started
ISAAC_PID=""
cleanup() {
  if [[ -n "$ISAAC_PID" ]] && kill -0 "$ISAAC_PID" 2>/dev/null; then
    echo "Stopping Isaac Sim (PID $ISAAC_PID)..."
    kill "$ISAAC_PID" 2>/dev/null || true
    wait "$ISAAC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -d "$ISAAC_VENV" ]] || [[ ! -x "$ISAAC_VENV/bin/python" ]]; then
  echo "Error: Isaac Sim venv not found at $ISAAC_VENV. Set ISAAC_VENV or install Isaac Sim."
  exit 1
fi

echo "=== Starting Isaac Sim (sim + bridge) in background ==="
# Run under 'script' so Isaac Sim gets a PTY; many GUI apps exit immediately when stdout is not a TTY.
# script -q suppresses its own messages and writes child output to LOG_FILE.
: > "$LOG_FILE"
script -q -c "PYTHONUNBUFFERED=1 \"$ISAAC_VENV/bin/python\" -m monty_demo.x3plus_isaac_arm_demo" "$LOG_FILE" &
ISAAC_PID=$!
echo "Isaac Sim PID: $ISAAC_PID (log: $LOG_FILE)"

# If the process exits before the marker, show the log and fail.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
    echo "Isaac Sim exited early. Last 80 lines of log:"
    echo "---"
    tail -n 80 "$LOG_FILE" 2>/dev/null || cat "$LOG_FILE"
    exit 1
  fi
  sleep 1
done

# Event-driven wait: tail -f streams new lines; grep -m 1 exits as soon as the line appears (no polling).
echo "Waiting for '$WAIT_MARKER' (timeout ${WAIT_TIMEOUT}s)..."
if timeout "$WAIT_TIMEOUT" tail -f "$LOG_FILE" 2>/dev/null | grep -m 1 -q "$WAIT_MARKER"; then
  echo "Bridge ready."
else
  EXIT=$?
  if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
    echo "Isaac Sim exited before bridge ready. Last 80 lines of log:"
    echo "---"
    tail -n 80 "$LOG_FILE" 2>/dev/null || cat "$LOG_FILE"
  elif [[ $EXIT -eq 124 ]]; then
    echo "Timeout waiting for '$WAIT_MARKER'. Check $LOG_FILE"
  else
    echo "Wait failed (exit $EXIT). Check $LOG_FILE"
  fi
  exit 1
fi

# Give sim a moment to start publishing
sleep 3

echo "=== Starting ROS2 bringup ==="
source /opt/ros/jazzy/setup.bash
if [[ ! -f "$REPO_ROOT/install/setup.bash" ]]; then
  echo "Error: install/setup.bash not found. Run: source scripts/rebuild_and_source.sh"
  exit 1
fi
source "$REPO_ROOT/install/setup.bash"
exec ros2 launch monty_demo x3plus_bringup.launch.py
