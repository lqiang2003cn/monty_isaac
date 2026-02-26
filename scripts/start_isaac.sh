#!/usr/bin/env bash
# Start X3plus Isaac Sim (sim + bridge). Run this in terminal 1.
# In terminal 2 run: ./scripts/start_ros2_bringup.sh
#
# No rebuild needed first. You only need the workspace sourced so Python finds monty_demo
# (e.g. you ran "source scripts/rebuild_and_source.sh" at least once, or source install/setup.bash).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ISAAC_VENV="${ISAAC_VENV:-$HOME/isaacsim_venv}"

if [[ ! -d "$ISAAC_VENV" ]] || [[ ! -x "$ISAAC_VENV/bin/python" ]]; then
  echo "Error: Isaac Sim venv not found at $ISAAC_VENV. Set ISAAC_VENV or install Isaac Sim."
  exit 1
fi

if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  source "$REPO_ROOT/install/setup.bash"
fi

echo "=== Starting Isaac Sim (X3plus arm + gripper, ROS2 bridge) ==="
echo "In another terminal run: ./scripts/start_ros2_bringup.sh"
echo ""
export PYTHONUNBUFFERED=1
exec "$ISAAC_VENV/bin/python" -m monty_demo.x3plus_isaac_arm_demo
