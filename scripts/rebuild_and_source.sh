#!/usr/bin/env bash
# Rebuild monty_demo and source the workspace. Needed when you change code/launch/config/urdf.
# Run before start_ros2_bringup.sh (terminal 2). Not required before start_isaac (terminal 1).
#
# Usage (source so env applies to current shell):
#   source scripts/rebuild_and_source.sh
# Then in terminal 2: ./scripts/start_ros2_bringup.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Rebuilding monty_demo ==="
source /opt/ros/jazzy/setup.bash
colcon build --packages-select monty_demo

echo "=== Sourcing workspace ==="
source "$REPO_ROOT/install/setup.bash"

echo "Done. Workspace is sourced in this shell."
echo "Run: ./scripts/run_x3plus_isaac_ros2.sh"
