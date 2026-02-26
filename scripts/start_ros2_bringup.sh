#!/usr/bin/env bash
# Start ROS2 bringup (controller_manager, joint_state_broadcaster, joint_trajectory_controller).
# Run this in terminal 2 after Isaac Sim is running in terminal 1.
#
# Rebuild before this if you changed launch/config/urdf: source scripts/rebuild_and_source.sh
# Then: ./scripts/start_ros2_bringup.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/jazzy/setup.bash
if [[ ! -f "$REPO_ROOT/install/setup.bash" ]]; then
  echo "Error: install/setup.bash not found. Run: source scripts/rebuild_and_source.sh"
  exit 1
fi
source "$REPO_ROOT/install/setup.bash"

echo "=== Starting ROS2 bringup (x3plus_bringup) ==="
exec ros2 launch monty_demo x3plus_bringup.launch.py
