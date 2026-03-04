#!/usr/bin/env bash
# Generate shared/x3plus_isaac/urdf/x3plus_isaac.urdf from the single-source xacro.
# NOTE: For Docker workflows this is NOT needed — both isaac_comp and ros2_comp
# Dockerfiles generate the URDF at build time. This script is for local / non-Docker use.
# Requires: ros-jazzy-xacro (or any ROS 2 distro with xacro installed).
#
# Usage (run on host from docker_all/):
#   ./scripts/generate_isaac_urdf.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ALL="$(cd "$SCRIPT_DIR/.." && pwd)"

XACRO_SRC="$DOCKER_ALL/components/ros2_comp/app/monty_demo/x3plus_robot/urdf/x3plus.urdf.xacro"
OUT_DIR="$DOCKER_ALL/shared/x3plus_isaac/urdf"
OUT_FILE="$OUT_DIR/x3plus_isaac.urdf"

if ! command -v xacro &>/dev/null; then
    # Try sourcing ROS 2 setup
    for setup in /opt/ros/*/setup.bash; do
        # shellcheck disable=SC1090
        source "$setup" 2>/dev/null && break
    done
fi

if ! command -v xacro &>/dev/null; then
    echo "ERROR: xacro not found. Install ros-<distro>-xacro or run inside a ROS 2 container." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

xacro "$XACRO_SRC" \
    include_ros2_control:=false \
    mesh_prefix:=../meshes \
    robot_name:=x3plus_isaac \
    > "$OUT_FILE"

echo "Generated: $OUT_FILE"
