#!/usr/bin/env bash
# Copy x3plus robot description (URDF + meshes) from repo src into docker_all so that
# both Isaac Sim and ROS2 control use the same model.
#
# Usage:
#   ./scripts/copy_x3plus_from_src.sh [SOURCE_DIR]
#
# SOURCE_DIR: path to the x3plus description package (default: repo root's src/ or src/lqtech_ros2_x3plus).
#   Expected layout: SOURCE_DIR/urdf/ and SOURCE_DIR/meshes/ (e.g. lqtech_ros2_x3plus or similar).
#
# Copies:
#   - urdf/* -> components/ros2_comp/app/monty_demo/x3plus_robot/urdf/
#   - meshes/* -> components/ros2_comp/app/monty_demo/x3plus_robot/meshes/
#   - meshes/* -> components/ros2_comp/app/monty_demo/monty_demo/x3plus_isaac/meshes/
# After copying, you may need to replace package://lqtech_ros2_x3plus/meshes with package://monty_demo/meshes
# in the xacro/urdf if the source uses lqtech_ros2_x3plus (setup already uses monty_demo in docker_all).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ALL="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCKER_ALL/../.." && pwd)"
MONTY_DEMO="$DOCKER_ALL/components/ros2_comp/app/monty_demo"
X3PLUS_ROBOT="$MONTY_DEMO/x3plus_robot"
X3PLUS_ISAAC_MESHES="$MONTY_DEMO/monty_demo/x3plus_isaac/meshes"

if [[ -n "$1" ]]; then
    SRC="$(cd "$1" && pwd)"
else
    # Default: repo src or src/lqtech_ros2_x3plus
    if [[ -d "$REPO_ROOT/src/lqtech_ros2_x3plus" ]]; then
        SRC="$REPO_ROOT/src/lqtech_ros2_x3plus"
    elif [[ -d "$REPO_ROOT/src/urdf" && -d "$REPO_ROOT/src/meshes" ]]; then
        SRC="$REPO_ROOT/src"
    else
        echo "Usage: $0 [SOURCE_DIR]" >&2
        echo "SOURCE_DIR must contain urdf/ and meshes/ (e.g. lqtech_ros2_x3plus package)." >&2
        echo "Default: \$REPO_ROOT/src or \$REPO_ROOT/src/lqtech_ros2_x3plus" >&2
        echo "Put the x3plus description package in src/ then run this script." >&2
        exit 1
    fi
fi

if [[ ! -d "$SRC/urdf" || ! -d "$SRC/meshes" ]]; then
    echo "Source $SRC must contain urdf/ and meshes/." >&2
    exit 1
fi

echo "Copying x3plus description from: $SRC"
echo "  -> $X3PLUS_ROBOT/urdf and $X3PLUS_ROBOT/meshes"
echo "  -> $X3PLUS_ISAAC_MESHES"

mkdir -p "$X3PLUS_ROBOT/urdf" "$X3PLUS_ROBOT/meshes" "$X3PLUS_ISAAC_MESHES"
cp -r "$SRC/urdf/"* "$X3PLUS_ROBOT/urdf/"
cp -r "$SRC/meshes/"* "$X3PLUS_ROBOT/meshes/"
cp -r "$SRC/meshes/"* "$X3PLUS_ISAAC_MESHES/"

echo "Done. If source URDF/xacro use package://lqtech_ros2_x3plus/meshes, run:"
echo "  sed -i 's|package://lqtech_ros2_x3plus/meshes/|package://monty_demo/meshes/|g' $X3PLUS_ROBOT/urdf/*.xacro $X3PLUS_ROBOT/urdf/*.urdf"
echo "(Already applied in current docker_all tree.)"
