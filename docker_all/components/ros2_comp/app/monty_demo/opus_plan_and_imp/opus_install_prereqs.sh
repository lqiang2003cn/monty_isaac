#!/usr/bin/env bash
# Install all prerequisites for X3plus unified ros2_control (Isaac Sim + real robot).
# Run from repo root or from this directory. Uses sudo for apt.
# Usage: chmod +x opus_install_prereqs.sh && ./opus_install_prereqs.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# monty_demo lives at docker_all/components/ros2_comp/app/monty_demo; workspace root may be docker_all or repo root
MONTY_DEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MONTY_DEMO_DIR/../.." 2>/dev/null && pwd)"
if [[ -z "$REPO_ROOT" ]] || [[ ! -f "$MONTY_DEMO_DIR/package.xml" ]]; then
  REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"

echo "=== X3plus opus prerequisites installer ==="
echo "Workspace root: $REPO_ROOT"
echo ""

FAILED=()
OK=()

# --- ROS 2 Jazzy (must be installed first for apt packages) ---
if ! source /opt/ros/jazzy/setup.bash 2>/dev/null; then
  echo "[SKIP] ROS 2 Jazzy not found at /opt/ros/jazzy. Install it first (e.g. ros-jazzy-desktop)."
  FAILED+=("ros-jazzy (base)")
else
  echo "[OK] ROS 2 Jazzy sourced"
  OK+=("ros-jazzy")
fi

# --- Apt packages ---
APT_PACKAGES=(
  ros-jazzy-ros2-control
  ros-jazzy-ros2-controllers
  ros-jazzy-controller-manager
  ros-jazzy-joint-state-topic-hardware-interface
  ros-jazzy-xacro
  ros-jazzy-robot-state-publisher
  ros-jazzy-joint-state-publisher-gui
  python3-colcon-common-extensions
)

for pkg in "${APT_PACKAGES[@]}"; do
  if dpkg -l "$pkg" &>/dev/null; then
    echo "[OK] $pkg (already installed)"
    OK+=("$pkg")
  else
    echo "Installing $pkg..."
    if sudo apt-get install -y "$pkg"; then
      echo "[OK] $pkg"
      OK+=("$pkg")
    else
      echo "[FAIL] $pkg"
      FAILED+=("$pkg")
    fi
  fi
done

# --- Summary ---
echo ""
echo "=== Summary ==="
echo "Installed/OK: ${#OK[@]} items"
echo "Failed: ${#FAILED[@]} items"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed items: ${FAILED[*]}"
  echo "Fix failures above (e.g. install ROS Jazzy first, then re-run this script)."
  exit 1
fi
echo "Prerequisites done. With Docker: from docker_all run ./scripts/build_all.sh and ./scripts/run_compose.sh. Otherwise: source /opt/ros/jazzy/setup.bash && colcon build --packages-select monty_demo && source install/setup.bash"
echo "Please make sure to manually install Rosmaster lib before running on real robot."
