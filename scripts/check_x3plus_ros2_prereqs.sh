#!/usr/bin/env bash
# Check prerequisites for X3plus Isaac arm ROS2 control demo.
# Run from repo root: ./scripts/check_x3plus_ros2_prereqs.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== X3plus ROS2 control prerequisites ==="
ALL_OK=true

# 1. Isaac Sim venv
if [[ -d "$HOME/isaacsim_venv" ]] && [[ -x "$HOME/isaacsim_venv/bin/python" ]]; then
  echo "[OK] Isaac Sim venv: $HOME/isaacsim_venv"
else
  echo "[--] Isaac Sim venv: not found at ~/isaacsim_venv (install Isaac Sim and create venv)"
  ALL_OK=false
fi

# 2. ROS2 Jazzy
if source /opt/ros/jazzy/setup.bash 2>/dev/null; then
  echo "[OK] ROS2 Jazzy: /opt/ros/jazzy"
else
  echo "[--] ROS2 Jazzy: not found (install ros-jazzy-desktop or similar)"
  ALL_OK=false
fi

# 3. xacro (required by bringup launch to expand URDF)
if source /opt/ros/jazzy/setup.bash 2>/dev/null; then
  if command -v xacro &>/dev/null; then
    echo "[OK] xacro on PATH (ros-jazzy-xacro)"
  else
    echo "[!!] Install: sudo apt-get install -y ros-jazzy-xacro"
    ALL_OK=false
  fi
fi

# 4. Workspace built + launch available
if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  source "$REPO_ROOT/install/setup.bash" 2>/dev/null || true
  if ros2 pkg prefix monty_demo &>/dev/null && [[ -f "$(ros2 pkg prefix monty_demo)/share/monty_demo/launch/x3plus_bringup.launch.py" ]]; then
    echo "[OK] Workspace built; monty_demo launch installed"
  else
    echo "[--] Run from repo root: source /opt/ros/jazzy/setup.bash && colcon build --packages-select monty_demo"
    ALL_OK=false
  fi
else
  echo "[--] install/setup.bash not found; run colcon build from repo root"
  ALL_OK=false
fi

# 5. controller_manager + ros2_controllers (bringup launch)
if source /opt/ros/jazzy/setup.bash 2>/dev/null; then
  if ros2 pkg prefix controller_manager &>/dev/null; then
    echo "[OK] controller_manager (ros-jazzy-controller-manager)"
  else
    echo "[!!] Install: sudo apt-get install -y ros-jazzy-ros2-control ros-jazzy-ros2-controllers"
    ALL_OK=false
  fi
fi

# 6. Joint-state-topic hardware interface (requires apt)
if source /opt/ros/jazzy/setup.bash 2>/dev/null; then
  if ros2 pkg prefix joint_state_topic_hardware_interface &>/dev/null; then
    echo "[OK] ros-jazzy-joint-state-topic-hardware-interface installed"
  else
    echo "[!!] Install: sudo apt-get install -y ros-jazzy-joint-state-topic-hardware-interface"
    ALL_OK=false
  fi
fi

# 7. Meshes for x3plus_isaac

MESH_DIR="$REPO_ROOT/src/monty_demo/monty_demo/x3plus_isaac/meshes"
if [[ -d "$MESH_DIR/X3plus" ]] && [[ -d "$MESH_DIR/sensor" ]]; then
  echo "[OK] x3plus_isaac meshes: $MESH_DIR (X3plus, sensor)"
else
  echo "[--] Meshes: copy or symlink into $MESH_DIR (see src/monty_demo/x3plus_isaac/meshes/README.md)"
  ALL_OK=false
fi

echo "=== Done ==="
if $ALL_OK; then
  echo "All prerequisites met. Run: (1) python -m monty_demo.x3plus_isaac_arm_demo  (2) ros2 launch monty_demo x3plus_bringup.launch.py"
else
  echo "Fix the items above, then re-run this script."
  exit 1
fi
