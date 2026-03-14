#!/usr/bin/env bash
# Start the simulation stack (isaac_comp + ros2_comp with MoveIt + planner).
# The planner's go_home_on_start parameter (default true) will automatically
# move the arm to the home position once MoveIt and joint states are ready.
#
# Usage (from repo root or docker_all):
#   ./scripts/sim_up.sh                  # no args = up --build
#   ./scripts/sim_up.sh up -d --build    # detached
#
# Environment variables (override before calling):
#   USE_BT=true   — launch the BT pick-place executor (needed for bt_pick_place_sim_test)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ALL="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DOCKER_ALL"

export USE_MOVEIT=true
export USE_BT="${USE_BT:-true}"

if [[ $# -eq 0 ]] || { [[ $# -eq 1 ]] && [[ -z "${1:-}" ]]; }; then
  echo "[sim_up] No arguments: starting sim stack (docker compose --profile sim up --build)."
  echo "[sim_up] USE_MOVEIT=$USE_MOVEIT  USE_BT=$USE_BT"
  set -- up --build
fi
echo "[sim_up] go_home will execute automatically once the planner and MoveIt are ready."
exec docker compose --profile sim "$@"
