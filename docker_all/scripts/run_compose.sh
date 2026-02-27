#!/usr/bin/env bash
# Run the stack (isaac_comp + ros2_comp). Run from docker_all:
#   ./scripts/run_compose.sh
# or from repo root:
#   ./docker_all/scripts/run_compose.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ALL="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DOCKER_ALL"
docker compose up "$@"
