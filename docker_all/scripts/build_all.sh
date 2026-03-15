#!/usr/bin/env bash
# Build all component images. Run from docker_all:
#   ./scripts/build_all.sh
# or from repo root:
#   ./docker_all/scripts/build_all.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ALL="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DOCKER_ALL"
# BuildKit enables cache mounts (e.g. vision_comp pip cache so torch isn't re-downloaded)
export DOCKER_BUILDKIT=1
docker compose build
