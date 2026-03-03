#!/usr/bin/env bash
# Run "real" profile (zmq_bridge_comp + ros2_comp). If REMOTE_ORIN_HOST is set,
# (1) start local registry if REGISTRY_HOST set, (2) build remote_zmq_service for
# linux/arm64 and push to local registry, (3) sync compose to Orin, (4) Orin pulls
# and runs the container. No Docker Hub on Orin required.
#
# Usage (from repo root or docker_all):
#   ./scripts/real_up.sh                  # no args = up --build (start everything, logs in terminal)
#   ./scripts/real_up.sh up -d --build    # detached (no logs in terminal)
#
# Env (optional):
#   REMOTE_ORIN_HOST     default wheeltec@192.168.31.142; override if different
#   REGISTRY_HOST        this machine's LAN IP so Orin can pull (default: auto-detect); set to use local registry
#   REMOTE_ORIN_SSH_PASSWORD  optional; if set, use sshpass for non-interactive SSH (install: apt install sshpass)
#   REMOTE_ORIN_REPO_PATH   path on Orin to sync into (default: ~/monty_isaac); docker_all is synced under it
#   REMOTE_ORIN_SERIAL_DEVICE  serial device on Orin (default: /dev/ttyUSB0)
#   SKIP_REMOTE_BUILD    set to 1 to skip remote sync/build and only run local compose
#   REMOTE_BUILD_ON_ORIN set to 1 to build on Orin instead (needs Docker Hub reachable on Orin)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ALL="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCKER_ALL/.." && pwd)"

REMOTE_HOST="${REMOTE_ORIN_HOST:-wheeltec@192.168.31.142}"
REMOTE_PASSWORD="${REMOTE_ORIN_SSH_PASSWORD:-}"
REMOTE_REPO="${REMOTE_ORIN_REPO_PATH:-~/monty_isaac}"
REMOTE_SERIAL="${REMOTE_ORIN_SERIAL_DEVICE:-/dev/ttyUSB0}"
# This machine's IP that the Orin can reach (for local registry). Auto-detect if unset.
REGISTRY_HOST="${REGISTRY_HOST:-$(hostname -I 2>/dev/null | awk '{print $1}' || true)}"
REGISTRY_PORT="${REGISTRY_PORT:-5000}"

run_ssh_cmd() {
  local cmd="$1"
  if [[ -n "$REMOTE_PASSWORD" ]] && command -v sshpass &>/dev/null; then
    sshpass -p "$REMOTE_PASSWORD" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash -lc '$cmd'"
  else
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash -lc '$cmd'"
  fi
}

run_remote_build() {
  if [[ -z "$REMOTE_HOST" ]]; then
    return 0
  fi
  if [[ -n "${SKIP_REMOTE_BUILD:-}" ]] && [[ "${SKIP_REMOTE_BUILD}" != "0" ]]; then
    echo "[real_up] SKIP_REMOTE_BUILD set, skipping remote sync/build on $REMOTE_HOST"
    return 0
  fi

  rsync_opts=(-az -e "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no")
  if [[ -n "$REMOTE_PASSWORD" ]] && command -v sshpass &>/dev/null; then
    export SSHPASS="$REMOTE_PASSWORD"
    rsync_opts=(-az -e "sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no")
  elif [[ -n "$REMOTE_PASSWORD" ]]; then
    echo "[real_up] WARNING: REMOTE_ORIN_SSH_PASSWORD set but sshpass not found. rsync/ssh may prompt for password." >&2
  fi

  if [[ -n "${REMOTE_BUILD_ON_ORIN:-}" ]] && [[ "${REMOTE_BUILD_ON_ORIN}" != "0" ]]; then
    # Build on Orin: sync full docker_all for build context
    echo "[real_up] Syncing docker_all to $REMOTE_HOST:$REMOTE_REPO/docker_all (full, for build) ..."
    rsync_opts_build=("${rsync_opts[@]}" --delete)
    if ! rsync "${rsync_opts_build[@]}" "$DOCKER_ALL/" "$REMOTE_HOST:$REMOTE_REPO/docker_all/"; then
      echo "[real_up] WARNING: rsync to $REMOTE_HOST failed. Continuing with local compose; ZMQ may be unreachable." >&2
      return 0
    fi
    echo "[real_up] Sync done."
    # Build on Orin (requires Docker Hub reachable on Orin)
    echo "[real_up] Building and starting remote_zmq_service on $REMOTE_HOST ..."
    remote_cmd="cd $REMOTE_REPO/docker_all && docker compose -f components/remote_zmq_service/docker-compose.remote.yml up -d --build --force-recreate"
    if ! run_ssh_cmd "$remote_cmd"; then
      echo "[real_up] WARNING: Remote build/start on $REMOTE_HOST failed. Continuing with local compose; ZMQ may be unreachable." >&2
    else
      echo "[real_up] Remote build/start on $REMOTE_HOST completed."
    fi
    return 0
  fi

  # 1) Ensure local registry is running (Orin will pull from REGISTRY_HOST:5000)
  if [[ -z "$REGISTRY_HOST" ]]; then
    echo "[real_up] WARNING: REGISTRY_HOST not set (could not auto-detect). Set it to your machine's LAN IP and run: $SCRIPT_DIR/start_local_registry.sh" >&2
    return 0
  fi
  if ! docker inspect monty_registry &>/dev/null; then
    echo "[real_up] Starting local registry (monty_registry) ..."
    "$SCRIPT_DIR/start_local_registry.sh" || true
  else
    docker start monty_registry 2>/dev/null || true
  fi
  REGISTRY_IMAGE="$REGISTRY_HOST:$REGISTRY_PORT/monty_remote_zmq_service:latest"

  # 2) Build image for linux/arm64, load into host, then push (host daemon has insecure-registries; buildx --push uses builder's client which doesn't)
  echo "[real_up] Building remote_zmq_service for linux/arm64 ..."
  if ! docker buildx version &>/dev/null; then
    echo "[real_up] WARNING: docker buildx not found. Install Docker Buildx or set REMOTE_BUILD_ON_ORIN=1 to build on Orin." >&2
    return 0
  fi
  if ! docker buildx inspect monty_arm64 &>/dev/null; then
    echo "[real_up] Creating buildx builder monty_arm64 for linux/arm64 ..."
    docker buildx create --name monty_arm64 --driver docker-container --platform linux/arm64 --use 2>/dev/null || true
  fi
  docker buildx use monty_arm64 2>/dev/null || docker buildx use default 2>/dev/null || true
  if ! docker buildx build --platform linux/arm64 \
    -t "$REGISTRY_IMAGE" \
    -f "$DOCKER_ALL/components/remote_zmq_service/Dockerfile" \
    "$DOCKER_ALL" \
    --load; then
    echo "[real_up] WARNING: Build failed. Ensure buildx builder monty_arm64 exists (driver docker-container)." >&2
    return 0
  fi
  echo "[real_up] Pushing to $REGISTRY_IMAGE (host Docker; ensure insecure-registries in daemon.json) ..."
  if ! docker push "$REGISTRY_IMAGE"; then
    echo "[real_up] WARNING: Push to $REGISTRY_IMAGE failed. Add $REGISTRY_HOST:$REGISTRY_PORT to insecure-registries in /etc/docker/daemon.json and restart Docker." >&2
    return 0
  fi
  echo "[real_up] Pushed to $REGISTRY_IMAGE"

  # 3) Sync compose file to Orin
  echo "[real_up] Syncing remote compose file to $REMOTE_HOST:$REMOTE_REPO/docker_all ..."
  REMOTE_COMPOSE_DIR="components/remote_zmq_service"
  if ! run_ssh_cmd "mkdir -p $REMOTE_REPO/docker_all/$REMOTE_COMPOSE_DIR"; then
    echo "[real_up] WARNING: Could not create remote dir. Continuing with local compose; ZMQ may be unreachable." >&2
    return 0
  fi
  if ! rsync "${rsync_opts[@]}" "$DOCKER_ALL/$REMOTE_COMPOSE_DIR/docker-compose.remote.yml" "$REMOTE_HOST:$REMOTE_REPO/docker_all/$REMOTE_COMPOSE_DIR/"; then
    echo "[real_up] WARNING: rsync compose file to $REMOTE_HOST failed. Continuing with local compose; ZMQ may be unreachable." >&2
    return 0
  fi

  # 4) On Orin: pull from registry and start container (REMOTE_ZMQ_IMAGE so compose uses registry image)
  remote_cmd="docker pull $REGISTRY_IMAGE && cd $REMOTE_REPO/docker_all && REMOTE_ZMQ_IMAGE=$REGISTRY_IMAGE docker compose -f components/remote_zmq_service/docker-compose.remote.yml up -d --force-recreate"
  if ! run_ssh_cmd "$remote_cmd"; then
    echo "[real_up] WARNING: Remote pull/start on $REMOTE_HOST failed. Ensure Orin has $REGISTRY_HOST:$REGISTRY_PORT in insecure-registries (see docker_all/REMOTE_SETUP.md)." >&2
  else
    echo "[real_up] Remote remote_zmq_service on $REMOTE_HOST started (pulled from $REGISTRY_IMAGE)."
  fi
}

run_remote_build
cd "$DOCKER_ALL"
# If no args (or only empty arg), default to "up --build" so logs stream to terminal
if [[ $# -eq 0 ]] || { [[ $# -eq 1 ]] && [[ -z "${1:-}" ]]; }; then
  echo "[real_up] No arguments: starting local stack (docker compose --profile real up --build). Logs will stream below."
  set -- up --build
fi
exec docker compose --profile real "$@"
