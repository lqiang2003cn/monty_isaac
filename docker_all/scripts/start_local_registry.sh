#!/usr/bin/env bash
# Start a local Docker Registry on port 5000 so the Orin can pull images from this machine.
# Run once; the container uses --restart unless-stopped so it auto-starts after reboot. No need to run again.
set -e
if docker inspect monty_registry &>/dev/null; then
  docker start monty_registry 2>/dev/null || true
  echo "Registry monty_registry already exists (will auto-start on reboot)."
else
  docker run -d -p 5000:5000 --restart unless-stopped --name monty_registry registry:2
  echo "Started local registry: monty_registry (port 5000). Runs once; auto-restarts on reboot."
fi
echo "Orin can pull from this host at: $(hostname -I 2>/dev/null | awk '{print $1}'):5000"
echo "Set REGISTRY_HOST to that IP if needed; ensure Orin has it in insecure-registries (see REMOTE_SETUP.md)."
