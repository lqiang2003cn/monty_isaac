# syntax=docker/dockerfile:1
# isaac_comp: Isaac Sim component with X3plus demo.
# Build context: docker_all/  (docker compose build from docker_all).
# Robot description (URDF + meshes) is mounted at runtime via docker-compose volumes.

ARG ISAAC_BASE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_BASE}

LABEL maintainer="monty" \
      description="Isaac Sim 5.1 component for monty_demo"

WORKDIR /isaac-sim

# Demo script only — URDF + meshes come from the /shared volume mount
COPY components/isaac_comp/app /app

ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Clear the base image's entrypoint so CMD is used directly
ENTRYPOINT []
CMD ["./python.sh", "-m", "monty_demo.x3plus_isaac_arm_demo"]
