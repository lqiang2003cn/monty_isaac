# isaac_comp: Isaac Sim 5.1 component with X3plus demo.
# Build context: docker_all (e.g. docker compose build from docker_all).
# Robot description (URDF + meshes) is mounted at runtime from shared/x3plus_isaac via docker-compose.
ARG ISAAC_BASE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_BASE}

# Demo script only; URDF + meshes come from the shared/ volume mount at /shared/x3plus_isaac
COPY components/isaac_comp/app /app

ENV PYTHONPATH=/app

WORKDIR /isaac-sim

ENTRYPOINT []
CMD ["./python.sh", "-m", "monty_demo.x3plus_isaac_arm_demo"]
