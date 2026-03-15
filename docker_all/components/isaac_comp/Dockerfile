# syntax=docker/dockerfile:1
# isaac_comp: Isaac Sim component with X3plus demo.
# Build context: docker_all/  (docker compose build from docker_all).
# URDF is generated from the single-source xacro at build time (no manual step).
# Meshes are mounted at runtime from shared/x3plus_isaac/meshes/.
#
# Base image is an explicit stage (isaac-base) pinned by digest so BuildKit caches it:
# after the first build, "FROM isaac-base" reuses that stage and does not re-pull ~7.5GB.
# Use normal rebuilds (no --no-cache, no docker builder prune) so the cache is kept.

# ---------------------------------------------------------------------------
# Stage 1 — Isaac base (cached after first pull; do not use --no-cache on rebuild)
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54 AS isaac-base

# ---------------------------------------------------------------------------
# Stage 2 — generate plain Isaac URDF from the single-source xacro
# ---------------------------------------------------------------------------
FROM ros:jazzy AS urdf-gen

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends ros-jazzy-xacro

COPY components/ros2_comp/app/monty_demo/x3plus_robot/urdf/ /urdf/
RUN . /opt/ros/jazzy/setup.sh \
    && xacro /urdf/x3plus.urdf.xacro \
         include_ros2_control:=false \
         mesh_prefix:=../meshes \
         robot_name:=x3plus_isaac \
       > /x3plus_isaac.urdf

# ---------------------------------------------------------------------------
# Stage 3 — Isaac Sim runtime (FROM internal stage so no re-pull on rebuild)
# ---------------------------------------------------------------------------
FROM isaac-base

LABEL maintainer="monty" \
      description="Isaac Sim 5.1 component for monty_demo"

WORKDIR /isaac-sim

COPY --from=urdf-gen /x3plus_isaac.urdf /shared/x3plus_isaac/urdf/x3plus_isaac.urdf
COPY shared/x3plus_config/init_positions.yaml /shared/x3plus_config/init_positions.yaml
COPY components/isaac_comp/app /app

ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Clear the base image's entrypoint so CMD is used directly
ENTRYPOINT []
CMD ["./python.sh", "-m", "monty_demo.x3plus_isaac_arm_demo"]
