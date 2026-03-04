# syntax=docker/dockerfile:1
# ros2_comp: ROS 2 Jazzy component for monty_demo.
# Build context: docker_all/  (docker compose build from docker_all).
# Multi-stage: build tools stay in the build stage; runtime image is leaner.
# Isaac URDF is generated from the single-source xacro at build time.

# ---------------------------------------------------------------------------
# Stage 1 — build
# ---------------------------------------------------------------------------
FROM ros:jazzy AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-controller-manager \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-joint-state-topic-hardware-interface \
        ros-jazzy-xacro \
        ros-jazzy-moveit \
        ros-jazzy-trac-ik-kinematics-plugin \
        python3-colcon-common-extensions \
        python3-rosdep

WORKDIR /ws

COPY components/ros2_comp/app/monty_demo /ws/src/monty_demo
# Meshes: single source is shared/x3plus_isaac/meshes/ (overlays any local copy)
COPY shared/x3plus_isaac/meshes /ws/src/monty_demo/x3plus_robot/meshes

RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select monty_demo

# Generate plain Isaac URDF from the single-source xacro (baked into image)
RUN . /opt/ros/jazzy/setup.sh \
    && mkdir -p /shared/x3plus_isaac/urdf \
    && xacro /ws/src/monty_demo/x3plus_robot/urdf/x3plus.urdf.xacro \
         include_ros2_control:=false \
         mesh_prefix:=../meshes \
         robot_name:=x3plus_isaac \
       > /shared/x3plus_isaac/urdf/x3plus_isaac.urdf

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM ros:jazzy

LABEL maintainer="monty" \
      description="ROS 2 Jazzy component for monty_demo"

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-controller-manager \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-joint-state-topic-hardware-interface \
        ros-jazzy-xacro \
        ros-jazzy-moveit \
        ros-jazzy-trac-ik-kinematics-plugin

WORKDIR /workspace

COPY --from=build /ws/install /workspace/install
COPY --from=build /shared/x3plus_isaac/urdf/x3plus_isaac.urdf /shared/x3plus_isaac/urdf/x3plus_isaac.urdf

# Source ROS2 + workspace in every new shell (docker exec ros2_comp bash and bash -l -c)
RUN echo 'source /opt/ros/jazzy/setup.bash 2>/dev/null || true' > /etc/profile.d/ros2_workspace.sh \
    && echo 'test -f /workspace/install/setup.bash && source /workspace/install/setup.bash' >> /etc/profile.d/ros2_workspace.sh \
    && echo '\n# ROS2 + workspace (monty_demo)' >> /etc/bash.bashrc \
    && echo 'source /opt/ros/jazzy/setup.bash 2>/dev/null || true' >> /etc/bash.bashrc \
    && echo 'test -f /workspace/install/setup.bash && source /workspace/install/setup.bash' >> /etc/bash.bashrc

COPY --chmod=755 components/ros2_comp/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "monty_demo", "opus_x3plus_bringup.launch.py"]
