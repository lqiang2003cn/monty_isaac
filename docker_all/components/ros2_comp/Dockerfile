# syntax=docker/dockerfile:1
# ros2_comp: ROS 2 Jazzy component for monty_demo.
# Build context: docker_all/  (docker compose build from docker_all).
# Multi-stage: build tools stay in the build stage; runtime image is leaner.

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
        python3-colcon-common-extensions \
        python3-rosdep

WORKDIR /ws

COPY components/ros2_comp/app/monty_demo /ws/src/monty_demo

RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select monty_demo

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
        ros-jazzy-xacro

WORKDIR /workspace

COPY --from=build /ws/install /workspace/install

COPY --chmod=755 components/ros2_comp/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "monty_demo", "x3plus_bringup.launch.py"]
