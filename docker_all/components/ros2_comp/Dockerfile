# syntax=docker/dockerfile:1
# ros2_comp: ROS 2 Jazzy component for monty_demo.
# Build context: docker_all/  (docker compose build from docker_all).
# Multi-stage: build tools stay in the build stage; runtime image is leaner.
# Isaac URDF is generated from the single-source xacro at build time.

# ---------------------------------------------------------------------------
# Stage 1 — git deps (independent stage: never invalidated by apt changes)
# ---------------------------------------------------------------------------
FROM alpine/git AS git-deps
RUN git clone --depth 1 https://github.com/BehaviorTree/BehaviorTree.ROS2.git \
        /repos/BehaviorTree.ROS2

# ---------------------------------------------------------------------------
# Stage 2 — shared base (apt packages installed ONCE, used by build+runtime)
# ---------------------------------------------------------------------------
FROM ros:jazzy AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-controller-manager \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-joint-state-topic-hardware-interface \
        ros-jazzy-xacro \
        ros-jazzy-moveit \
        ros-jazzy-trac-ik-kinematics-plugin \
        ros-jazzy-behaviortree-cpp \
        ros-jazzy-image-transport \
        ros-jazzy-image-publisher \
        ros-jazzy-camera-info-manager \
        ros-jazzy-cv-bridge \
        ros-jazzy-realsense2-camera \
        ros-jazzy-rviz2 \
        python3-colcon-common-extensions \
        python3-matplotlib \
        python3-rosdep \
        python3-zmq \
        libgflags-dev \
        nlohmann-json3-dev \
        libgoogle-glog-dev \
        git

# ---------------------------------------------------------------------------
# Stage 3 — build
# ---------------------------------------------------------------------------
FROM base AS build

WORKDIR /ws

# External repos from the cached git-deps stage (survives apt-get changes)
COPY --from=git-deps /repos/BehaviorTree.ROS2 /ws/src/BehaviorTree.ROS2

# Build external deps first — this layer is cached until git tags or apt
# packages change.  realsense2_camera is installed via apt (no source build).
RUN --mount=type=cache,target=/ws/build,id=colcon-build \
    . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-up-to \
        btcpp_ros2_interfaces behaviortree_ros2

# Local source — placed after external-deps build so edits here only
# trigger the lightweight local rebuild below.
COPY components/ros2_comp/app/monty_demo /ws/src/monty_demo
COPY shared/x3plus_isaac/meshes /ws/src/monty_demo/x3plus_robot/meshes
COPY shared/x3plus_config/init_positions.yaml /shared/x3plus_config/init_positions.yaml
COPY components/ros2_comp/app/monty_bt /ws/src/monty_bt

# Build local packages only.  The cache mount keeps colcon's build/ tree
# across rebuilds so even a full layer re-run is an incremental compile.
RUN --mount=type=cache,target=/ws/build,id=colcon-build \
    . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select monty_bt monty_demo

# Generate plain Isaac URDF from the single-source xacro (baked into image)
RUN . /opt/ros/jazzy/setup.sh \
    && mkdir -p /shared/x3plus_isaac/urdf \
    && xacro /ws/src/monty_demo/x3plus_robot/urdf/x3plus.urdf.xacro \
         include_ros2_control:=false \
         mesh_prefix:=../meshes \
         robot_name:=x3plus_isaac \
       > /shared/x3plus_isaac/urdf/x3plus_isaac.urdf

# ---------------------------------------------------------------------------
# Stage 4 — runtime (inherits base: all apt packages already installed)
# ---------------------------------------------------------------------------
FROM base

LABEL maintainer="monty" \
      description="ROS 2 Jazzy component for monty_demo"

WORKDIR /workspace

COPY --from=build /ws/install /workspace/install
COPY --from=build /shared/x3plus_isaac/urdf/x3plus_isaac.urdf /shared/x3plus_isaac/urdf/x3plus_isaac.urdf
COPY --from=build /shared/x3plus_config/init_positions.yaml /shared/x3plus_config/init_positions.yaml

# Source ROS2 + workspace in every new shell (docker exec ros2_comp bash and bash -l -c)
RUN echo 'source /opt/ros/jazzy/setup.bash 2>/dev/null || true' > /etc/profile.d/ros2_workspace.sh \
    && echo 'test -f /workspace/install/setup.bash && source /workspace/install/setup.bash' >> /etc/profile.d/ros2_workspace.sh \
    && echo '\n# ROS2 + workspace (monty_demo)' >> /etc/bash.bashrc \
    && echo 'source /opt/ros/jazzy/setup.bash 2>/dev/null || true' >> /etc/bash.bashrc \
    && echo 'test -f /workspace/install/setup.bash && source /workspace/install/setup.bash' >> /etc/bash.bashrc

COPY --chmod=755 components/ros2_comp/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "monty_demo", "opus_x3plus_bringup.launch.py"]
