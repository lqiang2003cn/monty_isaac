# ============================================
# Isaac Sim 5.1.0 + ROS2 Jazzy Development Container
# Based on official NVIDIA Isaac Sim container (Ubuntu 24.04)
# ============================================

# Use official NVIDIA Isaac Sim 5.1.0 container as base
# Requires: docker login nvcr.io (with NGC API key)
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Switch to root for installation
USER root

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set up timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ============================================
# 1. Install ROS2 Jazzy (for Ubuntu 24.04)
# ============================================

# Install base packages needed for ROS2
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    ca-certificates \
    git \
    vim \
    nano \
    build-essential \
    cmake \
    sudo \
    locales \
    pkg-config \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Jazzy Desktop
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-ros-base \
    ros-dev-tools \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update --rosdistro jazzy

# Additional ROS2 packages commonly used with Isaac
RUN apt-get update && apt-get install -y \
    ros-jazzy-vision-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-compressed-image-transport \
    ros-jazzy-diagnostic-updater \
    ros-jazzy-rviz2 \
    ros-jazzy-rqt* \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-xacro \
    ros-jazzy-urdf \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-nav2-bringup \
    ros-jazzy-navigation2 \
    ros-jazzy-slam-toolbox \
    ros-jazzy-moveit \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-image-geometry \
    ros-jazzy-depth-image-proc \
    ros-jazzy-stereo-image-proc \
    ros-jazzy-camera-info-manager \
    ros-jazzy-launch-testing-ament-cmake \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 2. Isaac ROS Dependencies
# ============================================

RUN apt-get update && apt-get install -y \
    libbullet-dev \
    libopencv-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    libboost-all-dev \
    libpcl-dev \
    libeigen3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages for Isaac ROS (use pip from Isaac Sim's Python)
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    opencv-python \
    scipy \
    pyyaml \
    trimesh \
    transforms3d \
    flask \
    networkx \
    defusedxml

# ============================================
# 3. Create Isaac ROS Workspace
# ============================================

RUN mkdir -p /workspaces/isaac_ros_ws/src
WORKDIR /workspaces/isaac_ros_ws/src

# Clone Isaac ROS packages
RUN git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection.git && \
    git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_depth_segmentation.git || true

# Clone custom robot packages
RUN git clone --depth 1 https://github.com/lqiang2003cn/lqtech_ros2_x3plus.git || true

# Install dependencies via rosdep
WORKDIR /workspaces/isaac_ros_ws
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y" || true

# Build Isaac ROS workspace (some packages may fail without full CUDA setup)
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-ignore isaac_ros_nitros isaac_ros_nitros_interfaces || true"

# ============================================
# 4. User Setup
# ============================================

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

# Create user - use existing group if GID exists, or create new one
RUN if ! getent group $USER_GID > /dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME; \
    fi && \
    if ! id -u $USER_UID > /dev/null 2>&1; then \
        useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME; \
    else \
        # User with this UID exists, just ensure home dir and sudoers \
        existing_user=$(getent passwd $USER_UID | cut -d: -f1); \
        if [ "$existing_user" != "$USERNAME" ]; then \
            usermod -l $USERNAME -d /home/$USERNAME -m $existing_user 2>/dev/null || true; \
        fi; \
    fi && \
    mkdir -p /home/$USERNAME && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Set up environment for the user
RUN echo "# ROS2 Jazzy" >> /home/$USERNAME/.bashrc && \
    echo "source /opt/ros/jazzy/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "" >> /home/$USERNAME/.bashrc && \
    echo "# Isaac ROS Workspace" >> /home/$USERNAME/.bashrc && \
    echo "source /workspaces/isaac_ros_ws/install/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc && \
    echo "" >> /home/$USERNAME/.bashrc && \
    echo "# ROS2 Environment" >> /home/$USERNAME/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> /home/$USERNAME/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> /home/$USERNAME/.bashrc && \
    echo "" >> /home/$USERNAME/.bashrc && \
    echo "# Isaac Sim 5.1.0" >> /home/$USERNAME/.bashrc && \
    echo "export ACCEPT_EULA=Y" >> /home/$USERNAME/.bashrc && \
    echo "export PRIVACY_CONSENT=Y" >> /home/$USERNAME/.bashrc && \
    echo "alias isaacsim='/isaac-sim/isaac-sim.sh'" >> /home/$USERNAME/.bashrc && \
    echo "alias python-isaac='/isaac-sim/python.sh'" >> /home/$USERNAME/.bashrc && \
    echo "export PATH=/isaac-sim/kit/python/bin:\$PATH" >> /home/$USERNAME/.bashrc && \
    echo "" >> /home/$USERNAME/.bashrc && \
    echo "# Helpful aliases" >> /home/$USERNAME/.bashrc && \
    echo "alias cb='cd /workspaces/isaac_ros_ws && colcon build --symlink-install'" >> /home/$USERNAME/.bashrc && \
    echo "alias cbs='cd /workspaces/isaac_ros_ws && colcon build --symlink-install --packages-select'" >> /home/$USERNAME/.bashrc && \
    echo "alias src='source /workspaces/isaac_ros_ws/install/setup.bash'" >> /home/$USERNAME/.bashrc

# Give ownership of workspace to user (use UID directly to avoid name issues)
RUN chown -R $USER_UID:$USER_GID /workspaces && \
    chown -R $USER_UID:$USER_GID /home/$USERNAME

# Make Isaac Sim accessible to all users and create cache directories
RUN chmod -R a+rX /isaac-sim && \
    chmod a+x /isaac-sim/*.sh /isaac-sim/kit/*.sh /isaac-sim/kit/python/bin/python3* && \
    mkdir -p /isaac-sim/kit/cache/Kit/107.3/69cbf6ad && \
    chmod -R 777 /isaac-sim/kit/cache && \
    mkdir -p /home/$USERNAME/Documents/Kit/shared && \
    chown -R $USER_UID:$USER_GID /home/$USERNAME/Documents

# Initialize rosdep for the developer user
USER $USERNAME
RUN rosdep update --rosdistro jazzy

# ============================================
# 5. Final Setup
# ============================================

USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /workspaces/isaac_ros_ws
USER $USERNAME

CMD ["/bin/bash"]
