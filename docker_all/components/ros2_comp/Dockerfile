# Build context: docker_all (repo root or docker_all/)
# So COPY paths are relative to that context.
FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

# Install ROS 2 packages required by monty_demo (from package.xml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-controller-manager \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-joint-state-topic-hardware-interface \
    ros-jazzy-xacro \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy monty_demo into standard colcon src layout
COPY components/ros2_comp/app/monty_demo /workspace/src/monty_demo

# Build the package
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select monty_demo

COPY components/ros2_comp/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "monty_demo", "x3plus_bringup.launch.py"]
