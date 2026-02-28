# ZMQ bridge component: ROS2 topic bridge to real robot via ZMQ.
# Build context is docker_all when composed from docker-compose.yml.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install Python ZMQ bindings + ssh/sshpass for optional remote service auto-start
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-zmq openssh-client sshpass \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy component app (zmq_bridge_node.py, lqtech_zmq_client.py, zmq_protocol.py, requirements.txt)
COPY components/zmq_bridge_comp/app /app

# Ensure script is executable
RUN chmod +x /app/zmq_bridge_node.py

# Source ROS and run the bridge node
CMD ["bash", "-c", "source /opt/ros/jazzy/setup.bash && exec python3 /app/zmq_bridge_node.py"]
