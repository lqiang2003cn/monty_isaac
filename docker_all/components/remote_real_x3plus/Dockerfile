# Remote real x3plus service: runs on the machine with serial (e.g. Orin Nano).
# Build context is docker_all when building from repo root: docker build -f components/remote_real_x3plus/Dockerfile .
# When cross-building for arm64 from x86, --platform ensures base image and RUN use arm64 (avoids exec format error).
ARG TARGETPLATFORM=linux/arm64
FROM --platform=$TARGETPLATFORM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Serial port access (optional: udev for non-root; we document --device for Docker)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire app/ dir so any new files under app/ are included without editing this Dockerfile
COPY components/remote_real_x3plus/app /app

RUN pip install --no-cache-dir -r requirements.txt

# ZMQ_PORT and SERIAL_DEVICE can override defaults (5555, /dev/ttyUSB0)
# Run with: docker run --device /dev/ttyUSB0 -p 5555:5555 -e ZMQ_PORT=5555 ...
CMD ["python3", "-u", "/app/lqtech_zmq_service.py"]
