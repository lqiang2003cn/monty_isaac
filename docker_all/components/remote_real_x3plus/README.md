# remote_real_x3plus

ZMQ server that runs **on the robot machine** (e.g. NVIDIA Orin Nano) and talks to the x3plus over serial. The host machine runs `ros2_comp` with `mode:=zmq`, which connects to this service over the network.

## Contents

- **lqtech_zmq_service.py** — ZMQ REP server (get/set joint positions via JSON).
- **x3plus_serial.py** — Rosmaster serial protocol driver for x3plus.
- **zmq_protocol.py** — Shared protocol constants (keep in sync with `ros2_comp` at `monty_demo/opus_plan_and_imp/zmq_protocol.py`).

## Target: NVIDIA Orin Nano (ARM64)

Orin Nano is **aarch64/ARM64**. Docker images built on x86 are not directly runnable; use one of:

1. **Build on the Orin** (recommended): clone/copy this repo onto the Orin and build there. The image will be native ARM64.
2. **Cross-build from x86**: `docker buildx build --platform linux/arm64 -t monty_remote_real_x3plus:latest -f docker_all/components/remote_real_x3plus/Dockerfile docker_all` then transfer image (save/load or push to a registry and pull on Orin).

## Build (on the Orin, from repo root)

From the **monty_isaac** repo root (so `docker_all` is the build context):

```bash
cd /path/to/monty_isaac
docker build -f docker_all/components/remote_real_x3plus/Dockerfile -t monty_remote_real_x3plus:latest docker_all
```

Or from inside `docker_all`:

```bash
cd docker_all
docker build -f components/remote_real_x3plus/Dockerfile -t monty_remote_real_x3plus:latest .
```

## Run (on the Orin)

Serial device must be passed into the container. Ensure the host can access the serial port (e.g. `sudo chmod a+rw /dev/ttyUSB0` or add user to `dialout`).

```bash
docker run -d --restart unless-stopped \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -p 5555:5555 \
  -e ZMQ_PORT=5555 \
  -e SERIAL_DEVICE=/dev/ttyUSB0 \
  --name remote_real_x3plus \
  monty_remote_real_x3plus:latest
```

If your serial port has another name (e.g. `/dev/ttyACM0`), set `SERIAL_DEVICE` and pass that device:

```bash
docker run -d --restart unless-stopped \
  --device /dev/ttyACM0:/dev/ttyACM0 \
  -p 5555:5555 \
  -e SERIAL_DEVICE=/dev/ttyACM0 \
  --name remote_real_x3plus \
  monty_remote_real_x3plus:latest
```

## Modify and redeploy

1. Edit sources under `docker_all/components/remote_real_x3plus/app/` (e.g. `lqtech_zmq_service.py`, `x3plus_serial.py`).
2. Rebuild the image on the Orin (same build commands as above).
3. Restart the container:
   ```bash
   docker stop remote_real_x3plus
   docker rm remote_real_x3plus
   docker run -d ...  # same run command as above
   ```
   Or use the optional `docker-compose.remote.yml` (see below) and run `docker-compose -f docker_all/components/remote_real_x3plus/docker-compose.remote.yml up -d --build --force-recreate` so the running container is replaced by the new image.

## Optional: docker-compose on the Orin

Copy `docker-compose.remote.yml` to the Orin (or the whole component) and run from the directory that contains the build context:

```bash
# From monty_isaac repo root
docker-compose -f docker_all/components/remote_real_x3plus/docker-compose.remote.yml --project-directory docker_all up -d --build --force-recreate
```

This builds and runs the service with device and port mapping.

## Protocol

Same JSON protocol as in `ros2_comp`: request `{"method": "getJointPositionArray"}` etc., response `{"joint_array": [...]}` or `{"result": "OK"}` or `{"error": "..."}`. Keep `zmq_protocol.py` in sync with `docker_all/components/ros2_comp/app/monty_demo/monty_demo/opus_plan_and_imp/zmq_protocol.py`.
