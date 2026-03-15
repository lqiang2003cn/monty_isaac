# docker_all — Component-based Docker layout

All source and Docker wiring for the monty_isaac stack live here. Each runtime component (Isaac Sim bridge, ROS2 bringup, etc.) is a separate image; they share a network and communicate via ROS2.

## Layout

- **components/** — One folder per Docker component. Each has a `Dockerfile` and a `compose.yml`.
  - **isaac_comp** — Isaac Sim 5.1 + ROS2 bridge (X3plus arm + gripper). Uses **shared/x3plus_isaac** (mounted at `/shared`).
  - **ros2_comp** — ROS2 workspace with `monty_demo`; runs `ros2 launch monty_demo opus_x3plus_bringup.launch.py`. Supports `mode:=isaac` (sim), `mode:=real` (USB serial), and `mode:=zmq` (remote robot via ZMQ).
  - **remote_real_x3plus** — ZMQ server + x3plus serial driver; **deploy on the robot machine** (e.g. NVIDIA Orin Nano, ARM64). Run in Docker with `--device /dev/ttyUSB0`. See `components/remote_real_x3plus/README.md` for build/run on Orin.
  - **vision_comp** — Vision model server (SAM2, Grounding DINO). **Before first build:** `git submodule update --init --recursive` so `components/vision_comp/sam2_src` and `groundingdino_src` are populated.
  - **_template** — Scaffold for adding new components (copy, rename, implement).
- **shared/** — **x3plus_isaac** (URDF + meshes for Isaac Sim) is the single source here; both isaac_comp and ros2_comp use it (mounted at `/shared` in Docker). Optional: `ros2/` for Fast DDS profile.
- **scripts/** — Host scripts: `build_all.sh`, `run_compose.sh`, `copy_x3plus_from_src.sh`.

**X3plus URDF and meshes:**
- **shared/x3plus_isaac/** — **Single source** for Isaac Sim: `urdf/x3plus_isaac.urdf` and `meshes/`. Mounted at `/shared` for isaac_comp and ros2_comp. Use `X3PLUS_DESCRIPTION_DIR` to override (e.g. `/shared/x3plus_isaac`).
- **components/ros2_comp/.../x3plus_robot/** — URDF/xacro and meshes for ROS2 (robot_state_publisher, ros2_control). Installed to `share/monty_demo/urdf` and `share/monty_demo/meshes` (package URL: `package://monty_demo/meshes/...`).

To refresh the description from the repo's `src/`, run `./scripts/copy_x3plus_from_src.sh [SOURCE_DIR]`. Put the x3plus package (e.g. `lqtech_ros2_x3plus` with `urdf/` and `meshes/`) in `src/` or pass its path.

## Build and run

From **docker_all**:

```bash
./scripts/build_all.sh
./scripts/run_compose.sh
```

Or from the repo root:

```bash
./docker_all/scripts/build_all.sh
./docker_all/scripts/run_compose.sh
```

Or call Docker Compose directly from `docker_all`:

```bash
cd docker_all
docker compose build
docker compose up
```

**vision_comp and large downloads:** The vision_comp build is defined once in the **root** compose (`context: components/vision_comp`) so the same context is always used from `docker_all` and the layer cache is reused. Ensure **BuildKit** is on (e.g. `DOCKER_BUILDKIT=1` in docker_all/.env) so the pip cache mount works; then `docker compose build vision_comp` from docker_all should not re-download torch after the first build.

**Remote ZMQ on Orin Nano (ARM64):** Use the wrapper script. By default it (1) **rsyncs** `docker_all/` to the Orin, (2) **builds** `remote_real_x3plus` for `linux/arm64` locally (Docker Buildx) and streams the image to the Orin (no Docker Hub pull on Orin), (3) starts the container on the Orin, (4) runs `docker compose` locally with `ROBOT_MODE=zmq`. From `docker_all` or repo root:

```bash
# From docker_all
REMOTE_ORIN_HOST=wheeltec@192.168.31.142 ./scripts/real_up.sh up --build

# From repo root
REMOTE_ORIN_HOST=wheeltec@192.168.31.142 ./docker_all/scripts/real_up.sh up --build

# With SSH password (non-interactive); install sshpass on host: apt install sshpass
REMOTE_ORIN_HOST=wheeltec@192.168.31.142 REMOTE_ORIN_SSH_PASSWORD=yourpassword ./scripts/real_up.sh up --build
```

Optional env: `REMOTE_ORIN_REPO_PATH` (sync destination on Orin, default `~/monty_isaac`), `SKIP_REMOTE_BUILD=1` to skip sync and remote deploy. If the Orin can reach Docker Hub, set `REMOTE_BUILD_ON_ORIN=1` to build the image on the Orin instead of local cross-build.

**Requirements:** Docker with Compose v2, and for `isaac_comp` an NVIDIA GPU and the NVIDIA Container Toolkit. The default `isaac_comp` Dockerfile uses **Isaac Sim 5.1** (`nvcr.io/nvidia/isaac-sim:5.1.0`); override with `--build-arg ISAAC_BASE=...` if needed.

**Non-Docker use:** The root-level `scripts/` (e.g. `start_isaac.sh`, `start_ros2_bringup.sh`) remain for running Isaac and ROS2 bringup on the host; colcon and `install/setup.bash` then refer to a workspace that contains the package (e.g. after moving source, you would build from a workspace that has `monty_demo` under `src/` or symlinked from `docker_all/components/ros2_comp/app/monty_demo`).

## Adding a new component

1. Copy `components/_template/` to `components/<new_name>/` (e.g. `components/zmq_service/`).
2. Implement your code in `components/<new_name>/app/` and adjust the `Dockerfile` (COPY app, set CMD/ENTRYPOINT).
3. Create a `compose.yml` in the component directory with the service definition. Attach to network `monty_net`.
4. Add an `include` entry in the root `docker_all/compose.yml` pointing to the new `compose.yml`.
5. **ROS2:** Use the same `ROS_DOMAIN_ID` (and optional `shared/ros2/fastdds_profile.xml`) so the new component discovers other ROS2 nodes. **ZMQ:** Expose a port and connect from other containers to `tcp://<service_name>:<port>`.

See `components/_template/README.md` for a short checklist.
