# docker_all — Component-based Docker layout

All source and Docker wiring for the monty_isaac stack live here. Each runtime component (Isaac Sim bridge, ROS2 bringup, etc.) is a separate image; they share a network and communicate via ROS2 (and optionally ZMQ or other protocols).

## Layout

- **components/** — One folder per Docker component. Each has a `Dockerfile` and optionally `docker-compose.fragment.yml`.
  - **isaac_comp** — Isaac Sim 5.1 + ROS2 bridge (X3plus arm + gripper). **Self-contained**: all code and resources under `components/isaac_comp/app/` (no dependency on ros2_comp at build time).
  - **ros2_comp** — ROS2 workspace with `monty_demo`; runs `ros2 launch monty_demo x3plus_bringup.launch.py`.
  - **_template** — Scaffold for adding new components (copy, rename, implement).
- **shared/** — Shared config (e.g. `ros2/` for Fast DDS profile) and scripts.
- **scripts/** — Host scripts: `build_all.sh`, `run_compose.sh`, `copy_x3plus_from_src.sh`.

**X3plus URDF and meshes** live under `components/ros2_comp/app/monty_demo/`:
- **x3plus_robot/** — URDF/xacro and meshes for ROS2 (robot_state_publisher, ros2_control). Installed to `share/monty_demo/urdf` and `share/monty_demo/meshes` (package URL: `package://monty_demo/meshes/...`).
- **monty_demo/x3plus_isaac/** — URDF and meshes for Isaac Sim (relative `../meshes/`). **isaac_comp** has its own copy under `components/isaac_comp/app/monty_demo/x3plus_isaac/` and does not depend on ros2_comp.

To refresh the description from the repo’s `src/`, run `./scripts/copy_x3plus_from_src.sh [SOURCE_DIR]`. Put the x3plus package (e.g. `lqtech_ros2_x3plus` with `urdf/` and `meshes/`) in `src/` or pass its path.

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

**Requirements:** Docker with Compose v2, and for `isaac_comp` an NVIDIA GPU and the NVIDIA Container Toolkit. The default `isaac_comp` Dockerfile uses **Isaac Sim 5.1** (`nvcr.io/nvidia/isaac-sim:5.1.0`); override with `--build-arg ISAAC_BASE=...` if needed.

**Non-Docker use:** The root-level `scripts/` (e.g. `start_isaac.sh`, `start_ros2_bringup.sh`) remain for running Isaac and ROS2 bringup on the host; colcon and `install/setup.bash` then refer to a workspace that contains the package (e.g. after moving source, you would build from a workspace that has `monty_demo` under `src/` or symlinked from `docker_all/components/ros2_comp/app/monty_demo`).

## Adding a new component

1. Copy `components/_template/` to `components/<new_name>/` (e.g. `components/zmq_service/`).
2. Implement your code in `components/<new_name>/app/` and adjust the `Dockerfile` (COPY app, set CMD/ENTRYPOINT).
3. In `docker-compose.fragment.yml`, set the service name, build context, env, and any ports (e.g. ZMQ). Attach the service to the same network: `networks: [monty_net]`.
4. Add that service to `docker_all/docker-compose.yml` (paste the fragment under `services:` or use Compose `include`).
5. **ROS2:** Use the same `ROS_DOMAIN_ID` (and optional `shared/ros2/fastdds_profile.xml`) so the new component discovers other ROS2 nodes. **ZMQ:** Expose a port and connect from other containers to `tcp://<service_name>:<port>`.

See `components/_template/README.md` for a short checklist.
