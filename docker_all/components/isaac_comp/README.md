# isaac_comp — Isaac Sim 5.1 (self-contained)

Runs **Isaac Sim 5.1** with the X3plus arm + gripper demo and ROS2 bridge. This component is **self-contained**: all code and resources live under `app/` and it does not depend on `ros2_comp` at build time.

## Layout

- **Dockerfile** — Base image `nvcr.io/nvidia/isaac-sim:5.1.0`; copies `app/` into the container.
- **app/** — Application tree:
  - **app/monty_demo/** — Python package for the Isaac demo.
  - **app/monty_demo/x3plus_isaac_arm_demo.py** — Entry point (`python -m monty_demo.x3plus_isaac_arm_demo`).
  - **app/monty_demo/x3plus_isaac/** — X3plus URDF and meshes (arm + gripper) used by Isaac Sim.

## Build and run

From **docker_all** (build context is docker_all so `COPY components/isaac_comp/app` works):

```bash
docker compose build isaac_comp
docker compose up isaac_comp
```

To use ROS2 control, run **ros2_comp** as well (same network, same `ROS_DOMAIN_ID`); it runs the bringup that talks to `/x3plus/joint_states` and `/x3plus/joint_commands`.

## Overriding the base image

```bash
docker compose build --build-arg ISAAC_BASE=nvcr.io/nvidia/isaac-sim:5.0.0 isaac_comp
```

## Optional: ISAAC_SIM_VENV

If the container uses a different venv path for the ROS2 bridge rclpy, set:

```bash
docker compose run -e ISAAC_SIM_VENV=/path/to/venv isaac_comp
```
