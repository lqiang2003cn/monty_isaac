# monty_demo

Demo ROS2 Python package for the monty_isaac workspace.

## Build

**Docker (recommended):** Build and run happen inside the `ros2_comp` container. From `docker_all/`: `./scripts/build_all.sh` then `./scripts/run_compose.sh`. See `docker_all/README.md`.

**Local workspace:** From the workspace root that contains this package in `src/monty_demo` (e.g. `docker_all` with package at `components/ros2_comp/app/monty_demo` symlinked or copied to `src/monty_demo`, or a colcon workspace that has this package):

```bash
colcon build --packages-select monty_demo
source install/setup.bash
```

## Run

**Talker** (publishes on `chatter`):

```bash
ros2 run monty_demo talker
```

**Listener** (subscribes to `chatter`):

```bash
ros2 run monty_demo listener
```

Run both in separate terminals to see messages flow.
