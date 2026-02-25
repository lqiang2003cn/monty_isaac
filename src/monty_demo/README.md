# monty_demo

Demo ROS2 Python package for the monty_isaac workspace.

## Build

From the workspace root (one level above `src/`):

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
